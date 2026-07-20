"""Train a transparent adaptation of the An et al. (2023) mouse-stroke U-Net.

The publication describes the architecture and several training settings, but
the released repository contains inference code only. It also exposes a
three-logit TorchScript model while the paper describes two output classes.
This implementation therefore is not an exact reproduction. It keeps the
published anisotropic kernels, feature counts, pooling, upsampling, lack of
batch normalization, Gaussian-noise augmentation, inverse-frequency weighted
cross entropy, Adam learning rate, and Glorot initialization. It uses a
two-class full-native-field output and an explicitly versioned validation
checkpoint/early-stopping policy for the LYS comparator protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared training split")
    parser.add_argument("--validation", required=True, help="Prepared validation split")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.45)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--metrics-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return train_an2023_unet_adapted(
        train_root=Path(args.input),
        validation_root=Path(args.validation),
        output_root=Path(args.output),
        pretrained_model=Path(args.pretrained_model) if args.pretrained_model else None,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
        noise_std=args.noise_std,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        metrics_threshold=args.metrics_threshold,
        seed=args.seed,
        device_name=args.device,
        amp=not args.no_amp,
        resume=args.resume,
    )


@dataclass(frozen=True)
class VolumeCase:
    case_id: str
    scan_path: Path
    label_path: Path
    scan: np.ndarray
    label: np.ndarray


def build_an2023_adapted_unet(torch: Any) -> Any:
    """Return the two-class, full-field adaptation of the published network."""
    nn = torch.nn

    class An2023AdaptedUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            kernel = (3, 3, 1)
            padding = (1, 1, 0)
            self.enc0 = nn.Sequential(
                nn.Conv3d(1, 8, kernel, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 16, kernel, padding=padding),
                nn.ReLU(inplace=True),
            )
            self.pool0 = nn.MaxPool3d((2, 2, 1))
            self.enc1 = nn.Sequential(
                nn.Conv3d(16, 16, kernel, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 32, kernel, padding=padding),
                nn.ReLU(inplace=True),
            )
            self.pool1 = nn.MaxPool3d((2, 2, 1))
            self.bottom = nn.Sequential(
                nn.Conv3d(32, 32, kernel, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel, padding=padding),
                nn.ReLU(inplace=True),
            )
            self.up1_reduce = nn.Conv3d(32, 16, 1)
            self.dec1 = nn.Sequential(
                nn.Conv3d(48, 16, kernel, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 16, kernel, padding=padding),
                nn.ReLU(inplace=True),
            )
            self.up0_reduce = nn.Conv3d(16, 8, 1)
            self.dec0 = nn.Sequential(
                nn.Conv3d(24, 8, kernel, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 8, kernel, padding=padding),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.Conv3d(8, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 2, 1),
            )

        def forward(self, value: Any) -> Any:
            original_shape = value.shape[2:]
            value, padding_values = _pad_to_multiple_of_four(torch, value)
            skip0 = self.enc0(value)
            skip1 = self.enc1(self.pool0(skip0))
            value = self.bottom(self.pool1(skip1))
            value = torch.nn.functional.interpolate(value, scale_factor=(2, 2, 1), mode="nearest")
            value = torch.relu(self.up1_reduce(value))
            value = self.dec1(torch.cat((value, skip1), dim=1))
            value = torch.nn.functional.interpolate(value, scale_factor=(2, 2, 1), mode="nearest")
            value = torch.relu(self.up0_reduce(value))
            value = self.dec0(torch.cat((value, skip0), dim=1))
            value = self.head(value)
            return _remove_padding(value, padding_values, original_shape)

    model = An2023AdaptedUNet()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 42442:
        raise RuntimeError(f"Unexpected An2023 adapted parameter count: {parameter_count}")
    return model


def train_an2023_unet_adapted(
    *,
    train_root: Path,
    validation_root: Path,
    output_root: Path,
    pretrained_model: Path | None = None,
    epochs: int = 200,
    early_stop_patience: int = 30,
    early_stop_min_delta: float = 0.001,
    learning_rate: float = 1e-4,
    gradient_accumulation: int = 8,
    noise_std: float = 0.45,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    metrics_threshold: float = 0.5,
    seed: int = 20260715,
    device_name: str = "auto",
    amp: bool = True,
    resume: bool = False,
) -> int:
    """Train, restore the best validation-Dice checkpoint, and export probabilities."""
    _validate_hyperparameters(
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        learning_rate=learning_rate,
        gradient_accumulation=gradient_accumulation,
        noise_std=noise_std,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        metrics_threshold=metrics_threshold,
    )
    if not train_root.is_dir() or not validation_root.is_dir():
        raise FileNotFoundError(f"Missing train/validation roots: {train_root}, {validation_root}")
    if train_root.resolve() == validation_root.resolve():
        raise ValueError("Training and validation roots must differ")
    if pretrained_model is not None and not pretrained_model.is_file():
        raise FileNotFoundError(pretrained_model)

    import torch

    _set_seed(seed, torch)
    device = _select_device(torch, device_name)
    use_amp = bool(amp and device.type == "cuda")
    train_cases = _load_cases(
        train_root,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    validation_cases = _load_cases(
        validation_root,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    overlap = sorted(
        {case.case_id for case in train_cases}
        & {case.case_id for case in validation_cases}
    )
    if overlap:
        raise ValueError(f"Training/validation case leakage: {overlap[:10]}")

    model = build_an2023_adapted_unet(torch).to(device)
    _initialize_glorot(model, torch)
    initialization = "glorot_from_scratch"
    pretrained_sha256 = None
    if pretrained_model is not None:
        checkpoint = _torch_load(torch, pretrained_model, device="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
        initialization = "external_supervised_pretraining"
        pretrained_sha256 = _sha256(pretrained_model)

    class_weights = _class_weights(train_cases, torch, device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = _grad_scaler(torch, enabled=use_amp)

    config = {
        "implementation": "an2023_paper_inspired_full_field_v1",
        "exact_paper_reproduction": False,
        "paper_training_code_available": False,
        "published_vs_released_output_discrepancy": (
            "paper describes two classes; released TorchScript has three logits"
        ),
        "adaptations": [
            "two-class output",
            "same-padding full native field instead of unpublished training crop implementation",
            "per-volume 0.5/99.5 percentile scaling to [-1,1]",
            "explicit validation-Dice checkpointing and early stopping",
            "gradient accumulation for P100 memory",
        ],
        "train_root": str(train_root.resolve()),
        "validation_root": str(validation_root.resolve()),
        "train_case_ids": [case.case_id for case in train_cases],
        "validation_case_ids": [case.case_id for case in validation_cases],
        "epochs": epochs,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        "learning_rate": learning_rate,
        "gradient_accumulation": gradient_accumulation,
        "physical_batch_size": 1,
        "effective_batch_size": gradient_accumulation,
        "noise_std_normalized_units": noise_std,
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "metrics_threshold": metrics_threshold,
        "loss": "inverse_global_class_frequency_weighted_cross_entropy",
        "optimizer": "Adam",
        "learning_rate_schedule": "constant",
        "seed": seed,
        "device": str(device),
        "amp": use_amp,
        "initialization": initialization,
        "pretrained_model": str(pretrained_model.resolve()) if pretrained_model else None,
        "pretrained_sha256": pretrained_sha256,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "class_weights": [float(value) for value in class_weights.detach().cpu().tolist()],
    }
    config_hash = _config_hash(config)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "run_config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if _config_hash(existing_config) != config_hash:
            raise RuntimeError("Existing output has a different training configuration")
    else:
        if any(output_root.iterdir()):
            raise FileExistsError(f"Unverified non-empty output directory: {output_root}")
        _write_json(config_path, config)

    start_epoch = 0
    best_dice = -math.inf
    best_epoch = 0
    no_improvement = 0
    last_checkpoint = output_root / "last_checkpoint.pth"
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"No resumable checkpoint: {last_checkpoint}")
        resumed = _torch_load(torch, last_checkpoint, device=device)
        if resumed.get("config_hash") != config_hash:
            raise RuntimeError("Resume checkpoint configuration hash differs")
        model.load_state_dict(resumed["model_state_dict"])
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if use_amp and resumed.get("scaler_state_dict"):
            scaler.load_state_dict(resumed["scaler_state_dict"])
        start_epoch = int(resumed["epoch"])
        best_dice = float(resumed["best_dice"])
        best_epoch = int(resumed["best_epoch"])
        no_improvement = int(resumed["no_improvement"])
        _restore_rng_state(resumed, torch)
    elif last_checkpoint.exists():
        raise FileExistsError(
            f"A checkpoint already exists at {last_checkpoint}; pass --resume or use a new output"
        )

    best_checkpoint = output_root / "best_by_validation_dice.pth"
    _write_status(output_root, "running", completed_epoch=start_epoch)
    stopped_early = False
    try:
        for epoch_index in range(start_epoch, epochs):
            train_loss = _train_epoch(
                model=model,
                cases=train_cases,
                loss_function=loss_function,
                optimizer=optimizer,
                scaler=scaler,
                torch=torch,
                device=device,
                noise_std=noise_std,
                gradient_accumulation=gradient_accumulation,
                use_amp=use_amp,
                seed=seed + epoch_index,
            )
            validation = _evaluate(
                model=model,
                cases=validation_cases,
                loss_function=loss_function,
                torch=torch,
                device=device,
                threshold=metrics_threshold,
                use_amp=use_amp,
            )
            epoch = epoch_index + 1
            _append_metrics(
                output_root / "metrics_epoch.csv",
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **validation,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )
            improved = validation["validation_dice"] > best_dice + early_stop_min_delta
            if improved:
                best_dice = float(validation["validation_dice"])
                best_epoch = epoch
                no_improvement = 0
                _save_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    torch=torch,
                    epoch=epoch,
                    best_dice=best_dice,
                    best_epoch=best_epoch,
                    no_improvement=no_improvement,
                    config_hash=config_hash,
                )
            else:
                no_improvement += 1
            _save_checkpoint(
                last_checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                torch=torch,
                epoch=epoch,
                best_dice=best_dice,
                best_epoch=best_epoch,
                no_improvement=no_improvement,
                config_hash=config_hash,
            )
            _write_status(output_root, "running", completed_epoch=epoch)
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={validation['validation_loss']:.6f} "
                f"val_dice={validation['validation_dice']:.6f} "
                f"best={best_dice:.6f}@{best_epoch}"
            )
            if no_improvement >= early_stop_patience:
                stopped_early = True
                break
    except KeyboardInterrupt:
        _write_status(output_root, "interrupted", completed_epoch=max(start_epoch, 0))
        return 130

    if not best_checkpoint.is_file():
        raise RuntimeError("Training completed without a validation-Dice checkpoint")
    selected = _torch_load(torch, best_checkpoint, device=device)
    model.load_state_dict(selected["model_state_dict"])
    final_model = output_root / "an2023_adapted_unet.model"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "selected_epoch": best_epoch,
            "selected_validation_dice": best_dice,
        },
        final_model,
    )
    _write_json(
        output_root / "selected_checkpoint.json",
        {
            "selection_metric": "validation_dice",
            "selected_epoch": best_epoch,
            "selected_validation_dice": best_dice,
            "checkpoint": str(best_checkpoint),
            "final_model": str(final_model),
            "final_model_sha256": _sha256(final_model),
        },
    )
    export = _export_probabilities(
        model=model,
        cases=validation_cases,
        output_root=output_root / "prediction_exports" / "validation" / "final",
        torch=torch,
        device=device,
        use_amp=use_amp,
        threshold=metrics_threshold,
    )
    completed_epoch = max(
        int(row["epoch"])
        for row in _read_csv(output_root / "metrics_epoch.csv")
    )
    _write_status(
        output_root,
        "early_stopped" if stopped_early else "completed",
        completed_epoch=completed_epoch,
        reason=(
            f"no_validation_dice_improvement_for_{early_stop_patience}_epochs"
            if stopped_early
            else "maximum_epochs_reached"
        ),
    )
    _write_json(
        output_root / "final_metrics.json",
        {
            "best_validation_dice": best_dice,
            "best_epoch": best_epoch,
            "completed_epoch": completed_epoch,
            "n_validation_cases": export["n_cases"],
        },
    )
    return 0


def _load_cases(
    root: Path,
    *,
    lower_percentile: float,
    upper_percentile: float,
) -> list[VolumeCase]:
    cases = []
    seen = set()
    for scan_path in sorted(root.rglob("scan.nii.gz")):
        case_id = scan_path.parent.name
        label_path = scan_path.parent / "scan_lesionIAM.nii.gz"
        if not label_path.is_file():
            continue
        if case_id in seen:
            raise ValueError(f"Duplicate case_id={case_id!r} below {root}")
        seen.add(case_id)
        scan_image = nib.load(str(scan_path))
        label_image = nib.load(str(label_path))
        scan = np.asarray(scan_image.dataobj)
        label = np.asarray(label_image.dataobj)
        if scan.ndim != 4 or scan.shape[-1] != 1:
            raise ValueError(f"{case_id}: expected X,Y,Z,1 scan; found {scan.shape}")
        scan = scan[..., 0].astype(np.float32)
        if scan.shape != label.shape:
            raise ValueError(f"{case_id}: scan/label shape mismatch")
        if not np.allclose(scan_image.affine, label_image.affine, rtol=0, atol=1e-5):
            raise ValueError(f"{case_id}: scan/label affine mismatch")
        unique = np.unique(label)
        if not np.all(np.isin(unique, (0, 1))):
            raise ValueError(f"{case_id}: non-binary label {unique[:20]}")
        scan = _robust_scale(
            scan,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            case_id=case_id,
        )
        cases.append(
            VolumeCase(
                case_id=case_id,
                scan_path=scan_path,
                label_path=label_path,
                scan=scan,
                label=(label > 0.5).astype(np.uint8),
            )
        )
    if not cases:
        raise ValueError(f"No prepared cases found below {root}")
    return cases


def _robust_scale(
    scan: np.ndarray,
    *,
    lower_percentile: float,
    upper_percentile: float,
    case_id: str,
) -> np.ndarray:
    finite = scan[np.isfinite(scan)]
    if not finite.size:
        raise ValueError(f"{case_id}: scan contains no finite values")
    low, high = np.percentile(finite, (lower_percentile, upper_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"{case_id}: degenerate robust intensity range {low}, {high}")
    clipped = np.clip(scan, low, high)
    scaled = 2.0 * (clipped - low) / (high - low) - 1.0
    if not np.isfinite(scaled).all():
        raise ValueError(f"{case_id}: normalization created non-finite values")
    return scaled.astype(np.float32)


def _class_weights(cases: list[VolumeCase], torch: Any, device: Any) -> Any:
    positive = sum(int(case.label.sum()) for case in cases)
    total = sum(int(case.label.size) for case in cases)
    negative = total - positive
    if positive <= 0 or negative <= 0:
        raise ValueError("Training data must contain both lesion and background voxels")
    weights = np.asarray([total / (2 * negative), total / (2 * positive)], dtype=np.float32)
    return torch.as_tensor(weights, device=device)


def _train_epoch(
    *,
    model: Any,
    cases: list[VolumeCase],
    loss_function: Any,
    optimizer: Any,
    scaler: Any,
    torch: Any,
    device: Any,
    noise_std: float,
    gradient_accumulation: int,
    use_amp: bool,
    seed: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    indices = np.random.default_rng(seed).permutation(len(cases))
    losses = []
    for position, index in enumerate(indices, start=1):
        case = cases[int(index)]
        image = torch.from_numpy(case.scan)[None, None].to(device)
        target = torch.from_numpy(case.label.astype(np.int64))[None].to(device)
        if noise_std:
            image = image + torch.randn_like(image) * noise_std
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
            loss = loss_function(logits, target)
            scaled_loss = loss / gradient_accumulation
        scaler.scale(scaled_loss).backward()
        losses.append(float(loss.detach().cpu()))
        if position % gradient_accumulation == 0 or position == len(indices):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def _evaluate(
    *,
    model: Any,
    cases: list[VolumeCase],
    loss_function: Any,
    torch: Any,
    device: Any,
    threshold: float,
    use_amp: bool,
) -> dict[str, float | int]:
    model.eval()
    losses = []
    dice_values = []
    precision_values = []
    recall_values = []
    target_voxels = 0
    predicted_voxels = 0
    with torch.no_grad():
        for case in cases:
            image = torch.from_numpy(case.scan)[None, None].to(device)
            target = torch.from_numpy(case.label.astype(np.int64))[None].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image)
                loss = loss_function(logits, target)
            probability = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
            prediction = probability >= threshold
            metrics = _binary_metrics(prediction, case.label > 0)
            losses.append(float(loss.detach().cpu()))
            dice_values.append(metrics["dice"])
            if metrics["precision"] is not None:
                precision_values.append(metrics["precision"])
            if metrics["recall"] is not None:
                recall_values.append(metrics["recall"])
            target_voxels += int(case.label.sum())
            predicted_voxels += int(prediction.sum())
    return {
        "validation_loss": float(np.mean(losses)),
        "validation_dice": float(np.mean(dice_values)),
        "validation_precision": _mean_or_nan(precision_values),
        "validation_recall": _mean_or_nan(recall_values),
        "validation_target_voxels": target_voxels,
        "validation_predicted_voxels": predicted_voxels,
    }


def _export_probabilities(
    *,
    model: Any,
    cases: list[VolumeCase],
    output_root: Path,
    torch: Any,
    device: Any,
    use_amp: bool,
    threshold: float,
) -> dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    model.eval()
    records = []
    with torch.no_grad():
        for index, case in enumerate(cases):
            image_tensor = torch.from_numpy(case.scan)[None, None].to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(image_tensor)
            probability = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
            if probability.shape != case.label.shape:
                raise RuntimeError(f"Native output shape mismatch for {case.case_id}")
            reference = nib.load(str(case.label_path))
            case_root = output_root / f"case_{index:04d}"
            case_root.mkdir()
            probability_path = case_root / "lesion_probability.nii.gz"
            target_path = case_root / "target_mask.nii.gz"
            _save_nifti_like(probability_path, probability, reference, np.float32)
            _save_nifti_like(target_path, case.label, reference, np.uint8)
            records.append(
                {
                    "split": "validation",
                    "case_id": case.case_id,
                    "epoch": "final_best_validation_dice",
                    "threshold": threshold,
                    "lesion_probability": str(probability_path),
                    "target_mask": str(target_path),
                    "source_scan": str(case.scan_path),
                }
            )
    _write_csv(output_root / "prediction_export_manifest.csv", records)
    return {
        "n_cases": len(records),
        "manifest": str(output_root / "prediction_export_manifest.csv"),
    }


def _binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    denominator = 2 * tp + fp + fn
    return {
        "dice": 1.0 if denominator == 0 else 2 * tp / denominator,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
    }


def _pad_to_multiple_of_four(torch: Any, value: Any) -> tuple[Any, tuple[int, int]]:
    x, y = value.shape[2], value.shape[3]
    pad_x = (-x) % 4
    pad_y = (-y) % 4
    # torch pad is specified from the final dimension backwards: Z, Y, X.
    padded = torch.nn.functional.pad(value, (0, 0, 0, pad_y, 0, pad_x))
    return padded, (pad_x, pad_y)


def _remove_padding(value: Any, padding_values: tuple[int, int], shape: Any) -> Any:
    pad_x, pad_y = padding_values
    if pad_x:
        value = value[:, :, :-pad_x]
    if pad_y:
        value = value[:, :, :, :-pad_y]
    if tuple(value.shape[2:]) != tuple(shape):
        raise RuntimeError(f"Model output shape {value.shape[2:]} differs from input {shape}")
    return value


def _initialize_glorot(model: Any, torch: Any) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    torch: Any,
    epoch: int,
    best_dice: float,
    best_epoch: int,
    no_improvement: int,
    config_hash: str,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "no_improvement": no_improvement,
        "config_hash": config_hash,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_rng_state(checkpoint: dict[str, Any], torch: Any) -> None:
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint.get("torch_cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _grad_scaler(torch: Any, *, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _torch_load(torch: Any, path: Path, *, device: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _append_metrics(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_status(
    output_root: Path,
    status: str,
    *,
    completed_epoch: int,
    reason: str = "",
) -> None:
    _write_json(
        output_root / "run_status.json",
        {"status": status, "completed_epoch": completed_epoch, "reason": reason},
    )


def _save_nifti_like(
    path: Path,
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    dtype: Any,
) -> None:
    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(dtype)
    header.set_zooms(reference.header.get_zooms()[: data.ndim])
    nib.save(
        nib.Nifti1Image(data.astype(dtype, copy=False), reference.affine, header),
        str(path),
    )


def _validate_hyperparameters(**values: Any) -> None:
    if values["epochs"] < 1 or values["early_stop_patience"] < 1:
        raise ValueError("epochs and early-stop patience must be >= 1")
    if values["early_stop_min_delta"] < 0 or values["learning_rate"] <= 0:
        raise ValueError("Invalid early-stop delta or learning rate")
    if values["gradient_accumulation"] < 1 or values["noise_std"] < 0:
        raise ValueError("Invalid accumulation or noise standard deviation")
    if not 0 <= values["lower_percentile"] < values["upper_percentile"] <= 100:
        raise ValueError("Invalid normalization percentiles")
    if not 0 < values["metrics_threshold"] < 1:
        raise ValueError("metrics threshold must be between 0 and 1")


def _config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
