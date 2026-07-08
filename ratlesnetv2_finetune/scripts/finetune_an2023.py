"""Fine-tune the An et al. 2023 mouse stroke lesion TorchScript model.

The public repository at
https://github.com/scalableminds/stroke-lesion-segmentation contains an
inference-only notebook and ``lesion_model.pt``. This wrapper keeps our cloud
workflow close to the RatLesNetV2 one: it consumes the same prepared
``scan.nii.gz`` / ``scan_lesionIAM.nii.gz`` case folders, writes validation
metrics/checkpoints, and exports NIfTI predictions/overlays.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 import (
    EvaluationSummary,
    _aggregate_case_metrics,
    _append_case_metrics,
    _append_epoch_metrics,
    _append_loss,
    _append_lr_history,
    _current_lr,
    _format_epoch_metrics,
    _loss_to_float,
    _next_run_dir,
    _parse_export_epochs,
    _parse_export_splits,
    _safe_path_part,
    _save_nifti_like,
    _segmentation_metrics,
    _should_export_epoch,
    _write_final_metrics,
    _write_json,
    _write_metric_plots,
    _write_prediction_export_manifest,
    _write_run_status,
)

SCAN_FILENAMES = ("scan.nii.gz", "scan.nii")
LABEL_FILENAMES = ("scan_lesionIAM.nii.gz", "scan_lesionIAM.nii")


@dataclass(frozen=True)
class CasePaths:
    case_dir: Path
    scan: Path
    label: Path

    @property
    def case_id(self) -> str:
        return self.case_dir.name


@dataclass(frozen=True)
class CropPadPlan:
    source_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]
    source_slices: tuple[slice, slice, slice]
    target_slices: tuple[slice, slice, slice]


@dataclass
class LoadedCase:
    case: CasePaths
    scan: np.ndarray
    target: np.ndarray
    image: Any
    label: Any
    plan: CropPadPlan
    reference: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared train split root")
    parser.add_argument("--validation", default=None, help="Prepared validation split root")
    parser.add_argument("--test", default=None, help="Prepared held-out test split root")
    parser.add_argument("--output", required=True, help="Output folder for run checkpoints/logs")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to scalableminds/stroke-lesion-segmentation lesion_model.pt",
    )
    parser.add_argument(
        "--require-pretrained",
        action="store_true",
        help="Fail if --model-path is missing. Kept for parity with RatLesNetV2 commands.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU id; use -1 for CPU")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--metrics-threshold", type=float, default=0.8)
    parser.add_argument("--output-key", default="segment_type_task")
    parser.add_argument(
        "--output-index",
        type=int,
        default=0,
        help="Index inside the TorchScript output list for --output-key.",
    )
    parser.add_argument("--crop-x", type=int, default=152)
    parser.add_argument("--crop-y", type=int, default=196)
    parser.add_argument("--crop-z", type=int, default=30)
    parser.add_argument("--positive-class-weight", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "reduce-on-plateau"],
        default="none",
    )
    parser.add_argument(
        "--lr-scheduler-metric",
        choices=["validation_dice", "validation_loss"],
        default="validation_dice",
    )
    parser.add_argument("--lr-plateau-patience", type=int, default=3)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-min-delta", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--export-predictions",
        default="",
        help="Comma-separated splits for prediction exports: train,validation,test.",
    )
    parser.add_argument("--export-prediction-limit", type=int, default=8)
    parser.add_argument("--export-prediction-epochs", default="1,2,5,final")
    parser.add_argument("--max-train-cases", type=int, default=None)
    parser.add_argument("--max-validation-cases", type=int, default=None)
    parser.add_argument("--max-test-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    _set_seed(args.seed, torch=torch)
    device = _select_device(torch, args.gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_cases = _limit_cases(_find_cases(Path(args.input)), args.max_train_cases)
    _require_nonempty(train_cases, "train")
    val_cases = (
        _limit_cases(_find_cases(Path(args.validation)), args.max_validation_cases)
        if args.validation
        else []
    )
    test_cases = (
        _limit_cases(_find_cases(Path(args.test)), args.max_test_cases)
        if args.test
        else []
    )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _next_run_dir(output_root)
    _write_run_config(
        run_dir,
        args,
        device=device,
        train_cases=len(train_cases),
        val_cases=len(val_cases),
        test_cases=len(test_cases),
    )

    model_path = Path(args.model_path)
    if args.require_pretrained and not model_path.is_file():
        raise FileNotFoundError(f"An 2023 TorchScript model not found: {model_path}")
    model = _load_model(torch=torch, model_path=model_path, device=device)
    optimizer = torch.optim.Adam(_trainable_parameters(model), lr=args.lr)
    lr_scheduler = _build_lr_scheduler(torch=torch, optimizer=optimizer, args=args)

    target_shape = (args.crop_x, args.crop_y, args.crop_z)
    export_splits = _parse_export_splits(args.export_predictions)
    export_epochs = _parse_export_epochs(args.export_prediction_epochs)
    if export_splits and args.export_prediction_limit < 1:
        raise ValueError("--export-prediction-limit must be >= 1")

    if args.eval_only:
        summaries = _evaluate_requested_splits(
            model=model,
            torch=torch,
            cases_by_split={"validation": val_cases, "test": test_cases},
            target_shape=target_shape,
            args=args,
        )
        for summary in summaries:
            summary["lr"] = _current_lr(optimizer)
            _append_epoch_metrics(run_dir / "metrics_epoch.csv", summary)
            _append_case_metrics(run_dir / "metrics_cases.csv", summary)
        _export_requested_predictions(
            model=model,
            torch=torch,
            cases_by_split={"train": train_cases, "validation": val_cases, "test": test_cases},
            target_shape=target_shape,
            args=args,
            run_dir=run_dir,
            epoch=0,
            is_final=True,
            export_splits=export_splits,
            export_epochs=export_epochs,
        )
        if summaries:
            _write_final_metrics(run_dir / "final_metrics.json", summaries)
        if not args.no_plots:
            _write_metric_plots(run_dir)
        _write_run_status(run_dir / "run_status.json", status="evaluated", completed_epoch=0)
        return 0

    best_state: dict[str, dict[str, Any]] = {}
    no_dice_improvement = 0
    completed_epoch = 0
    stopped_early = False
    stop_reason = ""
    try:
        for epoch_num in range(1, args.epochs + 1):
            epoch_lr = _current_lr(optimizer)
            train_loss = _run_epoch(
                model=model,
                torch=torch,
                optimizer=optimizer,
                cases=train_cases,
                target_shape=target_shape,
                args=args,
                seed=args.seed + epoch_num,
            )
            _append_loss(run_dir / "training_loss", train_loss)

            summaries: list[EvaluationSummary] = []
            val_loss: float | None = None
            scheduler_metric_name = ""
            scheduler_metric_value: float | None = None
            should_eval = args.eval_every > 0 and epoch_num % args.eval_every == 0
            if should_eval and args.eval_train:
                summaries.append(
                    _run_evaluation(
                        split="train",
                        epoch=epoch_num,
                        model=model,
                        torch=torch,
                        cases=train_cases,
                        target_shape=target_shape,
                        args=args,
                    )
                )
            if should_eval and val_cases:
                val_eval = _run_evaluation(
                    split="validation",
                    epoch=epoch_num,
                    model=model,
                    torch=torch,
                    cases=val_cases,
                    target_shape=target_shape,
                    args=args,
                )
                val_loss = float(val_eval["loss"])
                _append_loss(run_dir / "validation_loss", val_loss)
                summaries.append(val_eval)
                if lr_scheduler is not None:
                    scheduler_metric_name, scheduler_metric_value = _scheduler_metric_value(
                        val_eval,
                        metric=args.lr_scheduler_metric,
                    )
                    lr_scheduler.step(scheduler_metric_value)
                improved = _update_best_checkpoints(
                    torch=torch,
                    model=model,
                    run_dir=run_dir,
                    summary=val_eval,
                    best_state=best_state,
                    min_dice_delta=args.early_stop_min_delta,
                )
                no_dice_improvement = 0 if improved else no_dice_improvement + 1

            next_lr = _current_lr(optimizer)
            for summary in summaries:
                summary["lr"] = epoch_lr
                _append_epoch_metrics(run_dir / "metrics_epoch.csv", summary)
                _append_case_metrics(run_dir / "metrics_cases.csv", summary)
            _append_lr_history(
                run_dir / "lr_history.csv",
                epoch=epoch_num,
                train_loss=train_loss,
                lr=epoch_lr,
                next_lr=next_lr,
                scheduler_metric=scheduler_metric_name,
                scheduler_value=scheduler_metric_value,
            )
            if summaries:
                _export_requested_predictions(
                    model=model,
                    torch=torch,
                    cases_by_split={
                        "train": train_cases,
                        "validation": val_cases,
                        "test": test_cases,
                    },
                    target_shape=target_shape,
                    args=args,
                    run_dir=run_dir,
                    epoch=epoch_num,
                    is_final=False,
                    export_splits=export_splits,
                    export_epochs=export_epochs,
                )
            if summaries and not args.no_plots:
                _write_metric_plots(run_dir)

            val_text = "" if val_loss is None else f" Val Loss: {val_loss:.8g}."
            lr_text = f" LR: {epoch_lr:.4g}."
            if next_lr != epoch_lr:
                lr_text += f" Next LR: {next_lr:.4g}."
            print(
                f"Epoch: {epoch_num}. Loss: {train_loss:.8g}.{val_text}{lr_text}"
                + _format_epoch_metrics(summaries)
            )

            if args.save_every > 0 and epoch_num % args.save_every == 0:
                _save_script_model(torch, model, run_dir / f"an2023_epoch{epoch_num:03d}.pt")
            _save_script_model(torch, model, run_dir / "last.pt")
            completed_epoch = epoch_num
            if (
                args.early_stop_patience is not None
                and val_cases
                and should_eval
                and no_dice_improvement >= args.early_stop_patience
            ):
                stopped_early = True
                stop_reason = (
                    f"Early stopping after {no_dice_improvement} validation evaluations "
                    "without Dice improvement."
                )
                print(stop_reason)
                break
    except KeyboardInterrupt:
        _save_script_model(torch, model, run_dir / "interrupted.pt")
        _save_script_model(torch, model, run_dir / "last.pt")
        _write_run_status(
            run_dir / "run_status.json",
            status="interrupted",
            completed_epoch=completed_epoch,
            interrupted_epoch=completed_epoch + 1,
            reason="KeyboardInterrupt",
        )
        if not args.no_plots:
            _write_metric_plots(run_dir)
        return 130

    _save_script_model(torch, model, run_dir / "an2023_finetuned.pt")
    final_summaries = _evaluate_requested_splits(
        model=model,
        torch=torch,
        cases_by_split={"validation_final": val_cases, "test": test_cases},
        target_shape=target_shape,
        args=args,
        epoch=completed_epoch,
    )
    for summary in final_summaries:
        summary["lr"] = _current_lr(optimizer)
        _append_epoch_metrics(run_dir / "metrics_epoch.csv", summary)
        _append_case_metrics(run_dir / "metrics_cases.csv", summary)
    _export_requested_predictions(
        model=model,
        torch=torch,
        cases_by_split={"train": train_cases, "validation": val_cases, "test": test_cases},
        target_shape=target_shape,
        args=args,
        run_dir=run_dir,
        epoch=completed_epoch,
        is_final=True,
        export_splits=export_splits,
        export_epochs=export_epochs,
    )
    if final_summaries:
        _write_final_metrics(run_dir / "final_metrics.json", final_summaries)
    if not args.no_plots:
        _write_metric_plots(run_dir)
    _write_run_status(
        run_dir / "run_status.json",
        status="early_stopped" if stopped_early else "completed",
        completed_epoch=completed_epoch,
        reason=stop_reason,
    )
    return 0


def _set_seed(seed: int, *, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(torch: Any, gpu: int) -> Any:
    if gpu < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        if gpu >= torch.cuda.device_count():
            raise ValueError(
                f"Requested CUDA GPU {gpu}, but only {torch.cuda.device_count()} are visible"
            )
        return torch.device(f"cuda:{gpu}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No CUDA/MPS GPU available. Pass --gpu -1 to force CPU.")


def _find_cases(root: Path) -> list[CasePaths]:
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared split root not found: {root}")
    cases = []
    for case_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        scan = _first_existing(case_dir, SCAN_FILENAMES)
        if scan is None:
            continue
        label = _first_existing(case_dir, LABEL_FILENAMES)
        if label is None:
            raise FileNotFoundError(f"Missing scan_lesionIAM label next to {scan}")
        cases.append(CasePaths(case_dir=case_dir, scan=scan, label=label))
    return cases


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return None


def _limit_cases(cases: list[CasePaths], limit: int | None) -> list[CasePaths]:
    if limit is None:
        return cases
    if limit < 1:
        raise ValueError("Case limit must be >= 1")
    return cases[:limit]


def _require_nonempty(cases: list[CasePaths], split: str) -> None:
    if not cases:
        raise ValueError(f"{split} split contains no cases")


def _load_model(*, torch: Any, model_path: Path, device: Any) -> Any:
    if not model_path.is_file():
        raise FileNotFoundError(f"An 2023 TorchScript model not found: {model_path}")
    model = torch.jit.load(str(model_path), map_location=device)
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def _trainable_parameters(model: Any) -> list[Any]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError(
            "Loaded TorchScript model has no trainable parameters. "
            "This release may be inference-only in a way that cannot be fine-tuned."
        )
    return params


def _build_lr_scheduler(*, torch: Any, optimizer: Any, args: argparse.Namespace) -> Any | None:
    if args.lr_scheduler == "none":
        return None
    mode = "max" if args.lr_scheduler_metric == "validation_dice" else "min"
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        threshold=args.lr_plateau_min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )


def _load_case(case: CasePaths, *, target_shape: tuple[int, int, int], torch: Any, device: Any):
    import nibabel as nib

    scan_img = nib.load(str(case.scan))
    label_img = nib.load(str(case.label))
    scan = _scan_array(scan_img)
    target = _label_array(label_img)
    if scan.shape != target.shape:
        raise ValueError(
            f"Scan/label shape mismatch for {case.case_id}: {scan.shape} vs {target.shape}"
        )
    scan_model, plan = _center_crop_or_pad(scan, target_shape, fill=0.0)
    target_model, _ = _center_crop_or_pad(target, target_shape, fill=0)
    normalized = _an2023_normalize(scan_model)
    image = (
        torch.as_tensor(normalized, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    label = torch.as_tensor(target_model.astype(np.int64), dtype=torch.long, device=device)
    label = label.unsqueeze(0)
    return LoadedCase(
        case=case,
        scan=scan,
        target=target,
        image=image,
        label=label,
        plan=plan,
        reference=scan_img,
    )


def _scan_array(img: Any) -> np.ndarray:
    arr = np.asanyarray(img.dataobj)
    arr = np.squeeze(arr)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D scan after squeezing, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _label_array(img: Any) -> np.ndarray:
    arr = np.squeeze(np.asanyarray(img.dataobj))
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D label after squeezing, got shape {arr.shape}")
    return (arr > 0.5).astype(np.uint8)


def _an2023_normalize(scan: np.ndarray) -> np.ndarray:
    finite = np.asarray(scan, dtype=np.float32)
    max_value = float(np.nanmax(finite))
    if not np.isfinite(max_value) or max_value <= 0.0:
        raise ValueError("Cannot apply An 2023 normalization: scan max must be finite and > 0")
    uint8_like = np.clip(finite / max_value * 255.0, 0.0, 255.0)
    return (uint8_like - 127.5) / 127.5


def _center_crop_or_pad(
    value: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    fill: float | int,
) -> tuple[np.ndarray, CropPadPlan]:
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    source_shape = tuple(int(v) for v in arr.shape)
    target = tuple(int(v) for v in target_shape)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(source_shape, target, strict=True):
        if source_size >= target_size:
            source_start = (source_size - target_size) // 2
            source_stop = source_start + target_size
            target_start = 0
            target_stop = target_size
        else:
            source_start = 0
            source_stop = source_size
            target_start = (target_size - source_size) // 2
            target_stop = target_start + source_size
        source_slices.append(slice(source_start, source_stop))
        target_slices.append(slice(target_start, target_stop))
    out = np.full(target, fill, dtype=arr.dtype)
    source_tuple = tuple(source_slices)
    target_tuple = tuple(target_slices)
    out[target_tuple] = arr[source_tuple]
    return out, CropPadPlan(source_shape, target, source_tuple, target_tuple)


def _restore_crop_or_pad(value: np.ndarray, plan: CropPadPlan, *, fill: float | int = 0):
    arr = np.asarray(value)
    if tuple(arr.shape) != plan.target_shape:
        raise ValueError(f"Expected transformed shape {plan.target_shape}, got {arr.shape}")
    out = np.full(plan.source_shape, fill, dtype=arr.dtype)
    out[plan.source_slices] = arr[plan.target_slices]
    return out


def _run_epoch(
    *,
    model: Any,
    torch: Any,
    optimizer: Any,
    cases: list[CasePaths],
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
    seed: int,
) -> float:
    model.train()
    order = list(range(len(cases)))
    random.Random(seed).shuffle(order)
    total = 0.0
    for index in order:
        loaded = _load_case(
            cases[index],
            target_shape=target_shape,
            torch=torch,
            device=_device(model),
        )
        logits = _extract_logits(model(loaded.image), key=args.output_key, index=args.output_index)
        target, _output_plan = _align_target_to_logits(
            torch=torch,
            target=loaded.label,
            logits=logits,
        )
        loss = _ce_dice_loss(
            torch=torch,
            logits=logits,
            target=target,
            positive_class_weight=args.positive_class_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += _loss_to_float(loss)
    return total / len(cases)


def _run_evaluation(
    *,
    split: str,
    epoch: int,
    model: Any,
    torch: Any,
    cases: list[CasePaths],
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> EvaluationSummary:
    model.eval()
    total = 0.0
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for case in cases:
            loaded = _load_case(
                case,
                target_shape=target_shape,
                torch=torch,
                device=_device(model),
            )
            logits = _extract_logits(
                model(loaded.image),
                key=args.output_key,
                index=args.output_index,
            )
            target, output_plan = _align_target_to_logits(
                torch=torch,
                target=loaded.label,
                logits=logits,
            )
            loss = _ce_dice_loss(
                torch=torch,
                logits=logits,
                target=target,
                positive_class_weight=args.positive_class_weight,
            )
            probabilities = torch.softmax(logits, dim=1)
            pred_mask = _full_volume_prediction_mask(
                probabilities=probabilities,
                output_plan=output_plan,
                input_plan=loaded.plan,
                threshold=args.metrics_threshold,
            )
            metrics = _segmentation_metrics(
                pred_mask.astype(bool),
                loaded.target.astype(bool),
                threshold=args.metrics_threshold,
            )
            metrics["case_id"] = case.case_id
            metrics["loss"] = _loss_to_float(loss)
            rows.append(metrics)
            total += metrics["loss"]
    summary = EvaluationSummary(
        split=split,
        epoch=epoch,
        loss=total / len(cases),
        n_cases=len(cases),
        cases=rows,
    )
    summary.update(_aggregate_case_metrics(rows))
    return summary


def _evaluate_requested_splits(
    *,
    model: Any,
    torch: Any,
    cases_by_split: dict[str, list[CasePaths]],
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
    epoch: int = 0,
) -> list[EvaluationSummary]:
    summaries = []
    for split, cases in cases_by_split.items():
        if cases:
            summaries.append(
                _run_evaluation(
                    split=split,
                    epoch=epoch,
                    model=model,
                    torch=torch,
                    cases=cases,
                    target_shape=target_shape,
                    args=args,
                )
            )
    return summaries


def _extract_logits(output: Any, *, key: str, index: int) -> Any:
    if isinstance(output, dict):
        if key not in output:
            raise KeyError(f"Model output does not contain key {key!r}; keys={sorted(output)}")
        logits = output[key]
    else:
        logits = output
    if isinstance(logits, (list, tuple)):
        if not 0 <= index < len(logits):
            raise IndexError(
                f"Model output {key!r} has {len(logits)} tensors; index {index} is invalid"
            )
        logits = logits[index]
    if logits.ndim == 4 and logits.shape[0] >= 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 5 or logits.shape[1] < 2:
        raise ValueError(
            f"Expected logits with shape BxCxXxYxZ and C>=2, got {tuple(logits.shape)}"
        )
    return logits


def _align_target_to_logits(*, torch: Any, target: Any, logits: Any) -> tuple[Any, CropPadPlan]:
    target_np = target.squeeze(0).detach().cpu().numpy()
    aligned, plan = _center_crop_or_pad(target_np, tuple(int(v) for v in logits.shape[2:]), fill=0)
    aligned_tensor = torch.as_tensor(
        aligned.astype(np.int64),
        dtype=torch.long,
        device=logits.device,
    )
    return aligned_tensor.unsqueeze(0), plan


def _ce_dice_loss(
    *,
    torch: Any,
    logits: Any,
    target: Any,
    positive_class_weight: float,
    eps: float = 1e-6,
) -> Any:
    import torch.nn.functional as F

    weight = None
    if positive_class_weight != 1.0:
        weights = [1.0] * int(logits.shape[1])
        weights[1] = positive_class_weight
        weight = torch.tensor(weights, dtype=logits.dtype, device=logits.device)
    ce = F.cross_entropy(logits, target, weight=weight)
    probs = torch.softmax(logits, dim=1)[:, 1, ...]
    target_lesion = (target == 1).to(dtype=probs.dtype)
    num = 2.0 * torch.sum(probs * target_lesion)
    denom = torch.sum(probs * probs + target_lesion * target_lesion)
    dice = 1.0 - (num + eps) / (denom + eps)
    return ce + dice


def _full_volume_prediction_mask(
    *,
    probabilities: Any,
    output_plan: CropPadPlan,
    input_plan: CropPadPlan,
    threshold: float,
) -> np.ndarray:
    lesion_probability = probabilities[0, 1].detach().cpu().numpy().astype(np.float32)
    output_mask = (lesion_probability >= threshold).astype(np.uint8)
    input_mask = _restore_crop_or_pad(output_mask, output_plan, fill=0)
    return _restore_crop_or_pad(input_mask, input_plan, fill=0)


def _device(model: Any) -> Any:
    return next(model.parameters()).device


def _scheduler_metric_value(summary: EvaluationSummary, *, metric: str) -> tuple[str, float]:
    if metric == "validation_dice":
        return "validation_dice", float(summary["dice_mean"])
    if metric == "validation_loss":
        return "validation_loss", float(summary["loss"])
    raise ValueError(f"Unknown LR scheduler metric: {metric!r}")


def _update_best_checkpoints(
    *,
    torch: Any,
    model: Any,
    run_dir: Path,
    summary: EvaluationSummary,
    best_state: dict[str, dict[str, Any]],
    min_dice_delta: float,
) -> bool:
    dice = summary.get("dice_mean")
    loss = float(summary["loss"])
    improved_dice = False
    if dice is not None:
        previous = best_state.get("validation_dice", {}).get("value")
        if previous is None or float(dice) > float(previous) + min_dice_delta:
            _save_script_model(torch, model, run_dir / "best_by_validation_dice.pt")
            best_state["validation_dice"] = _best_checkpoint_record(
                summary=summary,
                metric="dice_mean",
                value=float(dice),
                filename="best_by_validation_dice.pt",
            )
            improved_dice = True

    previous_loss = best_state.get("validation_loss", {}).get("value")
    if previous_loss is None or loss < float(previous_loss):
        _save_script_model(torch, model, run_dir / "best_by_validation_loss.pt")
        best_state["validation_loss"] = _best_checkpoint_record(
            summary=summary,
            metric="loss",
            value=loss,
            filename="best_by_validation_loss.pt",
        )
    _write_json(run_dir / "best_checkpoints.json", best_state)
    return improved_dice


def _best_checkpoint_record(
    *,
    summary: EvaluationSummary,
    metric: str,
    value: float,
    filename: str,
) -> dict[str, Any]:
    return {
        "epoch": int(summary["epoch"]),
        "split": summary["split"],
        "metric": metric,
        "value": value,
        "filename": filename,
        "loss": float(summary["loss"]),
        "dice_mean": _json_number(summary.get("dice_mean")),
        "precision_mean": _json_number(summary.get("precision_mean")),
        "recall_mean": _json_number(summary.get("recall_mean")),
        "target_voxels": int(summary.get("target_voxels", 0)),
        "pred_voxels": int(summary.get("pred_voxels", 0)),
    }


def _json_number(value: Any) -> float | None:
    return None if value is None else float(value)


def _save_script_model(torch: Any, model: Any, path: Path) -> None:
    torch.jit.save(model, str(path))


def _export_requested_predictions(
    *,
    model: Any,
    torch: Any,
    cases_by_split: dict[str, list[CasePaths]],
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
    run_dir: Path,
    epoch: int,
    is_final: bool,
    export_splits: set[str],
    export_epochs: set[int | str],
) -> None:
    if not export_splits or not _should_export_epoch(
        epoch,
        is_final=is_final,
        export_epochs=export_epochs,
    ):
        return
    label = "final" if is_final else f"epoch_{epoch:03d}"
    for split in sorted(export_splits):
        cases = cases_by_split.get(split, [])
        if not cases:
            continue
        _export_predictions_for_split(
            model=model,
            torch=torch,
            cases=cases,
            target_shape=target_shape,
            args=args,
            run_dir=run_dir,
            out_dir=run_dir / "prediction_exports" / split / label,
            split=split,
            epoch=epoch,
            label=label,
        )


def _export_predictions_for_split(
    *,
    model: Any,
    torch: Any,
    cases: list[CasePaths],
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
    run_dir: Path,
    out_dir: Path,
    split: str,
    epoch: int,
    label: str,
) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for case in cases[: args.export_prediction_limit]:
            loaded = _load_case(
                case,
                target_shape=target_shape,
                torch=torch,
                device=_device(model),
            )
            logits = _extract_logits(
                model(loaded.image),
                key=args.output_key,
                index=args.output_index,
            )
            probabilities = torch.softmax(logits, dim=1)
            lesion_probability = probabilities[0, 1].detach().cpu().numpy().astype(np.float32)
            _target, output_plan = _align_target_to_logits(
                torch=torch,
                target=loaded.label,
                logits=logits,
            )
            probability_crop = _restore_crop_or_pad(lesion_probability, output_plan, fill=0.0)
            pred_crop = (probability_crop >= args.metrics_threshold).astype(np.uint8)
            probability = _restore_crop_or_pad(probability_crop, loaded.plan, fill=0.0)
            pred_mask = _restore_crop_or_pad(pred_crop, loaded.plan, fill=0)
            export_dir = out_dir / _safe_path_part(case.case_id)
            export_dir.mkdir(parents=True, exist_ok=True)
            _save_nifti_like(
                export_dir / "scan.nii.gz",
                loaded.scan.astype(np.float32),
                loaded.reference,
            )
            _save_nifti_like(export_dir / "target_mask.nii.gz", loaded.target, loaded.reference)
            _save_nifti_like(export_dir / "pred_mask.nii.gz", pred_mask, loaded.reference)
            _save_nifti_like(
                export_dir / "lesion_probability.nii.gz",
                probability.astype(np.float32),
                loaded.reference,
            )
            _write_overlay_png(export_dir / "overlay.png", loaded.scan, loaded.target, pred_mask)
            records.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "case_id": case.case_id,
                    "case_dir": str(case.case_dir),
                    "export_dir": str(export_dir),
                    "scan": str(export_dir / "scan.nii.gz"),
                    "target_mask": str(export_dir / "target_mask.nii.gz"),
                    "pred_mask": str(export_dir / "pred_mask.nii.gz"),
                    "lesion_probability": str(export_dir / "lesion_probability.nii.gz"),
                    "overlay": str(export_dir / "overlay.png"),
                }
            )
    if records:
        _write_prediction_export_manifest(out_dir / "prediction_export_manifest.csv", records)
        _publish_latest_qc_overlay(run_dir, records[0], split=split, epoch=epoch, label=label)


def _write_overlay_png(path: Path, scan: np.ndarray, target: np.ndarray, pred: np.ndarray) -> None:
    try:
        import os
        import tempfile

        mpl_config = Path(tempfile.gettempdir()) / "matplotlib"
        mpl_config.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        xdg_cache = Path(tempfile.gettempdir()) / "xdg-cache"
        (xdg_cache / "fontconfig").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    axis = int(np.argmin(scan.shape))
    axes = tuple(i for i in range(3) if i != axis)
    counts = target.astype(bool).sum(axis=axes) + pred.astype(bool).sum(axis=axes)
    index = int(np.argmax(counts)) if counts.size else scan.shape[axis] // 2
    image = _normalize_for_overlay(np.take(scan, index, axis=axis))
    target_slice = np.take(target, index, axis=axis).astype(bool)
    pred_slice = np.take(pred, index, axis=axis).astype(bool)
    rgb = np.stack([image, image, image], axis=-1)
    rgb[target_slice, 1] = 1.0
    rgb[target_slice, 0] *= 0.35
    rgb[target_slice, 2] *= 0.35
    rgb[pred_slice, 0] = 1.0
    rgb[pred_slice, 1] *= 0.35
    rgb[pred_slice, 2] *= 0.35
    rgb[np.logical_and(target_slice, pred_slice)] = np.array([1.0, 1.0, 0.0])
    plt.imsave(path, np.rot90(rgb), format="png")


def _normalize_for_overlay(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _publish_latest_qc_overlay(
    run_dir: Path,
    record: dict[str, Any],
    *,
    split: str,
    epoch: int,
    label: str,
) -> None:
    overlay = Path(str(record.get("overlay", "")))
    if not overlay.exists():
        return
    split_name = _safe_path_part(split)
    split_dest = run_dir / f"latest_{split_name}_qc_overlay.png"
    shutil.copyfile(overlay, split_dest)
    metadata = {
        "split": split,
        "epoch": int(epoch),
        "label": label,
        "case_id": record.get("case_id", ""),
        "source_overlay": str(overlay),
        "latest_overlay": str(split_dest),
    }
    _write_json(run_dir / f"latest_{split_name}_qc_overlay.json", metadata)
    default_dest = run_dir / "latest_qc_overlay.png"
    if split == "validation" or not default_dest.exists():
        shutil.copyfile(overlay, default_dest)
        default_metadata = dict(metadata)
        default_metadata["latest_overlay"] = str(default_dest)
        _write_json(run_dir / "latest_qc_overlay.json", default_metadata)


def _write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    device: Any,
    train_cases: int,
    val_cases: int,
    test_cases: int,
) -> None:
    payload = vars(args).copy()
    payload["device"] = str(device)
    payload["train_cases"] = train_cases
    payload["validation_cases"] = val_cases
    payload["test_cases"] = test_cases
    with (run_dir / "run_config.json").open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
