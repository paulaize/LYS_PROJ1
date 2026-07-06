"""Cloud-oriented RatLesNetV2 finetuning entry point.

This script intentionally depends on an external checkout of
https://github.com/jmlipman/RatLesNetv2. It mirrors the upstream training loop
but adds the pieces needed for transfer learning: explicit epoch/lr controls,
optional pretrained weights, validation loss output, and checkpointing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratlesnet-repo", required=True, help="Path to upstream RatLesNetV2")
    parser.add_argument("--input", required=True, help="Prepared train split root")
    parser.add_argument("--validation", default=None, help="Prepared validation split root")
    parser.add_argument("--test", default=None, help="Prepared held-out test split root")
    parser.add_argument("--output", required=True, help="Output folder for run checkpoints/logs")
    parser.add_argument(
        "--pretrained-model",
        default=None,
        help="Optional RatLesNetv2.model state dict",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--modalities", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU id; use -1 for CPU")
    parser.add_argument(
        "--loadMemory",
        type=int,
        default=0,
        help="1 loads train/validation into RAM",
    )
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate validation metrics every N epochs; 0 disables epoch evaluation",
    )
    parser.add_argument(
        "--eval-train",
        action="store_true",
        help="Also evaluate segmentation metrics on the training split each eval epoch",
    )
    parser.add_argument(
        "--metrics-threshold",
        type=float,
        default=0.5,
        help="Threshold for one-channel binary model outputs. Two-channel outputs use argmax.",
    )
    parser.add_argument("--max-train-cases", type=int, default=None, help="Small cloud smoke test")
    parser.add_argument(
        "--max-validation-cases",
        type=int,
        default=None,
        help="Small cloud smoke test",
    )
    parser.add_argument(
        "--max-test-cases",
        type=int,
        default=None,
        help="Small cloud smoke test",
    )
    parser.add_argument(
        "--allow-partial-state-dict",
        action="store_true",
        help="Load matching pretrained keys only. Use only when architecture details differ.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ratlesnet_repo = Path(args.ratlesnet_repo).resolve()
    if not ratlesnet_repo.exists():
        raise FileNotFoundError(f"RatLesNetV2 checkout not found: {ratlesnet_repo}")
    sys.path.insert(0, str(ratlesnet_repo))

    _patch_nibabel_get_data_compat()

    import torch
    from lib.DataWrapper import DataWrapper
    from lib.losses import CrossEntropyDiceLoss
    from lib.RatLesNetv2 import RatLesNetv2
    from lib.utils import he_normal, now

    _set_seed(args.seed, np=np, torch=torch)
    device = _select_device(torch, args.gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_input = _existing_dir(args.input, "--input")
    validation_input = _existing_dir(args.validation, "--validation") if args.validation else None
    test_input = _existing_dir(args.test, "--test") if args.test else None
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _next_run_dir(output_root)

    print(now() + f"Using device: {device}")
    print(now() + f"Loading train data from {train_input}")
    load_memory = args.loadMemory == 1
    train_data = DataWrapper(str(train_input), "train", device, loadMemory=load_memory)
    _limit_wrapper(train_data, args.max_train_cases)
    _require_nonempty(train_data, "train")
    if validation_input is not None:
        print(now() + f"Loading validation data from {validation_input}")
        val_data = DataWrapper(str(validation_input), "validation", device, loadMemory=load_memory)
        _limit_wrapper(val_data, args.max_validation_cases)
        _require_nonempty(val_data, "validation")
    else:
        val_data = None
    if test_input is not None:
        print(now() + f"Loading held-out test data from {test_input}")
        test_data = DataWrapper(str(test_input), "test", device, loadMemory=load_memory)
        _limit_wrapper(test_data, args.max_test_cases)
        _require_nonempty(test_data, "test")
    else:
        test_data = None

    model = RatLesNetv2(modalities=args.modalities, filters=args.filters)
    model.to(device)
    if args.pretrained_model:
        _load_pretrained(
            torch,
            model,
            Path(args.pretrained_model),
            device=device,
            strict=not args.allow_partial_state_dict,
        )
    else:
        model.apply(_weight_init(torch, he_normal))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _write_run_config(
        run_dir,
        args,
        device=device,
        train_cases=len(train_data),
        val_cases=len(val_data or []),
        test_cases=len(test_data or []),
    )

    print(now() + f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        epoch_num = epoch + 1
        train_loss = _run_epoch(
            model=model,
            data=train_data,
            loss_fn=CrossEntropyDiceLoss,
            optimizer=optimizer,
        )
        _append_loss(run_dir / "training_loss", train_loss)

        val_loss: float | None = None
        epoch_summaries: list[EvaluationSummary] = []
        should_eval = args.eval_every > 0 and epoch_num % args.eval_every == 0
        if should_eval and args.eval_train:
            train_eval = _run_evaluation(
                split="train",
                epoch=epoch_num,
                model=model,
                data=train_data,
                loss_fn=CrossEntropyDiceLoss,
                torch=torch,
                threshold=args.metrics_threshold,
            )
            epoch_summaries.append(train_eval)
        if should_eval and val_data is not None:
            val_eval = _run_evaluation(
                split="validation",
                epoch=epoch_num,
                model=model,
                data=val_data,
                loss_fn=CrossEntropyDiceLoss,
                torch=torch,
                threshold=args.metrics_threshold,
            )
            val_loss = val_eval["loss"]
            _append_loss(run_dir / "validation_loss", val_loss)
            epoch_summaries.append(val_eval)
        for summary in epoch_summaries:
            _append_epoch_metrics(run_dir / "metrics_epoch.csv", summary)
            _append_case_metrics(run_dir / "metrics_cases.csv", summary)

        val_text = "" if val_loss is None else f" Val Loss: {val_loss:.8g}."
        metrics_text = _format_epoch_metrics(epoch_summaries)
        print(now() + f"Epoch: {epoch_num}. Loss: {train_loss:.8g}.{val_text}{metrics_text}")

        if args.save_every > 0 and epoch_num % args.save_every == 0:
            torch.save(model.state_dict(), run_dir / f"RatLesNetv2_epoch{epoch_num:03d}.model")

    torch.save(model.state_dict(), run_dir / "RatLesNetv2.model")
    print(now() + f"Saved final model: {run_dir / 'RatLesNetv2.model'}")

    final_summaries: list[EvaluationSummary] = []
    if val_data is not None:
        final_summaries.append(
            _run_evaluation(
                split="validation_final",
                epoch=args.epochs,
                model=model,
                data=val_data,
                loss_fn=CrossEntropyDiceLoss,
                torch=torch,
                threshold=args.metrics_threshold,
            )
        )
    if test_data is not None:
        final_summaries.append(
            _run_evaluation(
                split="test",
                epoch=args.epochs,
                model=model,
                data=test_data,
                loss_fn=CrossEntropyDiceLoss,
                torch=torch,
                threshold=args.metrics_threshold,
            )
        )
    for summary in final_summaries:
        _append_epoch_metrics(run_dir / "metrics_epoch.csv", summary)
        _append_case_metrics(run_dir / "metrics_cases.csv", summary)
    if final_summaries:
        _write_final_metrics(run_dir / "final_metrics.json", final_summaries)
        print(now() + f"Wrote metrics: {run_dir / 'metrics_epoch.csv'}")
    else:
        print(
            now()
            + "No validation/test split was provided; wrote training loss only. "
            "Do not treat this run as a performance estimate."
        )
    return 0


def _set_seed(seed: int, *, np: Any, torch: Any) -> None:
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


def _patch_nibabel_get_data_compat() -> None:
    """Keep upstream RatLesNetV2 working with nibabel >= 5.

    RatLesNetV2's DataWrapper still calls ``img.get_data()``, which nibabel
    removed as an active API in version 5. Patching the method here avoids
    downgrading Colab's scientific Python stack.
    """
    import nibabel as nib
    import numpy as np

    def get_data(self: Any, caching: str = "fill") -> Any:  # noqa: ARG001
        return np.asanyarray(self.dataobj)

    nib.dataobj_images.DataobjImage.get_data = get_data


def _existing_dir(path: str | None, arg_name: str) -> Path:
    if path is None:
        raise ValueError(f"{arg_name} is required")
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{arg_name} directory not found: {p}")
    return p


def _next_run_dir(output_root: Path) -> Path:
    existing = [int(p.name) for p in output_root.iterdir() if p.is_dir() and p.name.isdigit()]
    run_dir = output_root / str(max([0] + existing) + 1)
    run_dir.mkdir()
    return run_dir


def _limit_wrapper(wrapper: Any, limit: int | None) -> None:
    if limit is None:
        return
    if limit < 1:
        raise ValueError("Case limit must be >= 1")
    wrapper.list = wrapper.list[:limit]
    if hasattr(wrapper, "dataX"):
        wrapper.dataX = wrapper.dataX[:limit]
        wrapper.dataY = wrapper.dataY[:limit]
        wrapper.dataId = wrapper.dataId[:limit]


def _require_nonempty(wrapper: Any, split: str) -> None:
    if len(wrapper) < 1:
        raise ValueError(f"{split} split contains no cases")


def _weight_init(torch: Any, he_normal: Any) -> Any:
    def apply(module: Any) -> None:
        if isinstance(module, torch.nn.Conv3d):
            he_normal(module.weight)
            torch.nn.init.zeros_(module.bias)

    return apply


def _load_pretrained(
    torch: Any,
    model: Any,
    model_path: Path,
    *,
    device: Any,
    strict: bool,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if strict:
        model.load_state_dict(state)
        return

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    current.update(compatible)
    model.load_state_dict(current)
    skipped = sorted(set(state) - set(compatible))
    print(
        f"Loaded {len(compatible)} pretrained tensors; "
        f"skipped {len(skipped)} incompatible tensors"
    )


def _run_epoch(*, model: Any, data: Any, loss_fn: Any, optimizer: Any) -> float:
    model.train()
    total = 0.0
    for i in range(len(data)):
        x, y, _id = data[i]
        pred = model(x)[0]
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += _loss_to_float(loss)
    return total / len(data)


class EvaluationSummary(dict):
    """Small typed marker for metric summary dictionaries."""

    split: str
    epoch: int
    loss: float
    n_cases: int
    cases: list[dict[str, Any]]


def _run_evaluation(
    *,
    split: str,
    epoch: int,
    model: Any,
    data: Any,
    loss_fn: Any,
    torch: Any,
    threshold: float,
) -> EvaluationSummary:
    model.eval()
    total = 0.0
    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for i in range(len(data)):
            x, y, case_id = data[i]
            pred = model(x)[0]
            loss = loss_fn(pred, y)
            case_metrics = _segmentation_metrics(pred, y, threshold=threshold)
            case_metrics["case_id"] = _case_id_to_str(case_id)
            case_metrics["loss"] = _loss_to_float(loss)
            cases.append(case_metrics)
            total += case_metrics["loss"]
    summary = EvaluationSummary(
        split=split,
        epoch=epoch,
        loss=total / len(data),
        n_cases=len(data),
        cases=cases,
    )
    summary.update(_aggregate_case_metrics(cases))
    return summary


def _loss_to_float(loss: Any) -> float:
    return float(loss.detach().cpu().numpy())


def _case_id_to_str(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _case_id_to_str(value[0])
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return "_".join(str(v) for v in value.tolist())
    return str(value)


def _segmentation_metrics(pred: Any, target: Any, *, threshold: float = 0.5) -> dict[str, Any]:
    pred_mask = np.squeeze(_prediction_to_binary(pred, threshold=threshold)).astype(bool)
    target_mask = np.squeeze(_target_to_binary(target)).astype(bool)
    if pred_mask.shape != target_mask.shape:
        raise ValueError(
            "Prediction/target shape mismatch while computing metrics: "
            f"{pred_mask.shape} vs {target_mask.shape}"
        )

    tp = int(np.logical_and(pred_mask, target_mask).sum())
    fp = int(np.logical_and(pred_mask, ~target_mask).sum())
    tn = int(np.logical_and(~pred_mask, ~target_mask).sum())
    fn = int(np.logical_and(~pred_mask, target_mask).sum())
    total = tp + fp + tn + fn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "dice": _dice_from_counts(tp, fp, fn),
        "iou": _empty_agreement_score(tp, fp, fn),
        "accuracy": _safe_div(tp + tn, total),
        "balanced_accuracy": _mean_optional([recall, specificity]),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "target_voxels": int(target_mask.sum()),
        "pred_voxels": int(pred_mask.sum()),
    }


def _prediction_to_binary(pred: Any, *, threshold: float) -> np.ndarray:
    arr = _to_numpy(pred)
    if arr.dtype == bool:
        return arr
    if arr.ndim >= 5 and arr.shape[1] > 1:
        return np.argmax(arr, axis=1) == 1
    if arr.ndim >= 5 and arr.shape[1] == 1:
        return _score_to_binary(arr[:, 0, ...], threshold=threshold)
    if arr.ndim == 4 and 1 < arr.shape[0] <= 4:
        return np.argmax(arr, axis=0) == 1
    if arr.ndim == 4 and arr.shape[0] == 1:
        return _score_to_binary(arr[0, ...], threshold=threshold)
    return _score_to_binary(arr, threshold=threshold)


def _target_to_binary(target: Any) -> np.ndarray:
    arr = _to_numpy(target)
    if arr.dtype == bool:
        return arr
    if arr.ndim >= 5 and arr.shape[1] > 1:
        return np.argmax(arr, axis=1) == 1
    if arr.ndim == 4 and 1 < arr.shape[0] <= 4:
        return np.argmax(arr, axis=0) == 1
    return arr > 0.5


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _score_to_binary(scores: np.ndarray, *, threshold: float) -> np.ndarray:
    scores = np.asarray(scores)
    finite = scores[np.isfinite(scores)]
    if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
        scores = 1.0 / (1.0 + np.exp(-scores))
    return scores >= threshold


def _dice_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else (2 * tp) / denom


def _empty_agreement_score(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def _safe_div(num: int | float, denom: int | float) -> float | None:
    return None if denom == 0 else float(num / denom)


def _mean_optional(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return None if not valid else float(sum(valid) / len(valid))


def _aggregate_case_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "dice",
        "iou",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
    ]
    aggregate: dict[str, Any] = {}
    for name in metric_names:
        values = [case[name] for case in cases if case[name] is not None]
        aggregate[f"{name}_mean"] = None if not values else float(sum(values) / len(values))
        aggregate[f"{name}_median"] = None if not values else float(statistics.median(values))
    for name in ["tp", "fp", "tn", "fn", "target_voxels", "pred_voxels"]:
        aggregate[name] = int(sum(case[name] for case in cases))
    return aggregate


def _format_epoch_metrics(summaries: list[EvaluationSummary]) -> str:
    if not summaries:
        return ""
    parts = []
    for summary in summaries:
        dice = _format_optional(summary["dice_mean"])
        accuracy = _format_optional(summary["accuracy_mean"])
        parts.append(f" {summary['split']} Dice: {dice} Acc: {accuracy}.")
    return "".join(parts)


def _format_optional(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4g}"


def _append_loss(path: Path, loss: float) -> None:
    with path.open("a") as fh:
        fh.write(f"{loss:.10g}\n")


def _append_epoch_metrics(path: Path, summary: EvaluationSummary) -> None:
    columns = [
        "epoch",
        "split",
        "n_cases",
        "loss",
        "dice_mean",
        "dice_median",
        "iou_mean",
        "iou_median",
        "accuracy_mean",
        "accuracy_median",
        "balanced_accuracy_mean",
        "balanced_accuracy_median",
        "precision_mean",
        "precision_median",
        "recall_mean",
        "recall_median",
        "specificity_mean",
        "specificity_median",
        "tp",
        "fp",
        "tn",
        "fn",
        "target_voxels",
        "pred_voxels",
    ]
    _append_dict_row(path, columns, summary)


def _append_case_metrics(path: Path, summary: EvaluationSummary) -> None:
    columns = [
        "epoch",
        "split",
        "case_id",
        "loss",
        "dice",
        "iou",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "tp",
        "fp",
        "tn",
        "fn",
        "target_voxels",
        "pred_voxels",
    ]
    for case in summary["cases"]:
        row = {"epoch": summary["epoch"], "split": summary["split"], **case}
        _append_dict_row(path, columns, row)


def _append_dict_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _write_final_metrics(path: Path, summaries: list[EvaluationSummary]) -> None:
    payload = {
        summary["split"]: {
            key: value
            for key, value in summary.items()
            if key != "cases"
        }
        for summary in summaries
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


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
