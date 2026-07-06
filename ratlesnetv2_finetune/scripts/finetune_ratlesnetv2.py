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
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PLOT_WARNING_SHOWN = False


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
    parser.add_argument(
        "--require-pretrained",
        action="store_true",
        help="Fail instead of training from scratch when --pretrained-model is missing.",
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
        "--eval-only",
        action="store_true",
        help="Load the model and evaluate validation/test splits without training.",
    )
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
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable PNG metric plots. By default plots are updated when metrics are written.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop after N validation evaluations without Dice improvement.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation Dice improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--export-predictions",
        default="",
        help="Comma-separated splits for prediction exports: train,validation,test.",
    )
    parser.add_argument(
        "--export-prediction-limit",
        type=int,
        default=8,
        help="Maximum cases per split/epoch to export when --export-predictions is set.",
    )
    parser.add_argument(
        "--export-prediction-epochs",
        default="1,2,5,final",
        help="Comma-separated epochs to export, plus optional 'final' or 'all'.",
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
    export_splits = _parse_export_splits(args.export_predictions)
    export_epochs = _parse_export_epochs(args.export_prediction_epochs)
    if export_splits and args.export_prediction_limit < 1:
        raise ValueError("--export-prediction-limit must be >= 1 when exporting predictions")
    if args.early_stop_patience is not None and args.early_stop_patience < 1:
        raise ValueError("--early-stop-patience must be >= 1")

    model = RatLesNetv2(modalities=args.modalities, filters=args.filters)
    model.to(device)
    if args.require_pretrained and not args.pretrained_model:
        raise ValueError(
            "--require-pretrained was set, but --pretrained-model was not provided. "
            "Pass the upstream RatLesNetV2 weights path to fine-tune instead of "
            "training from scratch."
        )
    if args.pretrained_model:
        _load_pretrained(
            torch,
            model,
            Path(args.pretrained_model),
            device=device,
            strict=not args.allow_partial_state_dict,
        )
        print(now() + f"Loaded pretrained model: {args.pretrained_model}")
    else:
        print(now() + "No --pretrained-model supplied; initializing model from scratch.")
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

    if args.eval_only:
        summaries = _evaluate_requested_splits(
            model=model,
            loss_fn=CrossEntropyDiceLoss,
            torch=torch,
            threshold=args.metrics_threshold,
            epoch=0,
            validation_data=val_data,
            test_data=test_data,
        )
        if not summaries:
            raise ValueError("--eval-only requires --validation and/or --test")
        for summary in summaries:
            _append_epoch_metrics(run_dir / "metrics_epoch.csv", summary)
            _append_case_metrics(run_dir / "metrics_cases.csv", summary)
        _export_requested_predictions(
            model=model,
            torch=torch,
            threshold=args.metrics_threshold,
            run_dir=run_dir,
            epoch=0,
            is_final=True,
            export_splits=export_splits,
            export_epochs=export_epochs,
            export_limit=args.export_prediction_limit,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )
        _write_final_metrics(run_dir / "final_metrics.json", summaries)
        if not args.no_plots:
            _write_metric_plots(run_dir)
        _write_run_status(
            run_dir / "run_status.json",
            status="evaluated",
            completed_epoch=0,
            reason="eval_only",
        )
        print(now() + f"Evaluation-only run complete: {run_dir}")
        return 0

    best_state: dict[str, dict[str, Any]] = {}
    no_dice_improvement = 0
    completed_epoch = 0
    current_epoch = 0
    stopped_early = False
    stop_reason = ""
    print(now() + f"Start training for {args.epochs} epochs")
    try:
        for epoch in range(args.epochs):
            epoch_num = epoch + 1
            current_epoch = epoch_num
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
                improved = _update_best_checkpoints(
                    torch=torch,
                    model=model,
                    run_dir=run_dir,
                    summary=val_eval,
                    best_state=best_state,
                    min_dice_delta=args.early_stop_min_delta,
                )
                no_dice_improvement = 0 if improved else no_dice_improvement + 1
            for summary in epoch_summaries:
                _append_epoch_metrics(run_dir / "metrics_epoch.csv", summary)
                _append_case_metrics(run_dir / "metrics_cases.csv", summary)
            if epoch_summaries:
                _export_requested_predictions(
                    model=model,
                    torch=torch,
                    threshold=args.metrics_threshold,
                    run_dir=run_dir,
                    epoch=epoch_num,
                    is_final=False,
                    export_splits=export_splits,
                    export_epochs=export_epochs,
                    export_limit=args.export_prediction_limit,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                )
            if epoch_summaries and not args.no_plots:
                _write_metric_plots(run_dir)

            val_text = "" if val_loss is None else f" Val Loss: {val_loss:.8g}."
            metrics_text = _format_epoch_metrics(epoch_summaries)
            print(
                now()
                + f"Epoch: {epoch_num}. Loss: {train_loss:.8g}.{val_text}{metrics_text}"
            )

            if args.save_every > 0 and epoch_num % args.save_every == 0:
                torch.save(model.state_dict(), run_dir / f"RatLesNetv2_epoch{epoch_num:03d}.model")
            torch.save(model.state_dict(), run_dir / "last.model")
            completed_epoch = epoch_num
            if (
                args.early_stop_patience is not None
                and val_data is not None
                and should_eval
                and no_dice_improvement >= args.early_stop_patience
            ):
                stopped_early = True
                stop_reason = (
                    f"Early stopping after {no_dice_improvement} validation evaluations "
                    "without Dice improvement."
                )
                print(now() + stop_reason)
                break
    except KeyboardInterrupt:
        torch.save(model.state_dict(), run_dir / "interrupted.model")
        torch.save(model.state_dict(), run_dir / "last.model")
        _write_run_status(
            run_dir / "run_status.json",
            status="interrupted",
            completed_epoch=completed_epoch,
            interrupted_epoch=current_epoch,
            reason="KeyboardInterrupt",
        )
        if not args.no_plots:
            _write_metric_plots(run_dir)
        print(
            now()
            + "Training interrupted by Ctrl-C. Saved interrupted.model, "
            f"last.model, and run_status.json in {run_dir}"
        )
        return 130

    torch.save(model.state_dict(), run_dir / "RatLesNetv2.model")
    print(now() + f"Saved final model: {run_dir / 'RatLesNetv2.model'}")

    final_summaries: list[EvaluationSummary] = []
    if val_data is not None:
        final_summaries.append(
            _run_evaluation(
                split="validation_final",
                epoch=completed_epoch,
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
                epoch=completed_epoch,
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
    _export_requested_predictions(
        model=model,
        torch=torch,
        threshold=args.metrics_threshold,
        run_dir=run_dir,
        epoch=completed_epoch,
        is_final=True,
        export_splits=export_splits,
        export_epochs=export_epochs,
        export_limit=args.export_prediction_limit,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )
    if final_summaries:
        _write_final_metrics(run_dir / "final_metrics.json", final_summaries)
        if not args.no_plots:
            _write_metric_plots(run_dir)
        print(now() + f"Wrote metrics: {run_dir / 'metrics_epoch.csv'}")
    else:
        print(
            now()
            + "No validation/test split was provided; wrote training loss only. "
            "Do not treat this run as a performance estimate."
        )
    _write_run_status(
        run_dir / "run_status.json",
        status="early_stopped" if stopped_early else "completed",
        completed_epoch=completed_epoch,
        reason=stop_reason,
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


def _evaluate_requested_splits(
    *,
    model: Any,
    loss_fn: Any,
    torch: Any,
    threshold: float,
    epoch: int,
    validation_data: Any | None,
    test_data: Any | None,
) -> list[EvaluationSummary]:
    summaries: list[EvaluationSummary] = []
    if validation_data is not None:
        summaries.append(
            _run_evaluation(
                split="validation",
                epoch=epoch,
                model=model,
                data=validation_data,
                loss_fn=loss_fn,
                torch=torch,
                threshold=threshold,
            )
        )
    if test_data is not None:
        summaries.append(
            _run_evaluation(
                split="test",
                epoch=epoch,
                model=model,
                data=test_data,
                loss_fn=loss_fn,
                torch=torch,
                threshold=threshold,
            )
        )
    return summaries


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
            torch.save(model.state_dict(), run_dir / "best_by_validation_dice.model")
            best_state["validation_dice"] = _best_checkpoint_record(
                summary=summary,
                metric="dice_mean",
                value=float(dice),
                filename="best_by_validation_dice.model",
            )
            improved_dice = True

    previous_loss = best_state.get("validation_loss", {}).get("value")
    if previous_loss is None or loss < float(previous_loss):
        torch.save(model.state_dict(), run_dir / "best_by_validation_loss.model")
        best_state["validation_loss"] = _best_checkpoint_record(
            summary=summary,
            metric="loss",
            value=loss,
            filename="best_by_validation_loss.model",
        )
    _write_best_checkpoint_metadata(run_dir / "best_checkpoints.json", best_state)
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


def _write_best_checkpoint_metadata(path: Path, best_state: dict[str, dict[str, Any]]) -> None:
    with path.open("w") as fh:
        json.dump(best_state, fh, indent=2, sort_keys=True)


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


def _lesion_probability_map(pred: Any) -> np.ndarray:
    arr = _to_numpy(pred)
    if arr.ndim >= 5 and arr.shape[1] > 1:
        return np.squeeze(_softmax(arr, axis=1)[:, 1, ...])
    if arr.ndim >= 5 and arr.shape[1] == 1:
        return np.squeeze(_score_to_probability(arr[:, 0, ...]))
    if arr.ndim == 4 and 1 < arr.shape[0] <= 4:
        return np.squeeze(_softmax(arr, axis=0)[1, ...])
    if arr.ndim == 4 and arr.shape[0] == 1:
        return np.squeeze(_score_to_probability(arr[0, ...]))
    return np.squeeze(_score_to_probability(arr))


def _softmax(value: np.ndarray, *, axis: int) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _score_to_binary(scores: np.ndarray, *, threshold: float) -> np.ndarray:
    return _score_to_probability(scores) >= threshold


def _score_to_probability(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    finite = scores[np.isfinite(scores)]
    if finite.size and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
        clipped = np.clip(scores, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    return scores


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
    _write_json(path, payload)


def _write_run_status(
    path: Path,
    *,
    status: str,
    completed_epoch: int,
    interrupted_epoch: int | None = None,
    reason: str = "",
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "completed_epoch": int(completed_epoch),
        "reason": reason,
    }
    if interrupted_epoch is not None:
        payload["interrupted_epoch"] = int(interrupted_epoch)
    _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _parse_export_splits(value: str) -> set[str]:
    if value in {"", "none", "None", None}:
        return set()
    splits = {item.strip() for item in str(value).split(",") if item.strip()}
    valid = {"train", "validation", "test"}
    unknown = sorted(splits - valid)
    if unknown:
        raise ValueError(f"Unknown --export-predictions split(s): {', '.join(unknown)}")
    return splits


def _parse_export_epochs(value: str) -> set[int | str]:
    epochs: set[int | str] = set()
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"final", "all"}:
            epochs.add(item)
            continue
        epoch = int(item)
        if epoch < 0:
            raise ValueError("--export-prediction-epochs values must be >= 0")
        epochs.add(epoch)
    return epochs or {"final"}


def _should_export_epoch(epoch: int, *, is_final: bool, export_epochs: set[int | str]) -> bool:
    return (
        "all" in export_epochs
        or epoch in export_epochs
        or (is_final and "final" in export_epochs)
    )


def _export_requested_predictions(
    *,
    model: Any,
    torch: Any,
    threshold: float,
    run_dir: Path,
    epoch: int,
    is_final: bool,
    export_splits: set[str],
    export_epochs: set[int | str],
    export_limit: int,
    train_data: Any | None,
    val_data: Any | None,
    test_data: Any | None,
) -> None:
    should_export = _should_export_epoch(
        epoch,
        is_final=is_final,
        export_epochs=export_epochs,
    )
    if not export_splits or not should_export:
        return
    split_data = {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }
    label = "final" if is_final else f"epoch_{epoch:03d}"
    for split in sorted(export_splits):
        data = split_data[split]
        if data is None:
            continue
        _export_predictions_for_split(
            model=model,
            data=data,
            torch=torch,
            threshold=threshold,
            run_dir=run_dir,
            out_dir=run_dir / "prediction_exports" / split / label,
            split=split,
            epoch=epoch,
            label=label,
            limit=export_limit,
        )


def _export_predictions_for_split(
    *,
    model: Any,
    data: Any,
    torch: Any,
    threshold: float,
    run_dir: Path,
    out_dir: Path,
    split: str,
    epoch: int,
    label: str,
    limit: int,
) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for index in range(min(len(data), limit)):
            x, y, case_id = data[index]
            pred = model(x)[0]
            case_id_text = _case_id_to_str(case_id)
            record = _save_prediction_artifacts(
                x=x,
                y=y,
                pred=pred,
                data=data,
                index=index,
                case_id=case_id_text,
                out_dir=out_dir / _safe_path_part(case_id_text),
                threshold=threshold,
            )
            record["split"] = split
            record["epoch"] = epoch
            records.append(record)
    if records:
        _write_prediction_export_manifest(out_dir / "prediction_export_manifest.csv", records)
        _publish_latest_qc_overlay(run_dir, records[0], split=split, epoch=epoch, label=label)


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
    split_metadata = {
        "split": split,
        "epoch": int(epoch),
        "label": label,
        "case_id": record.get("case_id", ""),
        "source_overlay": str(overlay),
        "latest_overlay": str(split_dest),
    }
    _write_json(run_dir / f"latest_{split_name}_qc_overlay.json", split_metadata)

    default_dest = run_dir / "latest_qc_overlay.png"
    if split == "validation" or not default_dest.exists():
        shutil.copyfile(overlay, default_dest)
        default_metadata = dict(split_metadata)
        default_metadata["latest_overlay"] = str(default_dest)
        _write_json(run_dir / "latest_qc_overlay.json", default_metadata)


def _save_prediction_artifacts(
    *,
    x: Any,
    y: Any,
    pred: Any,
    data: Any,
    index: int,
    case_id: str,
    out_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    import nibabel as nib

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_mask = np.squeeze(_prediction_to_binary(pred, threshold=threshold)).astype(np.uint8)
    target_mask = np.squeeze(_target_to_binary(y)).astype(np.uint8)
    probability = _lesion_probability_map(pred).astype(np.float32)

    source_dir = _source_case_dir(data, index)
    ref_img = _load_reference_image(source_dir, expected_shape=pred_mask.shape)
    scan = (
        _scan_volume_from_reference(ref_img)
        if ref_img is not None
        else _scan_volume_from_tensor(x)
    )
    if tuple(scan.shape) != tuple(pred_mask.shape):
        scan = _scan_volume_from_tensor(x)
    if tuple(scan.shape) != tuple(pred_mask.shape):
        scan = np.zeros(pred_mask.shape, dtype=np.float32)
    if ref_img is None or tuple(ref_img.shape[:3]) != tuple(pred_mask.shape):
        ref_img = nib.Nifti1Image(np.zeros(pred_mask.shape, dtype=np.float32), np.eye(4))

    _save_nifti_like(out_dir / "scan.nii.gz", scan.astype(np.float32, copy=False), ref_img)
    _save_nifti_like(out_dir / "target_mask.nii.gz", target_mask, ref_img)
    _save_nifti_like(out_dir / "pred_mask.nii.gz", pred_mask, ref_img)
    _save_nifti_like(out_dir / "lesion_probability.nii.gz", probability, ref_img)
    _write_overlay_png(out_dir / "overlay.png", scan, target_mask, pred_mask)
    return {
        "case_id": case_id,
        "case_dir": str(source_dir) if source_dir is not None else "",
        "export_dir": str(out_dir),
        "scan": str(out_dir / "scan.nii.gz"),
        "target_mask": str(out_dir / "target_mask.nii.gz"),
        "pred_mask": str(out_dir / "pred_mask.nii.gz"),
        "lesion_probability": str(out_dir / "lesion_probability.nii.gz"),
        "overlay": str(out_dir / "overlay.png"),
    }


def _source_case_dir(data: Any, index: int) -> Path | None:
    entries = getattr(data, "list", None)
    if entries is None or index >= len(entries):
        return None
    raw = entries[index]
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    path = Path(str(raw))
    return path if path.is_dir() else path.parent if path.exists() else None


def _load_reference_image(
    source_dir: Path | None,
    *,
    expected_shape: tuple[int, ...],
) -> Any | None:
    if source_dir is None:
        return None
    import nibabel as nib

    for name in ["scan.nii.gz", "scan.nii"]:
        path = source_dir / name
        if path.exists():
            img = nib.load(str(path))
            if tuple(img.shape[:3]) == tuple(expected_shape):
                return img
    return None


def _scan_volume_from_reference(ref_img: Any) -> np.ndarray:
    data = np.asanyarray(ref_img.dataobj)
    return _scan_volume_from_array(data)


def _scan_volume_from_tensor(value: Any) -> np.ndarray:
    return _scan_volume_from_array(_to_numpy(value))


def _scan_volume_from_array(value: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(value))
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0, ...]
    elif arr.ndim == 4 and arr.shape[-1] <= 4:
        arr = arr[..., 0]
    elif arr.ndim == 4 and arr.shape[0] <= 4:
        arr = arr[0, ...]
    return np.squeeze(arr).astype(np.float32, copy=False)


def _save_nifti_like(path: Path, data: np.ndarray, reference: Any) -> None:
    import nibabel as nib

    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    zooms = tuple(float(v) for v in reference.header.get_zooms()[: data.ndim])
    if len(zooms) == data.ndim:
        header.set_zooms(zooms)
    nib.save(nib.Nifti1Image(data, reference.affine, header), path)


def _write_overlay_png(path: Path, scan: np.ndarray, target: np.ndarray, pred: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    slice_index = _representative_slice(target, pred)
    image = _normalize_for_overlay(scan[:, :, slice_index])
    target_slice = target[:, :, slice_index].astype(bool)
    pred_slice = pred[:, :, slice_index].astype(bool)
    rgb = np.stack([image, image, image], axis=-1)
    rgb[target_slice, 1] = 1.0
    rgb[target_slice, 0] *= 0.35
    rgb[target_slice, 2] *= 0.35
    rgb[pred_slice, 0] = 1.0
    rgb[pred_slice, 1] *= 0.35
    rgb[pred_slice, 2] *= 0.35
    overlap = np.logical_and(target_slice, pred_slice)
    rgb[overlap] = np.array([1.0, 1.0, 0.0])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.rot90(rgb), interpolation="nearest")
    ax.set_title(f"slice {slice_index}: target green, prediction red, overlap yellow")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _representative_slice(target: np.ndarray, pred: np.ndarray) -> int:
    counts = target.astype(bool).sum(axis=(0, 1)) + pred.astype(bool).sum(axis=(0, 1))
    return int(np.argmax(counts)) if counts.size else 0


def _normalize_for_overlay(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "case"


def _write_prediction_export_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "epoch",
        "split",
        "case_id",
        "case_dir",
        "export_dir",
        "scan",
        "target_mask",
        "pred_mask",
        "lesion_probability",
        "overlay",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _write_metric_plots(run_dir: Path) -> None:
    global _PLOT_WARNING_SHOWN
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if not _PLOT_WARNING_SHOWN:
            print("matplotlib is not available; skipping metric PNG plots.")
            _PLOT_WARNING_SHOWN = True
        return

    epoch_rows = _read_metric_rows(run_dir / "metrics_epoch.csv")
    training_loss = _read_loss_series(run_dir / "training_loss")
    if training_loss or epoch_rows:
        _plot_loss_curves(run_dir / "loss_curves.png", training_loss, epoch_rows, plt)
    if epoch_rows:
        _plot_metric_curves(run_dir / "validation_metric_curves.png", epoch_rows, plt)
        _plot_voxel_count_curves(run_dir / "voxel_count_curves.png", epoch_rows, plt)
        _plot_final_metric_bars(run_dir / "final_metric_summary.png", epoch_rows, plt)


def _read_loss_series(path: Path) -> list[float]:
    if not path.exists():
        return []
    values: list[float] = []
    with path.open() as fh:
        for raw in fh:
            text = raw.strip()
            if text:
                values.append(float(text))
    return values


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["epoch"] = int(row["epoch"])
        for key, value in list(row.items()):
            if key in {"epoch", "split"} or value == "":
                continue
            row[key] = float(value)
    return rows


def _plot_loss_curves(
    path: Path,
    training_loss: list[float],
    rows: list[dict[str, Any]],
    plt: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if training_loss:
        ax.plot(range(1, len(training_loss) + 1), training_loss, label="train loss")
    for split in ["validation", "train"]:
        split_rows = _rows_for_split(rows, split)
        if split_rows:
            ax.plot(
                [row["epoch"] for row in split_rows],
                [row["loss"] for row in split_rows],
                marker="o",
                markersize=3,
                label=f"{split} eval loss",
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Evaluation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_metric_curves(path: Path, rows: list[dict[str, Any]], plt: Any) -> None:
    metrics = [
        ("dice_mean", "Dice"),
        ("iou_mean", "IoU"),
        ("precision_mean", "Precision"),
        ("recall_mean", "Recall"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        for split in ["validation", "train"]:
            split_rows = _rows_for_split(rows, split)
            split_rows = [row for row in split_rows if row.get(metric) is not None]
            if split_rows:
                ax.plot(
                    [row["epoch"] for row in split_rows],
                    [row[metric] for row in split_rows],
                    marker="o",
                    markersize=3,
                    label=split,
                )
        ax.set_title(label)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")
    axes[0, 0].legend()
    fig.suptitle("Segmentation Metrics During Training", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_voxel_count_curves(path: Path, rows: list[dict[str, Any]], plt: Any) -> None:
    split_rows = _rows_for_split(rows, "validation")
    if not split_rows:
        return
    epochs = [row["epoch"] for row in split_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(epochs, [row["target_voxels"] for row in split_rows], label="target voxels")
    axes[0].plot(epochs, [row["pred_voxels"] for row in split_rows], label="pred voxels")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Voxel count")
    axes[0].set_title("Validation Lesion Voxels")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row.get("precision_mean", 0.0) for row in split_rows], label="precision")
    axes[1].plot(epochs, [row.get("recall_mean", 0.0) for row in split_rows], label="recall")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Validation Precision/Recall")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_final_metric_bars(path: Path, rows: list[dict[str, Any]], plt: Any) -> None:
    final_rows = [
        row
        for row in rows
        if row["split"] in {"validation_final", "test"}
    ]
    if not final_rows:
        return
    metrics = [
        ("dice_mean", "Dice"),
        ("iou_mean", "IoU"),
        ("precision_mean", "Precision"),
        ("recall_mean", "Recall"),
    ]
    labels = [row["split"] for row in final_rows]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, (metric, label) in enumerate(metrics):
        offsets = x + (index - 1.5) * width
        ax.bar(offsets, [row.get(metric, 0.0) for row in final_rows], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Final Evaluation Summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _rows_for_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


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
