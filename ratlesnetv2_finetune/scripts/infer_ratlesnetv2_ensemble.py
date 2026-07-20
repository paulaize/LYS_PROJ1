"""Run the frozen five-model RatLesNetV2 ensemble on unlabeled T2w scans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from ratlesnetv2_finetune.scripts.evaluate_probability_ensemble import (
    _read_threshold_record,
)
from ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 import (
    _case_id_to_str,
    _lesion_probability_map,
    _load_pretrained,
    _patch_nibabel_get_data_compat,
    _transpose_to_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratlesnet-repo", required=True)
    parser.add_argument("--input", required=True, help="Root containing case/scan.nii.gz files")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Frozen RatLesNetv2.model; repeat once per fold",
    )
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--frozen-spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--expected-models", type=int, default=5)
    parser.add_argument(
        "--expected-spacing-mm",
        nargs=3,
        type=float,
        default=(0.07, 0.07, 0.5),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = infer_ratlesnetv2_ensemble(
        ratlesnet_repo=Path(args.ratlesnet_repo),
        input_root=Path(args.input),
        model_paths=[Path(path) for path in args.model],
        threshold_json=Path(args.threshold_json),
        frozen_spec=Path(args.frozen_spec),
        output_root=Path(args.output),
        device_name=args.device,
        expected_models=args.expected_models,
        expected_spacing=tuple(args.expected_spacing_mm),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def infer_ratlesnetv2_ensemble(
    *,
    ratlesnet_repo: Path,
    input_root: Path,
    model_paths: list[Path],
    threshold_json: Path,
    frozen_spec: Path,
    output_root: Path,
    device_name: str = "auto",
    expected_models: int = 5,
    expected_spacing: tuple[float, float, float] = (0.07, 0.07, 0.5),
    overwrite: bool = False,
    dependencies: tuple[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Write native-space mean probabilities and masks without postprocessing."""
    ratlesnet_repo = ratlesnet_repo.resolve()
    input_root = input_root.resolve()
    model_paths = [path.resolve() for path in model_paths]
    threshold_json = threshold_json.resolve()
    frozen_spec = frozen_spec.resolve()
    output_root = output_root.resolve()

    if not ratlesnet_repo.is_dir():
        raise FileNotFoundError(f"RatLesNetV2 source directory not found: {ratlesnet_repo}")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Inference input directory not found: {input_root}")
    if len(model_paths) != expected_models:
        raise ValueError(f"Expected {expected_models} models, received {len(model_paths)}")
    if len(set(model_paths)) != len(model_paths):
        raise ValueError("Each ensemble model path must be different")
    for path in model_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")

    threshold_record = _read_threshold_record(threshold_json)
    threshold = float(threshold_record["selected_threshold"])
    frozen = _validate_frozen_spec(
        frozen_spec,
        model_paths=model_paths,
        threshold=threshold,
        expected_models=expected_models,
    )
    scan_records = _validate_input_scans(input_root, expected_spacing=expected_spacing)

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Inference output already exists: {output_root}")
        shutil.rmtree(output_root)

    if dependencies is None:
        sys.path.insert(0, str(ratlesnet_repo))
        _patch_nibabel_get_data_compat()
        import torch
        from lib.DataWrapper import DataWrapper
        from lib.RatLesNetv2 import RatLesNetv2
    else:
        torch, DataWrapper, RatLesNetv2 = dependencies

    device = _select_device(torch, device_name)
    data = DataWrapper(str(input_root), "test", device, loadMemory=False)
    if len(data) != len(scan_records):
        raise RuntimeError(
            f"Upstream loader found {len(data)} cases but input validation found "
            f"{len(scan_records)}"
        )

    models = []
    for model_path in model_paths:
        model = RatLesNetv2(modalities=1, filters=32)
        model.to(device)
        _load_pretrained(torch, model, model_path, device=device, strict=True)
        model.eval()
        models.append(model)

    output_root.mkdir(parents=True)
    manifest_rows = []
    with torch.no_grad():
        for index in range(len(data)):
            x, _unused_label, raw_case_id = data[index]
            source_dir = Path(str(raw_case_id).rstrip("/\\"))
            case_id = _case_id_to_str(raw_case_id)
            if case_id not in scan_records:
                raise RuntimeError(f"Upstream loader returned unexpected case {case_id!r}")
            scan_path = source_dir / "scan.nii.gz"
            if scan_path.resolve() != scan_records[case_id].resolve():
                raise RuntimeError(f"Input path changed for case {case_id!r}: {scan_path}")
            reference = nib.load(str(scan_path))
            native_shape = tuple(int(value) for value in reference.shape[:3])

            model_probabilities = []
            for model in models:
                prediction = model(x)[0]
                probability = _lesion_probability_map(prediction).astype(np.float32)
                probability = _transpose_to_shape(probability, native_shape)
                finite = probability[np.isfinite(probability)]
                if not finite.size or float(finite.min()) < 0.0 or float(finite.max()) > 1.0:
                    raise ValueError(f"Invalid lesion probabilities for case {case_id!r}")
                model_probabilities.append(probability)

            ensemble_probability = np.mean(
                np.stack(model_probabilities, axis=0),
                axis=0,
                dtype=np.float32,
            )
            ensemble_mask = (ensemble_probability >= threshold).astype(np.uint8)
            case_output = output_root / "cases" / case_id
            case_output.mkdir(parents=True)
            probability_path = case_output / "ensemble_probability.nii.gz"
            mask_path = case_output / "ensemble_mask.nii.gz"
            _save_nifti_like(probability_path, ensemble_probability, reference)
            _save_nifti_like(mask_path, ensemble_mask, reference)
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "input_scan": str(scan_path.resolve()),
                    "ensemble_probability": str(probability_path),
                    "ensemble_mask": str(mask_path),
                    "threshold": threshold,
                    "n_models": expected_models,
                    "ensemble": "unweighted_mean_lesion_probability",
                    "postprocessing": "none",
                }
            )

    manifest_path = output_root / "inference_manifest.csv"
    _write_csv(manifest_path, manifest_rows)
    summary = {
        "architecture": "RatLesNetV2",
        "device": str(device),
        "n_cases": len(manifest_rows),
        "n_models": expected_models,
        "model_sha256": [_sha256(path) for path in model_paths],
        "threshold": threshold,
        "threshold_source": str(threshold_json),
        "frozen_spec": str(frozen_spec),
        "frozen_project_git_commit": frozen.get("project_git_commit"),
        "frozen_ratlesnetv2_git_commit": frozen.get("ratlesnetv2_git_commit"),
        "ensemble": "unweighted_mean_lesion_probability",
        "postprocessing": "none",
        "expected_spacing_mm": list(expected_spacing),
        "predictions_are_drafts": True,
        "manifest": str(manifest_path),
    }
    (output_root / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _validate_frozen_spec(
    path: Path,
    *,
    model_paths: list[Path],
    threshold: float,
    expected_models: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen specification not found: {path}")
    record = json.loads(path.read_text())
    if record.get("architecture") != "RatLesNetV2":
        raise ValueError("Frozen specification architecture is not RatLesNetV2")
    if record.get("ensemble") != "unweighted mean lesion probability":
        raise ValueError("Frozen specification does not select mean-probability ensembling")
    if record.get("postprocessing") != "none":
        raise ValueError("Frozen specification does not select postprocessing=none")
    if not np.isclose(float(record.get("threshold", -1)), threshold, rtol=0, atol=1e-12):
        raise ValueError("Frozen specification and threshold JSON disagree")
    fold_models = sorted(record.get("fold_models", []), key=lambda item: int(item["fold"]))
    if len(fold_models) != expected_models:
        raise ValueError(
            f"Frozen specification contains {len(fold_models)} models, expected {expected_models}"
        )
    expected_hashes = sorted(str(item["sha256"]) for item in fold_models)
    observed_hashes = sorted(_sha256(path) for path in model_paths)
    if observed_hashes != expected_hashes:
        raise ValueError("Downloaded model hashes do not match the frozen specification")
    return record


def _validate_input_scans(
    input_root: Path,
    *,
    expected_spacing: tuple[float, float, float],
) -> dict[str, Path]:
    scans = sorted(input_root.rglob("scan.nii.gz"))
    if not scans:
        raise FileNotFoundError(f"No scan.nii.gz files found under {input_root}")
    records = {}
    for scan_path in scans:
        case_id = scan_path.parent.name
        if case_id in records:
            raise ValueError(f"Duplicate inference case directory name: {case_id!r}")
        image = nib.load(str(scan_path))
        if image.ndim != 4 or image.shape[-1] != 1:
            raise ValueError(
                f"{scan_path}: expected X x Y x slices x 1, got {image.shape}"
            )
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if not np.allclose(spacing, expected_spacing, rtol=0, atol=1e-5):
            raise ValueError(
                f"{scan_path}: spacing {spacing} does not match expected {expected_spacing} mm"
            )
        data = np.asanyarray(image.dataobj)
        finite = data[np.isfinite(data)]
        if finite.size != data.size:
            raise ValueError(f"{scan_path}: scan contains non-finite voxels")
        if not finite.size or float(finite.std()) == 0.0:
            raise ValueError(f"{scan_path}: scan has zero intensity variance")
        if not np.isfinite(image.affine).all() or np.linalg.det(image.affine[:3, :3]) == 0:
            raise ValueError(f"{scan_path}: invalid NIfTI affine")
        records[case_id] = scan_path
    return records


def _select_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def _save_nifti_like(path: Path, data: np.ndarray, reference: Any) -> None:
    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    header.set_zooms(tuple(float(value) for value in reference.header.get_zooms()[:3]))
    nib.save(nib.Nifti1Image(data, reference.affine, header), path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty inference manifest")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
