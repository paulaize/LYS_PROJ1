"""Cloud-oriented RatLesNetV2 finetuning entry point.

This script intentionally depends on an external checkout of
https://github.com/jmlipman/RatLesNetv2. It mirrors the upstream training loop
but adds the pieces needed for transfer learning: explicit epoch/lr controls,
optional pretrained weights, validation loss output, and checkpointing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratlesnet-repo", required=True, help="Path to upstream RatLesNetV2")
    parser.add_argument("--input", required=True, help="Prepared train split root")
    parser.add_argument("--validation", default=None, help="Prepared validation split root")
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
    parser.add_argument("--max-train-cases", type=int, default=None, help="Small cloud smoke test")
    parser.add_argument(
        "--max-validation-cases",
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

    import numpy as np
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
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _next_run_dir(output_root)

    print(now() + f"Using device: {device}")
    print(now() + f"Loading train data from {train_input}")
    load_memory = args.loadMemory == 1
    train_data = DataWrapper(str(train_input), "train", device, loadMemory=load_memory)
    _limit_wrapper(train_data, args.max_train_cases)
    if validation_input is not None:
        print(now() + f"Loading validation data from {validation_input}")
        val_data = DataWrapper(str(validation_input), "validation", device, loadMemory=load_memory)
        _limit_wrapper(val_data, args.max_validation_cases)
    else:
        val_data = None

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
    )

    print(now() + f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        train_loss = _run_epoch(
            model=model,
            data=train_data,
            loss_fn=CrossEntropyDiceLoss,
            optimizer=optimizer,
        )
        _append_loss(run_dir / "training_loss", train_loss)

        val_loss: float | None = None
        if val_data is not None:
            val_loss = _run_validation(
                model=model,
                data=val_data,
                loss_fn=CrossEntropyDiceLoss,
                torch=torch,
            )
            _append_loss(run_dir / "validation_loss", val_loss)

        val_text = "" if val_loss is None else f" Val Loss: {val_loss:.8g}."
        print(now() + f"Epoch: {epoch}. Loss: {train_loss:.8g}.{val_text}")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), run_dir / f"RatLesNetv2_epoch{epoch + 1:03d}.model")

    torch.save(model.state_dict(), run_dir / "RatLesNetv2.model")
    print(now() + f"Saved final model: {run_dir / 'RatLesNetv2.model'}")
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
    total = None
    for i in range(len(data)):
        x, y, _id = data[i]
        pred = model(x)[0]
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total = loss.detach() if total is None else total + loss.detach()
    return float((total / len(data)).cpu().numpy())


def _run_validation(*, model: Any, data: Any, loss_fn: Any, torch: Any) -> float:
    model.eval()
    total = None
    with torch.no_grad():
        for i in range(len(data)):
            x, y, _id = data[i]
            pred = model(x)[0]
            loss = loss_fn(pred, y)
            total = loss.detach() if total is None else total + loss.detach()
    return float((total / len(data)).cpu().numpy())


def _append_loss(path: Path, loss: float) -> None:
    with path.open("a") as fh:
        fh.write(f"{loss:.10g}\n")


def _write_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    device: Any,
    train_cases: int,
    val_cases: int,
) -> None:
    payload = vars(args).copy()
    payload["device"] = str(device)
    payload["train_cases"] = train_cases
    payload["validation_cases"] = val_cases
    with (run_dir / "run_config.json").open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
