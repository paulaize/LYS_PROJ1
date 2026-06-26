"""Print cloud commands for a prepared RatLesNetV2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ratlesnetv2_finetune.commands import build_cloud_command_plan, format_command
from ratlesnetv2_finetune.dataset import load_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RatLesNetV2 dataset YAML plan")
    parser.add_argument(
        "--ratlesnet-repo",
        default="/content/RatLesNetv2",
        help="Cloud checkout path for the upstream RatLesNetV2 repo",
    )
    parser.add_argument(
        "--cloud-dataset-root",
        default=None,
        help="Prepared dataset root on the cloud VM. Defaults to output_root/dataset_name.",
    )
    parser.add_argument("--cloud-output", default="/content/ratlesnet_runs")
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--loadMemory", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_plan(args.config)
    dataset_root = args.cloud_dataset_root
    if dataset_root is None:
        dataset_root = str(Path(plan["output_root"]) / str(plan["dataset_name"]))

    train_input = Path(dataset_root) / "train"
    validation_input = Path(dataset_root) / "validation"
    has_validation = bool(plan.get("splits", {}).get("validation"))
    command_plan = build_cloud_command_plan(
        ratlesnet_repo=args.ratlesnet_repo,
        train_input=train_input,
        validation_input=validation_input if has_validation else None,
        output_dir=args.cloud_output,
        pretrained_model=args.pretrained_model,
        epochs=args.epochs,
        lr=args.lr,
        gpu=args.gpu,
        load_memory=args.loadMemory,
    )

    print("# Install Python dependencies first, for example:")
    print("pip install -r ratlesnetv2_finetune/requirements-colab.txt")
    print()
    print("# Clone upstream RatLesNetV2:")
    print(format_command(command_plan.clone_command))
    print()
    print("# Start LYS finetuning/training:")
    print(format_command(command_plan.finetune_command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

