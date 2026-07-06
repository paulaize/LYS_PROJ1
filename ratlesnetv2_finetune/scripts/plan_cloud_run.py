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
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-train", action="store_true")
    parser.add_argument("--metrics-threshold", type=float, default=0.5)
    parser.add_argument("--max-train-cases", type=int, default=None)
    parser.add_argument("--max-validation-cases", type=int, default=None)
    parser.add_argument("--max-test-cases", type=int, default=None)
    parser.add_argument("--allow-partial-state-dict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_plan(args.config)
    dataset_root = args.cloud_dataset_root
    if dataset_root is None:
        dataset_root = str(Path(plan["output_root"]) / str(plan["dataset_name"]))

    train_input = Path(dataset_root) / "train"
    validation_input = Path(dataset_root) / "validation"
    test_input = Path(dataset_root) / "test"
    has_validation = bool(plan.get("splits", {}).get("validation"))
    has_test = bool(plan.get("splits", {}).get("test"))
    command_plan = build_cloud_command_plan(
        ratlesnet_repo=args.ratlesnet_repo,
        train_input=train_input,
        validation_input=validation_input if has_validation else None,
        test_input=test_input if has_test else None,
        output_dir=args.cloud_output,
        pretrained_model=args.pretrained_model,
        epochs=args.epochs,
        lr=args.lr,
        gpu=args.gpu,
        load_memory=args.loadMemory,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_train=args.eval_train,
        metrics_threshold=args.metrics_threshold,
        max_train_cases=args.max_train_cases,
        max_validation_cases=args.max_validation_cases,
        max_test_cases=args.max_test_cases,
        allow_partial_state_dict=args.allow_partial_state_dict,
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
