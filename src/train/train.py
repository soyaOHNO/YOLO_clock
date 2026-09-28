from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
RUNS_ROOT = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train / resume YOLO-Pose for clock detection."
    )

    # 新規学習用
    parser.add_argument(
        "--dataset",
        default="yolo_clock_data_10",
        help="Dataset directory name under data/",
    )
    parser.add_argument(
        "--model",
        default="yolo26n-pose.pt",
        help="Pretrained model for a new training run.",
    )

    # 学習条件
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs. -1 disables periodic checkpoints.",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Run name.",
    )

    parser.add_argument(
        "--cache",
        choices=["false", "ram", "disk"],
        default="false",
    )

    # 再開用
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to last.pt. If specified, resume training from this checkpoint.",
    )

    return parser.parse_args()


def check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.\n"
            'Run: uv run python -c "import torch; '
            "print(torch.cuda.is_available())\""
        )

    print(f"[GPU] {torch.cuda.get_device_name(0)}")


def resume_training(args: argparse.Namespace) -> None:
    checkpoint = Path(args.resume)

    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    print("[MODE] Resume training")
    print(f"[CHECKPOINT] {checkpoint}")
    print(f"[PATIENCE] {args.patience}")
    print(f"[SAVE PERIOD] {args.save_period}")

    model = YOLO(str(checkpoint))

    model.train(
        resume=True,

        # 再開後に変更したい項目
        patience=args.patience,
        save_period=args.save_period,
    )

    save_dir = Path(model.trainer.save_dir)

    print("\nTraining finished.")
    print(f"Run directory: {save_dir}")
    print(f"best.pt: {save_dir / 'weights' / 'best.pt'}")
    print(f"last.pt: {save_dir / 'weights' / 'last.pt'}")


def new_training(args: argparse.Namespace) -> None:
    data_yaml = DATA_ROOT / args.dataset / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml}"
        )

    cache = False if args.cache == "false" else args.cache

    run_name = (
        args.name
        or f"{args.dataset}_{Path(args.model).stem}"
    )

    print("[MODE] New training")
    print(f"[DATA] {data_yaml}")
    print(f"[MODEL] {args.model}")
    print(f"[EPOCHS] {args.epochs}")
    print(f"[PATIENCE] {args.patience}")
    print(f"[SAVE PERIOD] {args.save_period}")

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),

        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        cache=cache,

        project=str(RUNS_ROOT / "pose"),
        name=run_name,

        # 時計画像なので反転しない
        fliplr=0.0,
        flipud=0.0,

        # validation
        val=True,
        plots=True,

        # checkpoint
        save=True,
        patience=args.patience,
        save_period=args.save_period,
    )

    save_dir = Path(model.trainer.save_dir)

    print("\nTraining finished.")
    print(f"Run directory: {save_dir}")
    print(f"best.pt: {save_dir / 'weights' / 'best.pt'}")
    print(f"last.pt: {save_dir / 'weights' / 'last.pt'}")


def main() -> None:
    args = parse_args()

    check_gpu()

    if args.resume:
        resume_training(args)
    else:
        new_training(args)


if __name__ == "__main__":
    main()