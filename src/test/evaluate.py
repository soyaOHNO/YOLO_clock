from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
RUNS_ROOT = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO-Pose clock model.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to best.pt or last.pt",
    )
    parser.add_argument(
        "--dataset",
        default="yolo_clock_data_10",
        help="Dataset directory name under data/",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="clock_val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights

    data_yaml = DATA_ROOT / args.dataset / "data.yaml"

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(RUNS_ROOT / "val"),
        name=args.name,
        plots=True,
    )

    print("\nValidation finished.")
    print(f"Results: {RUNS_ROOT / 'val' / args.name}")

    # Ultralytics exposes metric objects whose exact attributes may vary by version.
    # Printing the object itself keeps this script compatible while still surfacing metrics.
    print(metrics)


if __name__ == "__main__":
    main()
