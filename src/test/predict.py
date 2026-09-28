from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
RUNS_ROOT = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on clock images and save visualized predictions.")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument(
        "--source",
        default="data/yolo_clock_data_10/images/val",
        help="Image, directory, video, or other Ultralytics-compatible source",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--name", default="clock_predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    model = YOLO(str(weights))

    model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(RUNS_ROOT / "predict"),
        name=args.name,
    )

    print(f"Predictions saved to: {RUNS_ROOT / 'predict' / args.name}")


if __name__ == "__main__":
    main()
