from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick structural check for a YOLO-Pose dataset.")
    parser.add_argument("--dataset", default="yolo_clock_data_10")
    return parser.parse_args()


def check_split(dataset_dir: Path, split: str, expected_values: int) -> bool:
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    labels = sorted(label_dir.glob("*.txt"))

    print(f"[{split}] images={len(images)}, labels={len(labels)}")

    if len(images) == 0:
        print(f"  ERROR: {split} contains no images.")
        return False

    if len(labels) == 0:
        print(f"  ERROR: {split} contains no labels.")
        return False

    ok = True

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    if missing_labels:
        print(f"  ERROR: {len(missing_labels)} images have no label. Example: {missing_labels[:5]}")
        ok = False

    if missing_images:
        print(f"  ERROR: {len(missing_images)} labels have no image. Example: {missing_images[:5]}")
        ok = False

    for label_path in labels:
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue

            values = line.split()

            if len(values) != expected_values:
                print(
                    f"  ERROR: {label_path.name}:{line_no}: "
                    f"{len(values)} values; expected {expected_values}"
                )
                ok = False
                continue

            try:
                class_id = int(values[0])
                nums = [float(v) for v in values[1:]]
            except ValueError:
                print(f"  ERROR: {label_path.name}:{line_no}: non-numeric value")
                ok = False
                continue

            if class_id not in {0, 1, 2}:
                print(f"  ERROR: {label_path.name}:{line_no}: unexpected class id {class_id}")
                ok = False

            # bbox x/y/w/h + keypoint x/y/visibility
            coords = nums[:4]
            if not all(0.0 <= v <= 1.0 for v in coords):
                print(f"  ERROR: {label_path.name}:{line_no}: bbox outside [0,1]")
                ok = False

            kpts = nums[4:]
            for i in range(0, len(kpts), 3):
                x, y, vis = kpts[i : i + 3]
                if vis not in {0.0, 1.0, 2.0}:
                    print(f"  ERROR: {label_path.name}:{line_no}: invalid visibility {vis}")
                    ok = False
                if vis > 0 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    print(f"  ERROR: {label_path.name}:{line_no}: visible keypoint outside [0,1]")
                    ok = False

    return ok


def main() -> None:
    args = parse_args()
    dataset_dir = DATA_ROOT / args.dataset
    yaml_path = dataset_dir / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    kpt_shape = config.get("kpt_shape")

    if not isinstance(kpt_shape, list) or len(kpt_shape) != 2:
        raise ValueError(f"Invalid kpt_shape in {yaml_path}: {kpt_shape}")

    num_keypoints, dims = kpt_shape
    if dims != 3:
        raise ValueError(f"This checker expects visibility labels: kpt_shape second value must be 3, got {dims}")

    # class + bbox(4) + N keypoints * (x,y,visibility)
    expected_values = 1 + 4 + num_keypoints * 3

    print(f"Dataset: {dataset_dir}")
    print(f"kpt_shape: {kpt_shape}")
    print(f"Expected values per label row: {expected_values}")

    ok_train = check_split(dataset_dir, "train", expected_values)
    ok_val = check_split(dataset_dir, "val", expected_values)

    if not (ok_train and ok_val):
        raise SystemExit("\nDataset check FAILED.")

    print("\nDataset check PASSED.")


if __name__ == "__main__":
    main()
