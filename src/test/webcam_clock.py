from pathlib import Path
import math
import time

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "runs"
    / "pose"
    / "clock_10000_yolo26s"
    / "weights"
    / "best.pt"
)

CAMERA_ID = 0
CONF = 0.20
IMG_SIZE = 640

# Keypoint index
PIVOT = 0
TIP = 1

CLASS_NAMES = {
    0: "hour",
    1: "minute",
    2: "second",
}


def clock_angle(pivot, tip):
    """
    12時方向を0度として、時計回りに0〜360度を返す。

    OpenCV画像座標:
        x: 右が+
        y: 下が+
    """
    dx = tip[0] - pivot[0]
    dy = tip[1] - pivot[1]

    angle = math.degrees(math.atan2(dx, -dy))

    return angle % 360.0


def circular_diff(a, b):
    """2つの角度の最小差"""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def estimate_time(angles):
    """
    hour/minute/second handの角度から時刻を推定。

    Blender生成データと同じ、

        秒針 = second * 6
        分針 = (minute + second/60) * 6
        時針 = (hour + minute/60) * 30

    を逆算する。
    """

    second = None
    minute = None
    hour = None

    # 秒
    if 2 in angles:
        second = int(round(angles[2] / 6.0)) % 60

    # 分
    if 1 in angles:
        minute_angle = angles[1]

        if second is not None:
            candidates = range(60)

            minute = min(
                candidates,
                key=lambda m: circular_diff(
                    minute_angle,
                    (m + second / 60.0) * 6.0,
                ),
            )
        else:
            minute = int(round(minute_angle / 6.0)) % 60

    # 時
    if 0 in angles:
        hour_angle = angles[0]

        if minute is not None:
            candidates = range(12)

            hour = min(
                candidates,
                key=lambda h: circular_diff(
                    hour_angle,
                    (h + minute / 60.0) * 30.0,
                ),
            )
        else:
            hour = int(hour_angle / 30.0) % 12

    return hour, minute, second


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print(f"Model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        raise RuntimeError(
            f"Web camera could not be opened: CAMERA_ID={CAMERA_ID}"
        )

    previous_time = time.perf_counter()

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Failed to read camera frame.")
            break

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=0,
            verbose=False,
        )

        result = results[0]

        # 各クラスで最もconfidenceの高い検出だけ使用
        detections = {}

        if (
            result.boxes is not None
            and result.keypoints is not None
        ):
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            keypoints = result.keypoints.xy.cpu().numpy()

            for i, cls in enumerate(classes):

                conf = float(confidences[i])

                if (
                    cls not in detections
                    or conf > detections[cls]["conf"]
                ):
                    detections[cls] = {
                        "conf": conf,
                        "keypoints": keypoints[i],
                    }

        angles = {}

        for cls, detection in detections.items():

            kpts = detection["keypoints"]

            pivot = kpts[PIVOT]
            tip = kpts[TIP]

            # 未検出のkpは(0,0)になる可能性がある
            if (
                pivot[0] == 0
                and pivot[1] == 0
            ):
                continue

            if (
                tip[0] == 0
                and tip[1] == 0
            ):
                continue

            angle = clock_angle(
                pivot,
                tip,
            )

            angles[cls] = angle

            # Pivot
            cv2.circle(
                frame,
                (int(pivot[0]), int(pivot[1])),
                6,
                (0, 255, 0),
                -1,
            )

            # Tip
            cv2.circle(
                frame,
                (int(tip[0]), int(tip[1])),
                6,
                (0, 0, 255),
                -1,
            )

            # Pivot -> Tip
            cv2.line(
                frame,
                (int(pivot[0]), int(pivot[1])),
                (int(tip[0]), int(tip[1])),
                (255, 255, 0),
                2,
            )

            name = CLASS_NAMES.get(cls, str(cls))

            cv2.putText(
                frame,
                f"{name}: {angle:.1f} deg",
                (
                    int(tip[0]) + 10,
                    int(tip[1]),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        hour, minute, second = estimate_time(angles)

        if (
            hour is not None
            and minute is not None
        ):
            display_hour = 12 if hour == 0 else hour

            if second is not None:
                clock_text = (
                    f"{display_hour:02d}:"
                    f"{minute:02d}:"
                    f"{second:02d}"
                )
            else:
                clock_text = (
                    f"{display_hour:02d}:"
                    f"{minute:02d}"
                )

            cv2.putText(
                frame,
                clock_text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )
        else:
            cv2.putText(
                frame,
                "Clock not detected",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        # FPS
        current_time = time.perf_counter()

        fps = 1.0 / max(
            current_time - previous_time,
            1e-6,
        )

        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "YOLO Clock Reader",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()