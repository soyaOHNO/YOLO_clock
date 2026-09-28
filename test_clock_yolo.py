import cv2
import os
import glob
import math
import numpy as np
from ultralytics import YOLO

# ==========================================
# 設定パス
# ==========================================
IMAGE_DIR = r"yolo_clock_data_500\images\val"
MODEL_PATH = r"best.pt"

CLOCK_INFO = {
    0: {"name": "Hour", "max": 12.0, "color": (0, 150, 255)},   # オレンジ
    1: {"name": "Min",  "max": 60.0, "color": (255, 255, 0)},   # シアン
    2: {"name": "Sec",  "max": 60.0, "color": (0, 0, 255)}      # 赤
}

def apply_perspective_transform(bbox, kpts):
    x1, y1, x2, y2 = bbox
    src_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    side = max(x2-x1, y2-y1)
    dst_pts = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed_kpts = cv2.perspectiveTransform(kpts.reshape(-1, 1, 2), M).reshape(-1, 2)
    return transformed_kpts

def calculate_angle_clockwise(p1, p2):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    return angle if angle >= 0 else angle + 360

def main():
    print("=== YOLOv8 時計推論テスト開始 ===")
    model = YOLO(MODEL_PATH)
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        
    if not image_paths:
        print(f"エラー: {IMAGE_DIR} に画像が見つかりません。")
        return

    cv2.namedWindow("YOLO Clock Test", cv2.WINDOW_NORMAL)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue

        # 推論実行
        results = model.predict(source=img, conf=0.5, save=False)
        result = results[0]

        time_dict = {}

        if result.boxes and result.keypoints:
            boxes = result.boxes.xyxy.cpu().numpy()
            kpts_all = result.keypoints.xy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                cls_id = int(classes[i])
                bbox = boxes[i]
                p_kpts = kpts_all[i]
                
                # キーポイントが不完全な場合はスキップ
                if len(p_kpts) < 6 or np.all(p_kpts == 0):
                    continue

                pivot, tip, min_pt, mid_pt, max_pt, center = p_kpts

                # 針ベクトルの平行移動
                needle_vector = tip - pivot
                shifted_tip = center + needle_vector
                
                # 射影変換（真円空間での角度取得）
                target_pts = np.array([center, shifted_tip, min_pt], dtype=np.float32)
                t_center, t_tip, t_min = apply_perspective_transform(bbox, target_pts)

                # 12時を基準とした角度計算
                ang_12 = calculate_angle_clockwise(t_center, t_min)
                ang_tip = calculate_angle_clockwise(t_center, t_tip)
                theta = (ang_tip - ang_12) % 360
                
                info = CLOCK_INFO.get(cls_id, {"name": "Unk", "max": 60, "color": (255,255,255)})
                val = (theta / 360.0) * info["max"]
                time_dict[cls_id] = val
                
                # --- 描画処理 ---
                raw_center = tuple(center.astype(int))
                raw_shifted_tip = tuple(shifted_tip.astype(int))
                
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (100, 100, 100), 1)
                cv2.line(img, tuple(pivot.astype(int)), tuple(tip.astype(int)), info["color"], 2)
                cv2.line(img, raw_center, raw_shifted_tip, (255, 255, 255), 1)
                cv2.circle(img, tuple(min_pt.astype(int)), 6, (0, 255, 0), -1)
                
                # キーポイント（赤丸）
                for pt in p_kpts:
                    cv2.circle(img, tuple(pt.astype(int)), 3, (0, 0, 255), -1)
                
                cv2.putText(img, f"{val:.1f}", (raw_shifted_tip[0], raw_shifted_tip[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, info["color"], 2)
        
        # 予測された総合時刻の表示
        if time_dict:
            hh = int(time_dict.get(0, 0))
            mm = int(time_dict.get(1, 0))
            ss = int(time_dict.get(2, 0))
            time_str = f"Predict: {hh:02d}:{mm:02d}:{ss:02d}"
            cv2.putText(img, time_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(img, time_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)

        cv2.imshow("YOLO Clock Test", img)
        if cv2.waitKey(0) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()