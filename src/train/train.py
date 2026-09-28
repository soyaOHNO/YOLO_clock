from ultralytics import YOLO
import os

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_yaml_path = os.path.join(base_dir, "data", "yolo_clock_data_10", "data.yaml")
    model_path = os.path.join(base_dir, "models", "best.pt")

    print(f"=== YOLOv8 学習開始 ===")
    print(f"Data: {data_yaml_path}")
    print(f"Model: {model_path}")

    # Load the model
    model = YOLO(model_path)

    # ハイエンド環境 (RTX 5080 + Ryzen 9 9950X) 向けの設定
    results = model.train(
        data=data_yaml_path,
        epochs=500,            # 十分に学習できるように大幅増加 (改善が止まれば自動ストップします)
        imgsz=640,             # 学習画像の解像度 (より高精細にしたい場合は1024など)
        batch=-1,              # -1にすることでRTX 5080のVRAM限界までバッチサイズを自動最大化 (AutoBatch)
        workers=16,            # 16コア32スレッドのCPUパワーを活かしてデータ読み込みを高速並列化
        patience=50,           # 50エポック連続で精度向上がなければ早期終了 (Early stopping)
        cache=True,            # メモリ(RAM)に余裕がある前提で、データをメモリにキャッシュしてI/Oを爆速化
        project=os.path.join(base_dir, "runs"),
        name="clock_training_highspec",
        device="0"             # RTX 5080 を指定
    )
    
    print("=== 学習完了 ===")

if __name__ == "__main__":
    main()
