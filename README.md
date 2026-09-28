# YOLO Clock Pose Training

Blenderで生成した時計データセットを使って、Ultralytics YOLO-Poseを学習・評価・推論するための手順です。

## 1. 前提環境

このプロジェクトでは `uv` でPython環境を管理します。

想定環境:

- Windows 11
- Python 3.12
- NVIDIA GeForce RTX 5080
- CUDA対応PyTorch
- Ultralytics YOLO

プロジェクトルートで以下が成功することを確認します。

```powershell
uv --version
```

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

期待する状態は概ね以下です。

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 5080
```

Ultralytics側も確認できます。

```powershell
uv run yolo checks
```

---

## 2. ディレクトリ構成

```text
YOLO_clock/
├─ clock/
│  └─ model_1.blend
│
├─ data/
│  ├─ yolo_clock_data_10/
│  │  ├─ images/
│  │  │  ├─ train/
│  │  │  └─ val/
│  │  ├─ labels/
│  │  │  ├─ train/
│  │  │  └─ val/
│  │  ├─ data.yaml
│  │  └─ times.csv
│  │
│  └─ yolo_clock_data_10000/
│     ├─ images/
│     │  ├─ train/
│     │  └─ val/
│     ├─ labels/
│     │  ├─ train/
│     │  └─ val/
│     ├─ data.yaml
│     └─ times.csv
│
├─ src/
│  ├─ train/
│  │  └─ train.py
│  └─ test/
│     ├─ check_dataset.py
│     ├─ evaluate.py
│     └─ predict.py
│
├─ runs/
├─ pyproject.toml
└─ uv.lock
```

データセットはBlender側で自動生成します。

現在のPoseラベルは6キーポイントです。

```text
0: Pivot
1: Tip
2: Min
3: Mid
4: Max
5: Center
```

クラスは以下です。

```text
0: hour_hand
1: minute_hand
2: second_hand
```

---

# 3. 初回セットアップ

すでに環境構築済みなら、この章は飛ばして構いません。

依存関係を復元します。

```powershell
uv sync
```

GPUを確認します。

```powershell
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

---

# 4. データセット検査

学習前にラベル構造を検査します。

まず10枚版で確認します。

```powershell
uv run python src/test/check_dataset.py --dataset yolo_clock_data_10
```

正常なら最後に、

```text
Dataset check PASSED.
```

と表示されます。

10000枚版も同様です。

```powershell
uv run python src/test/check_dataset.py --dataset yolo_clock_data_10000
```

このチェックでは主に以下を確認します。

- 画像とラベルのファイル名が対応しているか
- `data.yaml` の `kpt_shape` が読めるか
- 1行の要素数がYOLO-Pose形式と一致するか
- bboxが0〜1に正規化されているか
- visibleなキーポイント座標が0〜1に入っているか
- visibility値が0 / 1 / 2か
- class IDが0 / 1 / 2か

---

# 5. 10枚データでスモークテスト

最初から10000枚を学習せず、10枚版でパイプライン全体を確認します。

```powershell
uv run python src/train/train.py `
  --dataset yolo_clock_data_10 `
  --model yolo26n-pose.pt `
  --epochs 5 `
  --batch 16 `
  --name smoke_test
```

PowerShellでは `` ` `` が改行継続です。

1行で実行しても構いません。

```powershell
uv run python src/train/train.py --dataset yolo_clock_data_10 --model yolo26n-pose.pt --epochs 5 --batch 16 --name smoke_test
```

成功すると主に以下が生成されます。

```text
runs/
└─ pose/
   └─ smoke_test/
      ├─ weights/
      │  ├─ best.pt
      │  └─ last.pt
      ├─ results.csv
      ├─ results.png
      └─ ...
```

通常は推論・評価には `best.pt` を使用します。

同名のrunディレクトリが既に存在する場合、Ultralyticsが自動で別名を付けることがあります。実際の保存先は学習終了時に表示される `Run directory` を確認してください。

---

# 6. 10枚データで評価

```powershell
uv run python src/test/evaluate.py `
  --weights runs/pose/smoke_test/weights/best.pt `
  --dataset yolo_clock_data_10 `
  --name smoke_test_val
```

出力先:

```text
runs/val/smoke_test_val/
```

ここでboxとpose/keypointの評価結果を確認します。

10枚版は精度を測るためではなく、データ形式と学習処理が正常に動くことを確認するためのものです。

---

# 7. 推論画像を目視確認

val画像に対して推論し、bbox・クラス・キーポイントを描画した結果を保存します。

```powershell
uv run python src/test/predict.py `
  --weights runs/pose/smoke_test/weights/best.pt `
  --source data/yolo_clock_data_10/images/val `
  --name smoke_test_predict
```

結果:

```text
runs/predict/smoke_test_predict/
```

画像を開いて、以下を確認します。

- 時計盤のbboxが適切か
- hour / minute / secondのクラスが正しいか
- Pivotが回転中心に乗っているか
- Tipが針先に乗っているか
- Min / Mid / Maxが目盛位置に乗っているか
- Centerが文字盤中心に乗っているか

---

# 8. 10000枚データで本学習

10枚版で問題がなければ10000枚版に移ります。

まずデータセットを確認します。

```powershell
uv run python src/test/check_dataset.py --dataset yolo_clock_data_10000
```

## 最初の本学習

まずは `yolo26s-pose.pt` を推奨します。

```powershell
uv run python src/train/train.py `
  --dataset yolo_clock_data_10000 `
  --model yolo26s-pose.pt `
  --epochs 100 `
  --batch -1 `
  --workers 8 `
  --patience 20 `
  --save-period 10 `
  --name clock_10000_yolo26s
```

1行で実行する場合:

```powershell
uv run python src/train/train.py --dataset yolo_clock_data_10000 --model yolo26s-pose.pt --epochs 100 --batch -1 --workers 8 --patience 20 --save-period 10 --name clock_10000_yolo26s
```

`batch=-1` はUltralyticsにGPUメモリ量に応じたbatch sizeを決めさせます。

`patience=20` は、validationのfitnessが20 epoch連続で改善しなかった場合にEarly Stoppingする設定です。

`save-period=10` は10 epochごとにチェックポイントを保存する設定です。

通常は以下が保存されます。

```text
runs/pose/clock_10000_yolo26s/weights/
├─ best.pt
├─ last.pt
├─ epoch10.pt
├─ epoch20.pt
├─ epoch30.pt
└─ ...
```

- `best.pt`: validation上で最も良かったモデル
- `last.pt`: 直近のepoch終了時点
- `epochN.pt`: `save-period` で指定した間隔のチェックポイント

RTX 5080 16GBで余裕がある場合は、後から `yolo26m-pose.pt` も比較します。

```powershell
uv run python src/train/train.py `
  --dataset yolo_clock_data_10000 `
  --model yolo26m-pose.pt `
  --epochs 100 `
  --batch -1 `
  --workers 8 `
  --patience 20 `
  --save-period 10 `
  --name clock_10000_yolo26m
```

---

# 9. 本学習モデルを評価

例:

```powershell
uv run python src/test/evaluate.py `
  --weights runs/pose/clock_10000_yolo26s/weights/best.pt `
  --dataset yolo_clock_data_10000 `
  --name clock_10000_yolo26s_val
```

---

# 10. 本学習モデルで推論

```powershell
uv run python src/test/predict.py `
  --weights runs/pose/clock_10000_yolo26s/weights/best.pt `
  --source data/yolo_clock_data_10000/images/val `
  --name clock_10000_yolo26s_predict
```

---

# 11. 学習の中断・再開

現在の `train.py` は、新規学習とresumeの両方に対応しています。

## 手動で中断する

学習中に停止したい場合は、PowerShellで `Ctrl + C` を押します。

Ultralyticsは各epoch終了時点で `last.pt` を更新するため、直前に完了したepochまでの学習状態は保持されます。

保存先の例:

```text
runs/pose/clock_10000_yolo26s/weights/last.pt
```

## 中断した学習を再開する

以下で `last.pt` から再開します。

```powershell
uv run python src/train/train.py `
  --resume runs/pose/clock_10000_yolo26s/weights/last.pt `
  --patience 20 `
  --save-period 10
```

1行で実行する場合:

```powershell
uv run python src/train/train.py --resume runs/pose/clock_10000_yolo26s/weights/last.pt --patience 20 --save-period 10
```

resume時は、モデル重み・optimizer状態・learning rate scheduler・完了済みepochなどの学習状態を引き継ぎます。

## Early Stopping

`--patience 20` の場合、validationのfitnessが20 epoch連続で改善しなければ、最大epoch数に到達する前でも自動終了します。

## 定期チェックポイント

`--save-period 10` の場合、

```text
epoch10.pt
epoch20.pt
epoch30.pt
...
```

のように途中モデルも保存されます。

手動中断や比較実験を行う場合は、`best.pt` / `last.pt` に加えて途中checkpointも残しておくと便利です。

---

# 12. 実験時に見るファイル

## best.pt

```text
runs/pose/<run_name>/weights/best.pt
```

validation上で最良だったモデルです。基本的にはこれを利用します。

## last.pt

```text
runs/pose/<run_name>/weights/last.pt
```

直近で完了したepochのモデルです。学習を途中で停止した場合のresumeに使用します。

## epochN.pt

`--save-period 10` を指定した場合は、10 epochごとのチェックポイントも保存されます。

```text
runs/pose/<run_name>/weights/epoch10.pt
runs/pose/<run_name>/weights/epoch20.pt
runs/pose/<run_name>/weights/epoch30.pt
```

途中epoch同士の比較や、特定時点のモデルを残したい場合に利用します。

## results.csv

各epochのlossや評価指標が保存されます。

```text
runs/pose/<run_name>/results.csv
```

## results.png

学習曲線をまとめて確認できます。

```text
runs/pose/<run_name>/results.png
```

---

# 13. 推奨する実行順

新しいデータセットを生成したら毎回、以下の順に実行します。

```text
Blenderでデータ生成
        ↓
check_dataset.py
        ↓
10枚または小規模データで短時間学習
        ↓
evaluate.py
        ↓
predict.pyで目視確認
        ↓
10000枚で本学習
        ↓
evaluate.py
        ↓
predict.py
        ↓
実写画像で検証
```

特にSim2Realでは、CG validation精度だけではなく、最終的には実写画像でのキーポイント誤差を確認する必要があります。

---

# 14. よく使うコマンド一覧

## GPU確認

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 10枚データ確認

```powershell
uv run python src/test/check_dataset.py --dataset yolo_clock_data_10
```

## スモークテスト

```powershell
uv run python src/train/train.py --dataset yolo_clock_data_10 --model yolo26n-pose.pt --epochs 5 --batch 16 --name smoke_test
```

## スモークテスト評価

```powershell
uv run python src/test/evaluate.py --weights runs/pose/smoke_test/weights/best.pt --dataset yolo_clock_data_10 --name smoke_test_val
```

## スモークテスト推論

```powershell
uv run python src/test/predict.py --weights runs/pose/smoke_test/weights/best.pt --source data/yolo_clock_data_10/images/val --name smoke_test_predict
```

## 10000枚本学習

```powershell
uv run python src/train/train.py --dataset yolo_clock_data_10000 --model yolo26s-pose.pt --epochs 100 --batch -1 --workers 8 --patience 20 --save-period 10 --name clock_10000_yolo26s
```

## 学習再開

```powershell
uv run python src/train/train.py --resume runs/pose/clock_10000_yolo26s/weights/last.pt --patience 20 --save-period 10
```

## 本学習評価

```powershell
uv run python src/test/evaluate.py --weights runs/pose/clock_10000_yolo26s/weights/best.pt --dataset yolo_clock_data_10000 --name clock_10000_yolo26s_val
```

## 本学習推論

```powershell
uv run python src/test/predict.py --weights runs/pose/clock_10000_yolo26s/weights/best.pt --source data/yolo_clock_data_10000/images/val --name clock_10000_yolo26s_predict
```
