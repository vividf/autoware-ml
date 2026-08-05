# 單 batch 過擬合對拍工具（cross-framework parity probe）

> 用途：當 autoware-ml 與 AWML（或任何兩個框架）在**同一份資料、同一份權重**
> 下訓練結果不同，而 config 逐項比對已經找不出差異時，用這個工具把
> 「資料層面」與「優化層面」的差異一刀切開 —— **幾分鐘**，不用 13 小時重訓。

原理：兩邊各自從**同一份權重**出發，對**同一個固定 batch** 反覆訓練 N 步，
記錄每步的 loss，然後比較兩條軌跡。所有隨機性預設關閉（增強、相機順序
shuffle、GridMask、DN、LR schedule、時序 memory），所以軌跡是可重現的。

判讀邏輯：

| 觀察 | 結論 |
| --- | --- |
| step-0 的 loss 就不同 | 兩邊**看到的輸入不同**，或前向/loss 實作不同 —— 先看 fingerprint |
| step-0 相同、軌跡分岔 | **優化層面**不同（參數分組、LR 解析、梯度處理、loss 正規化） |
| 兩者都相同 | 差異不在本工具涵蓋範圍 → 逐個打開被關掉的元件（`--keep-*`）二分搜尋 |

---

## 1. 組成

| 檔案 | 跑在哪 | 作用 |
| --- | --- | --- |
| `autoware_ml/tools/overfit_probe.py` | autoware-ml 容器 | 產生 autoware-ml 側的 loss trace |
| `tools/detection3d/overfit_probe.py`（AWML repo） | AWML 容器 | 產生 AWML 側的 loss trace |
| `autoware_ml/tools/compare_overfit.py` | **host**（純標準庫） | 比對兩條 trace 並給判決 |

trace 是 JSONL：第一行是 metadata（含 batch fingerprint），之後每行一步的
所有 loss 項。

---

## 2. 使用步驟

### Step 1：準備同一份起始權重

兩邊必須從**同一組數值**出發。最乾淨的做法是拿一份 AWML 訓練好的
checkpoint，轉成 autoware-ml 格式：

```bash
docker exec autoware-ml-yihsiang bash -lc "cd /workspace && \
  python -m autoware_ml.tools.convert_streampetr_checkpoint \
    --input  pretrained/awml_t4_best_epoch10.pth \
    --output pretrained/awml_t4_best_epoch10_converted.pth \
    --bgr-to-rgb"
```

用「已訓練完成」的權重而不是隨機初始化，可以避免兩邊 RNG 不同造成的
初始化差異污染比較。

### Step 2：跑 autoware-ml 側

```bash
docker exec autoware-ml-yihsiang bash -lc "cd /workspace && \
  python -m autoware_ml.tools.overfit_probe \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu \
    --weights pretrained/awml_t4_best_epoch10_converted.pth \
    --steps 200 --batch-size 2 --lr 1e-4 \
    --output parity_out/aml_overfit.jsonl \
    datamodule.data_root=/workspace/data/t4datasets \
    datamodule.train_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_train.pkl \
    datamodule.val_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_val.pkl \
    datamodule.test_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_test.pkl"
```

### Step 3：跑 AWML 側

```bash
docker exec awml_petr bash -lc "cd /workspace && \
  python tools/detection3d/overfit_probe.py \
    projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore.py \
    --weights work_dirs/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore/epoch_10.pth \
    --steps 200 --batch-size 2 --lr 1e-4 \
    --output parity_out/awml_overfit.jsonl"
```

> **`--lr` 必須兩邊一樣。** AWML 的 config 寫 5e-5，靠 `auto_scale_lr`
> 在 16 batch 時 ×2 才變成實際的 1e-4；本工具不做任何自動縮放，所以
> **兩邊都明確傳 `--lr 1e-4`**。沒傳的話 AWML 側會印警告。

### Step 4：比對（在 host 上跑）

```bash
cd /home/yihsiang/autoware-ml
python3 -m autoware_ml.tools.compare_overfit \
  parity_out/aml_overfit.jsonl \
  /home/yihsiang/AWML/parity_out/awml_overfit.jsonl
```

輸出三段：**① 輸入檢查**（fingerprint 是否一致）、**② 軌跡對照**（含判決）、
**③ 逐項 loss 分解**（step 0 與最後一步）。

---

## 3. 先看 fingerprint —— 這一步不能跳

比較 loss 之前必須確認兩邊真的**看到同一個 batch**。fingerprint 的每一欄
都是刻意設計成**跨框架可比**的（兩邊的張量巢狀結構、通道順序、時間戳
基準、矩陣排列慣例都不同）：

| 欄位 | 用途 | 為什麼這樣設計 |
| --- | --- | --- |
| `timestamp_deltas` | **同一批 frame 嗎** | AWML 存序列相對時間、autoware-ml 存絕對 epoch 秒 —— 只有 batch 內的差值可比 |
| `gt_counts` / `gt_box_mean` | 同樣的 GT 嗎（GT 過濾差異會現形） | 自動展平巢狀結構與 mmdet3d box 物件 |
| `img.mean` / `std` | 影像張量統計 | **刻意忽略通道順序**（autoware-ml 餵 RGB、AWML 餵 BGR，數學等價），也忽略 queue 維度 |
| `projected_pixels` | 相機幾何 | 把固定的 lidar 座標點投影過每台相機，比較**像素座標**而不是矩陣數值 —— 不受矩陣排列慣例、相機順序影響 |

判讀：

- `timestamp_deltas` / `gt_counts` 不同 → **拿到不同的 frame**（dataset 索引或排序不同）
- `projected_pixels` 不同 → 相機幾何不同（crop / resize / 內參更新不一致）
- 只有 `img.mean/std` 有微小差異（容忍 5e-3）→ 幾何相同、只是 resize 插值核不同
  （AWML 用 PIL BICUBIC、autoware-ml 用 cv2 INTER_LINEAR），屬預期

---

## 4. 二分搜尋：逐個打開被關掉的元件

預設關閉的東西正是「可能造成差異」的候選。若 step-0 與軌跡都吻合，就逐個
打開重跑，看哪個一打開就讓兩邊分岔：

| 旗標 | 打開什麼 | 為什麼值得測 |
| --- | --- | --- |
| `--keep-augmentation` | 幾何增強（旋轉/縮放/flip） | 增強的取樣粒度與實作差異 |
| `--keep-grid-mask` | GridMask | 隨機遮罩實作 |
| `--keep-dn` | denoising queries | DN 的加噪與目標建構 |
| `--keep-memory` | 時序 memory 跨步累積 | 時序路徑（注意：會讓每步不獨立） |
| `--precision bf16` / `fp16` | 混合精度 | 數值精度 |

打開隨機性元件後兩邊的 RNG 流不同，**單次比較會有雜訊**；這時要看的是
「軌跡的整體斜率/水準是否系統性偏移」，而不是逐步數字。

---

## 5. 其他參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--steps` | 200 | 過擬合步數。200 步足以看出軌跡差異；500 步更明顯但慢 |
| `--batch-size` | 2 | 固定 batch 的樣本數。太大沒必要，2 就能暴露差異 |
| `--start-index` | 0 | 固定 batch 取 dataset 的哪幾個索引。換幾組確認結論不是特定 frame 造成 |
| `--seed` | 0 | 只影響還開著的隨機元件 |
| `--term`（compare） | `loss` | 軌跡表要看哪一項 loss（例如 `loss_cls`、`loss_bbox`） |

---

## 6. 已知限制

- **預設關掉 DN 與時序 memory**，所以這兩條路徑的差異不會被涵蓋（要測就
  加 `--keep-dn` / `--keep-memory`）。
- 兩邊的優化器參數分組各自由自己的框架建構（autoware-ml 的
  `build_optimizer_groups`、AWML 的 mmengine `paramwise_cfg`）—— 這是刻意的：
  如果分組本身有差異，這個工具應該要能顯示出來。
- 不做 LR schedule（常數 LR），所以 warmup/cosine 的差異不在範圍內。
- 單 GPU、單 batch，所以跨 GPU 的 loss 正規化差異（例如 AWML 的
  `reduce_mean` avg_factor）在這裡看不出來。

---

## 7. 背景

這個工具是 StreamPETR parity 調查（見
[streampetr-awml-parity.md](streampetr-awml-parity.md)）的產物：當時所有
具名假設（評估器、partial-ignore、DN z 目標、fp16/bf16、增強粒度、
velocity-NaN GT）都被逐一排除或 ablate 證明無效，但 autoware-ml 仍穩定領先
~3 mAP，需要一個比「再花 13 小時重訓」便宜得多的定位工具。
