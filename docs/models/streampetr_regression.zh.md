---
icon: lucide/target
---

# StreamPETR AWML 配方對齊（「regression」）

> Commit `98eec9e31f6b47898bccf96c0a155ce00201d86d` — *feat: streampetr regression*
> （vividf，2026-07-23）· 27 個檔案，+2758 / −24

## 這個 commit 在做什麼

autoware-ml 原生的 StreamPETR（`StreamPETRDetectionModel`）本來就能跑，但它訓練的
**配方比較精簡**，不如舊 **AWML**（`mmdetection3d` 版）repo 裡的量產模型完整。
因此在 autoware-ml 裡重新訓練，**無法重現** AWML 的準確度數字。

這個 commit 就是要補上這個差距。這裡的「regression」指的是*重現* AWML 的結果 ——
把原生訓練配方拉回 AWML baseline，讓在 autoware-ml 重訓練的模型能對上
AWML 的 `projects/StreamPETR/configs/default/vov_flash_480x640_baseline.py`。

為了達成這點，它把原生模型缺的四塊訓練配方移植過來、加上遷移／驗證 AWML checkpoint
的工具，並且**完全不動推論與部署路徑**。

**推論與 ONNX/TensorRT 部署皆未改動。** 這裡新增的所有東西都只在 `self.training`
之下執行，因此匯出的計算圖與 runtime 成本與先前完全相同。

---

## 五大改動（依關注點分組）

### 1. 輔助 2D 偵測頭（Focal-PETR 風格）— 僅訓練用

AWML 會用一顆額外的 2D head 來「塑形」影像特徵，推論時丟棄。現在原生移植：

| 檔案 | 職責 |
| --- | --- |
| [autoware_ml/models/detection3d/heads/focal2d.py](../../autoware_ml/models/detection3d/heads/focal2d.py) | `FocalHead2D`：在每個相機的 neck 特徵圖上預測 per-token 類別 / centerness / LTRB box / 投影後的 3D 中心。共五個 loss；**推論絕不執行**。 |
| [autoware_ml/losses/detection2d/losses.py](../../autoware_ml/losses/detection2d/losses.py) | `QualityFocalLoss`、`GIoULoss`、`WeightedL1Loss`、`HeatmapGaussianFocalLoss`。 |
| [autoware_ml/models/detection3d/task_modules/assigners2d.py](../../autoware_ml/models/detection3d/task_modules/assigners2d.py) | `HungarianAssigner2D` + `BBoxL1Cost2D` / `IoUCost2D` / `Center2DL1Cost`。 |
| [autoware_ml/models/detection3d/task_modules/boxes2d.py](../../autoware_ml/models/detection3d/task_modules/boxes2d.py) | 2D box 工具（`cxcywh↔xyxy`、IoU/GIoU）。 |
| [autoware_ml/transforms/camera/annotations2d.py](../../autoware_ml/transforms/camera/annotations2d.py) | `LoadAnnotations2DFromBoxes3D`：把（已增強的）3D GT box 投影到每個相機，產生 2D box / 中心 / 深度 / 標籤。 |

[streampetr.py](../../autoware_ml/models/detection3d/streampetr.py) 中，模型多了一個 optional 的
`img_roi_head`；只有當它存在**且**在訓練時，其輸出與 loss 才會被加進總 loss。
否則行為完全不變。

### 2. `CPFPN` neck — 與 AWML checkpoint 權重相容

[autoware_ml/models/common/necks/cp_fpn.py](../../autoware_ml/models/common/necks/cp_fpn.py) 是參考版 StreamPETR
`CPFPN` 的原生移植（單純 1×1 lateral、最近鄰 top-down、只在最高解析度那層做一次 3×3
refine）。它對齊 `mm` 的 `ConvModule` 參數命名，因此 AWML checkpoint 權重可以
**逐一對名載入**。既有的 `GeneralizedLSSFPN`（concat + BN + ReLU）*並不*相容，這也是
為什麼對齊與 checkpoint 轉換都需要一顆專用 neck。

### 3. `traffic_cone` / `barrier` 的 partial-ignore

某些 T4Dataset 場景標註了所有類別，**唯獨少了** `traffic_cone` 與 `barrier`。
在這些 frame 上訓練時，不能把這兩個未標註類別的背景預測當成 false positive 懲罰。

| 檔案 | 職責 |
| --- | --- |
| [autoware_ml/models/detection3d/partial_ignore.py](../../autoware_ml/models/detection3d/partial_ignore.py) | 新模組：`resolve_partial_ignore_labels`（類別名 → index）與 `normalize_status_flags`（tensor/list/scalar → 每 sample 一個 bool）。 |
| [autoware_ml/losses/detection3d/focal.py](../../autoware_ml/losses/detection3d/focal.py) | `SigmoidFocalLoss` 現在支援 **per-query-per-class** 權重 `(N, C)`，不再只有 per-query `(N,)`，因此可以遮蔽個別類別欄位。 |
| [heads/streampetr.py](../../autoware_ml/models/detection3d/heads/streampetr.py) | 只在**負樣本（背景）query** 上把被忽略的類別欄位歸零 —— 被匹配到的 query 仍保有完整監督。主匹配 loss 與被 noise 進背景的 denoising（DN）query 都會處理。 |
| [datamodule/common/multiview_detection3d.py](../../autoware_ml/datamodule/common/multiview_detection3d.py) | 輸出每 frame 的 `traffic_cone_barrier_status` flag（缺少時視為 `True`）。 |

這個 flag 沿著 datamodule → model → head 流動。當它不存在或全為 `True` 時，loss 計算
與先前完全一致（對完整標註資料沒有行為改變）。

### 4. iteration-warmup + epoch-cosine 學習率排程

[autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py](../../autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py) 新增
`IterWarmupEpochCosineLR`（先以「iteration」線性 warmup N 步，再以「epoch」做 cosine
衰減）以對齊 AWML。模型現在會把 `scheduler_config`（例如 `interval: step`）轉交給
Lightning，讓逐步 warmup 能正確 tick。

### 5. 資料載入對齊細節

[autoware_ml/transforms/camera/loading.py](../../autoware_ml/transforms/camera/loading.py) — `LoadMultiViewImagesFromFiles` 新增：

- `shuffle_order`：每個 sample 隨機打亂相機順序（AWML 的 `shuffle_cameras=True`
  訓練時正則化）；每個輸出的 per-camera 陣列都遵循打亂後的順序，因此 sample 內部保持一致。
- `color_type`（`rgb`/`bgr`）：明確的通道順序，需與正規化統計量一致。

Config 也把 `normalize_to_unit: false`，讓像素保持在 `[0, 255]`
（因為 `img_norm_cfg` 的 mean/std 是 0–255 尺度的 ImageNet 統計量）。

---

## 遷移與驗證工具

| 檔案 | 職責 |
| --- | --- |
| [autoware_ml/tools/convert_streampetr_checkpoint.py](../../autoware_ml/tools/convert_streampetr_checkpoint.py) | 把每個參數從 `mm` 版 AWML layout（`Petr3D`/`StreamPETRHead`/`VoVNet`/`CPFPN`）改名成原生模組名稱，可選擇為 BGR→RGB 翻轉 stem conv，並可用 `--drop-pattern` 丟掉舊的類別頭（讓 10 類的 nuScenes checkpoint 餵給 7 類的 T4 模型，等同 `strict=False`）。輸出 `{"state_dict": ...}` 供 `autoware-ml train --weights` 使用。 |
| [autoware_ml/tools/streampetr_parity_check.py](../../autoware_ml/tools/streampetr_parity_check.py) | 拿一幀真實的 AWML dump（fp32、eval、無 DN/GridMask/dropout）重放進原生模型，比對影像特徵、位置編碼、每層 head 輸出，以及每一個 loss 項。 |
| `work_dirs/parity/streampetr_parity_reference.pt` | parity checker 所消費的 31 MB 參考 dump（二進位）。 |

---

## Configs

[autoware_ml/configs/tasks/detection3d/streampetr/](../../autoware_ml/configs/tasks/detection3d/streampetr/) 下新增／變更：

| Config | 用途 |
| --- | --- |
| `_awml_parity.yaml` | 共用的 AWML 配方參數（可組合，**不可單獨執行**）：pc_range ±51.2、完整訓練增強、2D FocalHead、partial-ignore、iter-warmup/epoch-cosine 排程、seed 0，以及以 mAP 為準的 checkpoint 選擇。 |
| `_reset_scheduler.yaml` | 設 `model.scheduler: null`，讓後面的 config 可以整個替換 scheduler 節點（OmegaConf 對 dict 是逐 key 合併；合併到 `null` 上則會整個取代）。 |
| `vov_480x640_t4dataset_j6gen2_base.yaml` | 配方對齊的**基礎**訓練（35 epoch、bs 4、lr 5e-5）。 |
| `vov_480x640_t4dataset_j6gen2_finetune_cone_barrier.yaml` | 配方對齊的 cone/barrier partial-ignore **微調**（40 epoch、bs 1、lr 6.25e-6）。 |
| `base.yaml` | 新增 collation key：`traffic_cone_barrier_status`、`gt_bboxes_2d`、`gt_labels_2d`、`centers_2d`、`depths_2d`。 |
| `vov_320x800_nuscenes.yaml`、`vov_480x640_t4dataset_j6gen2.yaml` | 載入器微調（`normalize_to_unit: false`）；j6gen2 的 test set 現在指向 `..._test.pkl`（原本重用 val）。 |

> 注意：autoware-ml 沒有 `auto_scale_lr`。config 內的 LR 已針對指定 batch size 預先縮放 ——
> 其他設定請自行以 `total_batch_size / 8` 重新縮放。

---

## 如何使用

```bash
# 1. 把 AWML checkpoint 轉成原生 layout
python -m autoware_ml.tools.convert_streampetr_checkpoint \
    --input  work_dirs/streampetr_2_7/epoch_20.pth \
    --output streampetr_2_7_epoch_20_converted.pth \
    --bgr-to-rgb

# 2.（選擇性）對照 AWML 參考 dump 驗證 forward/loss 對齊
python -m autoware_ml.tools.streampetr_parity_check \
    --reference  work_dirs/parity/streampetr_parity_reference.pt \
    --checkpoint streampetr_2_7_epoch_20_converted.pth

# 3a. 基礎訓練
autoware-ml train \
    --config-name tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base \
    --weights nuscenes_vov99_baseline_320x800_converted.pth

# 3b. 帶 cone/barrier partial-ignore 的微調
autoware-ml train \
    --config-name tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier \
    --weights streampetr_2_7_epoch_20_converted.pth
```

---

## 測試與建置

- [autoware_ml/tests/models/test_streampetr_partial_ignore.py](../../autoware_ml/tests/models/test_streampetr_partial_ignore.py) — 13 個單元
  測試，涵蓋 partial-ignore 標籤解析、class-wise focal 遮蔽（主 loss + DN query）、
  2D head forward/loss、2D 投影、CPFPN 形狀、LR 排程、checkpoint 轉換（改名 + drop
  pattern），以及相機 shuffle 的一致性。
- [docker/Dockerfile](../../docker/Dockerfile) — 與 StreamPETR 無關的建置穩定性修正：`pixi` 內嵌的
  `uv` 有固定 30 秒 HTTP timeout，因此大的 PyPI wheel 在壅塞的連線上會逾時。
  `pixi install` 步驟現在把下載並發數限制為 4，並最多重試 5 次（`uv` cache 讓每次重試都有進展）。

---

## 哪些**沒有**改動

- 推論的 forward 路徑、ONNX 匯出、TensorRT engine 建置 —— 2D head 與 partial-ignore
  邏輯都受 `self.training` 把關。
- 完整標註的資料集：沒有 `traffic_cone_barrier_status`（或全為 `True`）時，loss 與此
  commit 之前 bit-for-bit 完全相同。
