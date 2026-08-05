# StreamPETR：autoware-ml vs AWML 訓練結果差距分析報告

> 情境：同一組資料（T4dataset J6Gen2 base、kokseang_2_8 split）、同一份 nuScenes
> pretrained、同為 2 GPU × batch 8 × 10 epochs 的 StreamPETR 訓練，
> **autoware-ml 的 val mAP 落後 AWML 約 5 個點**（epoch 8 同點比較：0.357 vs
> 0.408；AWML 最終到 0.433，autoware-ml 在 epoch 8 之後不升反降）。本文記錄：
> 兩邊實際生效的設定逐項比對結果、已排除的嫌疑、確認存在的差異、調查過程中
> 發現的 bug，以及後續的驗證計畫。
：同一組資料（T4dataset J6Gen2 base、kokseang_2_8 split）、同一份 nuScenes pretrained、同為 2 GPU × batch 8 × 10 epochs 的 StreamPETR 訓練， autoware-ml 的 val mAP 落後 AWML 約 5 個點（epoch 8 同點比較：0.357 vs 0.408；AWML 最終到 0.433，autoware-ml 在 epoch 8 之後不升反降）。本文記錄： 兩邊實際生效的設定逐項比對結果、已排除的嫌疑、確認存在的差異、調查過程中 發現的 bug，以及後續的驗證計畫。
比對對象：

| | autoware-ml | AWML |
| --- | --- | --- |
| Config | `detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu` | `t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore` |
| Run | MLflow run `bbd05f97…7d94`（2026-07-24） | `work_dirs/.../20260723_013849`（2026-07-23） |
| 數據來源 | `mlruns/mlflow.db` + `hydra/train.log` + `artifacts/config/resolved.yaml` | mmengine text log + `vis_data/config.py`（resolved dump） |

> 比對原則：一律使用**實際生效**的 resolved config（hydra 存檔 / mmengine
> dump），不看 source config；行為層面的問題（sampler、GT 過濾、metric）
> 直接讀兩邊的程式碼實作確認。

---

## 1. 結果對比

### 1.1 Validation mAP（epoch 對齊，AWML/mmengine 為 1-indexed）

| epoch | autoware-ml | AWML | 差 |
| ---: | ---: | ---: | ---: |
| 1 | 0.171 | — | |
| 2 | 0.225 | 0.251 | −0.026 |
| 6 | 0.331 | 0.374 | −0.043 |
| 8 | **0.357** | 0.408 | **−0.051** |
| 10 | 0.371（fresh test，見 §5.2） | **0.433** | |

- 訓練中 val 顯示 best 是 **epoch 8 的 0.357**、epoch 9/10「退步」
  （ModelCheckpoint：`was not in top 1`）—— 但 §5.2 的 fresh test 證實
  ep10 實為 **0.371**，「退步」是訓練中 val 的假象。
- AWML 單調上升到 epoch 10 的 0.433（NDS 0.491）。

### 1.2 Per-class mAP @ epoch 8（同一時間點）

| class | autoware-ml | AWML | 差 |
| --- | ---: | ---: | ---: |
| car | 0.501 | 0.601 | −0.100 |
| truck | 0.374 | 0.432 | −0.058 |
| bus | 0.568 | 0.603 | −0.035 |
| bicycle | 0.339 | 0.427 | −0.088 |
| pedestrian | 0.359 | 0.401 | −0.042 |
| traffic_cone | 0.183 | 0.208 | −0.025 |
| barrier | 0.178 | 0.185 | −0.007 |

**七個類別全面落後**，不是單一 head 或單一類別的問題 —— 這指向整體訓練
效率/數值層面的差異，而非某個模組壞掉。

> **更新（§5.1 實測後）**：上表兩欄用的是**不同的 evaluator**，「全面落後」
> 是量尺差異造成的假象。換成同一把尺（autoware-ml evaluator）後，真實訓練
> 差距只剩 ~1.7 點，且集中在 car；bus/pedestrian/traffic_cone/barrier 打平。
> 詳見 §5.1 執行結果。

### 1.3 Training loss（最後一層 decoder 的主損失，兩邊定義相同、可直接比較）

| | autoware-ml（ep10） | AWML（ep10） |
| --- | ---: | ---: |
| train loss_cls | 0.448 | 0.367 |
| train loss_bbox | 0.844 | 0.731 |

loss 曲線形狀一致，但 autoware-ml **從第 1 個 epoch 起就整體偏高**
（ep1 總 loss 24.9 vs 22.5），且落後幅度全程維持 —— 不是後期才發散。

---

## 2. 已驗證為相同的部分（排除的嫌疑）

| 項目 | 結論 | 驗證方式 |
| --- | --- | --- |
| 資料集 / split | 同一組 `kokseang_2_8` info pkl（train/val/test） | resolved config 比對 |
| 類別與標註 | 7 類、相同 name_mapping、truck+trailer 合併、相同 attribute 過濾、partial-ignore（traffic_cone/barrier） | resolved config 比對 |
| GT 過濾 | 實質上都是「≥1 個 lidar 點」：autoware-ml 走 `use_valid_flag`（valid flag 由 `num_lidar_pts > 0` 產生），AWML 走 `_filter_with_mask` 的 `num_lidar_pts > 0`。**注意** `train_min_points_near: 2` 只有 lidar 模型（ptv3/transfusion/centerpoint）在用，StreamPETR pipeline 沒掛 `ObjectMinPointsFilter` | 讀兩邊實作 |
| 模型超參數 | VoVNet-99 eSE（stage4/5）+ CPFPN + 644 queries、memory 1024、propagated 256、6 層 decoder、DN（scalar 10 / noise 1.0 / split 0.75）、GridMask、2D 輔助頭權重 2/5/2/10/1 | resolved config 比對 |
| Loss 權重 | cls 2.0 / bbox 0.25 / code_weights `[2,2,1,1,1,1,1,1,1,1]` | resolved config 比對 |
| **Pretrained 載入** | 兩邊都完整載入 backbone/neck/decoder，都隨機初始化 7 類 cls 分支。autoware-ml 載入 880/1526 tensors；「646 keys 未初始化」是**誤報**（見 §4.2） | `train.log` 載入報告 + 讀 `streampetr.py` |
| 色彩通道 | AWML 餵 BGR + BGR 統計；autoware-ml 餵 RGB（cv2 + `COLOR_BGR2RGB`）+ RGB 統計 + convert 工具翻轉 stem conv —— **數學等價** | 讀 `transforms/camera/loading.py` + convert 工具 |
| Optimizer / LR | 都是 AdamW wd 0.01、總 batch 16、warmup 500 iters → **1e-4**（AWML 寫 5e-5 但 `auto_scale_lr` 依 batch 16/8 自動 ×2，log 可證）→ per-epoch cosine、backbone ×0.1、grad clip 1.0 | AWML log `Scaling the original LR by 2.0` + MLflow `lr-AdamW/pg1` 曲線 |
| Sampler / epoch 大小 | 兩邊都是每 epoch `randperm(seed+epoch)` 重排場景、round-robin 進 8 lanes × 2 ranks、trim 到最短 lane、不切割場景，~3300 steps/epoch（±1% 隨機浮動） | 讀兩邊 `GroupStreamingSampler` 實作 |
| 相機順序 shuffle | 兩邊訓練時都有（AWML `shuffle_cameras=True` 是預設） | 讀兩邊實作 |
| 增強 | resize ±2%、rand_flip、GlobalRotScaleTrans ±0.3925 rad / 0.95–1.05、pc_range ±51.2、Pad 32 | resolved config 比對 |

## 3. 確認存在的差異（依嫌疑度排序）

| # | 項目 | AWML | autoware-ml | 影響評估 |
| --- | --- | --- | --- | --- |
| 1 | 混合精度 | fp16 AMP + dynamic loss scale（`NoCacheAmpOptimWrapper`） | `bf16-mixed` | 真實差異。bf16 尾數比 fp16 少 3 bits；最容易 A/B 驗證的候選 |
| 2 | 程式碼血統 | mmdet3d 原版（StreamPETR 官方實作） | 原生重寫（head、2D 頭、position encoding、幾何增強走不同數學路徑，如 `reverse_angle` vs 相機矩陣逆變換） | 無法從 config 排除；數值級細微差異會累積成收斂差 |
| 3 | 後期行為 | mAP 單調升到 ep10 | 訓練中 val 顯示 ep8 見頂後退步 —— **§5.2 實測推翻：fresh test ep10 = 0.371 > ep8 0.356，其實有續升** | 已解決；真因是那些 epoch 的 validation 被靜默跳過（§4.1），已修（§7） |
| 4 | Metric 實作 | T4Metric：預測與 GT 都做範圍過濾（**base_link** 原點）、原始 precision 積分、有 bike-rack 過濾 | MeanAP：**不過濾出界預測**（多吃 FP → 偏低）、有 precision envelope（→ 偏高）、**LiDAR** 原點 | **§5.1 實測：同權重下 autoware-ml evaluator 低 3.4–3.8 點 —— 是表面差距的主因**（靜態分析原估 ≲1 點，低估了出界 FP 的影響） |
| 5 | 亂數 | seed 0 + `deterministic=True` | seed 0（排列序列不同） | 單次 run 雜訊 ~±0.5 點 |

> Metric 補充：若某類別在 val 完全沒有 GT，T4Metric 會把該類算成 AP=0 平均
> 進 mAP，autoware-ml 的 val mAP 則直接跳過該類 —— 該情境下兩邊會**系統性**
> 分歧（T4Metric 偏低）。本次比較不受影響，但跨資料集比較時要注意。

## 4. 調查中發現的 bug

### 4.1 Validation 被靜默跳過（初判為 MLflow logging bug，後續深挖後改判）

表象：`mlflow.db` 裡 `val/*` 只有 4 個 epoch（1/2/6/8）有數據，而 ModelCheckpoint
每個 epoch 都有 `reached` / `was not in top 1` 紀錄 —— 一開始以為是「只記進步
epoch」的 logging bug。

**真正的根因（修復時確認）**：Lightning 在第一次 dataloader setup 時把
`val_check_batch = int(max_batches × val_check_interval)` 用**當時的 epoch 長度
（3351）凍結**，之後只在 `(batch_idx+1) % 3351 == 0` 時觸發 validation。但
`GroupStreamingSampler` 每個 epoch 的長度隨場景排列而變（3150–3351），**只有
長度剛好是 3351 的 epoch（1/2/6/8，佔 4/10）有跑 validation，其餘 6 個 epoch
的 validation 被靜默跳過**。ModelCheckpoint 讀到的是 `callback_metrics` 裡
**殘留的舊值**（0.357 對 best 0.357 不算更好 → 印出 `was not in top 1`），
看起來像每個 epoch 都有評估。後果：

- val 曲線缺 6 個 epoch，且「ep8 之後退步」是殘留值造成的假象（§5.2 的
  fresh test 證實 ep10 實為 0.371）。
- best checkpoint 的選擇只在 4 個 epoch 之間比較，可能錯過真正的最佳權重。

### 4.2 權重載入報告的「未初始化」是別名重複計數的誤報

`StreamPETRDetectionModel` 把同一個 backbone/neck 物件同時註冊在兩個屬性下
（`self.img_backbone` 與 `self.image_feature_extractor.img_backbone`，
`autoware_ml/models/detection3d/streampetr.py:251-258`），state_dict 因此出現
兩套指向同一 tensor 的 key。checkpoint 從 `img_backbone.*` 載入後權重已生效，
但載入報告把 `image_feature_extractor.*` 那套別名（626+6 keys）連同刻意 drop
的 cls 層（14 keys）一起列成「Model keys not initialized from weights (646)」
—— 調查時一度誤判成 backbone 沒載入。建議 `utils/checkpoints.py` 報告前先
對 shared-tensor 別名去重。

## 5. 後續驗證計畫

依「一次隔離一個變因」排序：

### 5.1 權重交叉評估 —— 把 metric 差與訓練差一刀切開（最優先）

用同一份權重過兩個 evaluator，分數差就是**純 metric 差**；再回頭看訓練曲線
剩下的差距就是**純訓練差**：

```bash
# AWML 的 best（7 類，無 shape 衝突，不需要 drop pattern；
# AWML 是 BGR 訓練，轉到 autoware-ml 的 RGB pipeline 仍需 --bgr-to-rgb）
python -m autoware_ml.tools.convert_streampetr_checkpoint \
    --input  "/path/to/AWML/work_dirs/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore/best_NuScenes metric_T4Metric_mAP_epoch_10.pth" \
    --output pretrained/awml_t4_best_epoch10_converted.pth \
    --bgr-to-rgb

autoware-ml test \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu \
    datamodule.data_root=/workspace/data/t4datasets \
    datamodule.val_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_val.pkl \
    datamodule.test_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_val.pkl \
    --weights /workspace/pretrained/awml_t4_best_epoch10_converted.pth
```

判讀：AWML 權重在 autoware-ml evaluator 上若得 ~0.42–0.43，表示 metric
幾乎可比、差距全在訓練；若掉到 ~0.36，表示 metric/前處理才是主因，訓練
本身可能沒問題。

#### 執行結果（2026-07-27）

把 AWML 的 epoch 8 / epoch 10（best）checkpoint 都轉檔後在 autoware-ml
evaluator 上跑同一批 val frames：

| 權重 | T4Metric（AWML 自評） | autoware-ml evaluator | metric offset |
| --- | ---: | ---: | ---: |
| AWML epoch 8 | 0.408 | **0.374** | −0.034 |
| AWML epoch 10（best） | 0.433 | **0.395** | −0.038 |

**同 epoch、同尺的真實比較（autoware-ml evaluator、同一批 val frames）：**

| class @ep8 | autoware-ml 訓練 | AWML 訓練 | Δ |
| --- | ---: | ---: | ---: |
| car | 0.501 | 0.558 | **−0.057** |
| truck | 0.374 | 0.400 | −0.026 |
| bicycle | 0.339 | 0.367 | −0.028 |
| bus | 0.568 | 0.568 | ±0.000 |
| pedestrian | 0.359 | 0.360 | −0.002 |
| traffic_cone | 0.183 | 0.184 | −0.001 |
| barrier | 0.178 | 0.181 | −0.003 |
| **mAP** | **0.357** | **0.374** | **−0.017** |

結論：

1. **原本 5.1 點的差距，約 3.4 點是 metric 實作差異**（autoware-ml evaluator
   對同一份權重系統性低 3.4–3.8 點，方向與 §3 #4 的機制分析一致，但幅度
   比程式碼靜態分析預估的 ≲1 點大 —— 主要來自不過濾出界預測產生的 FP）。
2. **真實訓練差距只有 ~1.7 點，且高度集中在 car（−5.7）與 truck/bicycle**；
   bus / pedestrian / traffic_cone / barrier 已經打平 —— §1.2「七類全面落後」
   的印象是量尺差異造成的假象。
3. 跨框架比較 mAP 時**永遠要用同一個 evaluator**；兩邊 log 裡的 mAP 不可
   直接互比。

### 5.2 補齊 autoware-ml ep10 的數字

對 `best.ckpt`（ep8）與 `last.ckpt`（ep10）各跑一次 fresh test（同 evaluator、
同 val frames；指令同 §5.1，`--weights` 換成該 checkpoint），與 §5.1 的 AWML
轉檔權重形成四點全同條件的對照。

#### 執行結果（2026-07-27）

| epoch | autoware-ml | AWML | Δ |
| ---: | ---: | ---: | ---: |
| 8 | 0.356 | 0.374 | −0.018 |
| 10 | **0.371** | **0.395** | **−0.024** |

ep10 per-class（同尺）：car −0.054、bicycle −0.061、truck −0.023、
pedestrian −0.019、bus −0.008、traffic_cone ±0、barrier ±0。

新發現：

1. **「ep8 後退步」是訓練中 validation 的假象。** fresh test 顯示 autoware-ml
   ep8→ep10 實際上**繼續進步**（0.356 → 0.371）。修復 §4.1 時查明機制：
   ep9/10（以及 ep3/4/5/7）的 validation **根本沒有執行**，ModelCheckpoint
   比較的是殘留的 ep8 值（0.357 不優於 best 0.357 → `was not in top 1`）。
   **教訓：關鍵結論要用 fresh test 驗證，別只信訓練中的 val 曲線。**
2. 最終真實差距：**ep10 同尺 −2.4 點**，集中在 car（−5.4）與 bicycle（−6.1），
   其餘五類在 ±2.3 點內、bus/traffic_cone/barrier 完全打平。
3. 後續調查方向從「整體訓練效率」收斂為「**car 與 bicycle 的類別特定差距**」
   —— 建議優先看這兩類的預測分佈（距離、velocity、score 分佈）而不是先跑
   §5.3 的精度 A/B。

### 5.3 精度 A/B —— 隔離 bf16 vs fp16

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base_2gpu \
    trainer.precision=16-mixed \
    datamodule.data_root=/workspace/data/t4datasets \
    datamodule.train_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_train.pkl \
    datamodule.val_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_val.pkl \
    datamodule.test_ann_file=info/kokseang_2_8/t4dataset_j6gen2_base_infos_test.pkl \
    --weights /workspace/pretrained/nuscenes_vov99_baseline_320x800_converted.pth
```

看前 2–3 個 epoch 的 `train/loss_cls` 是否收斂到 AWML 的水準（ep1 ≈ 0.53、
ep2 ≈ 0.49）即可初步判斷，不必跑滿 10 epochs。

### 5.4 修 bug

1. **MLflow val logging**：讓每個 epoch 的 val metrics 都進 MLflow（§4.1）。
   修好後 val 曲線才完整，後續實驗才不會再瞎。
2. **載入報告去重**：`load_matching_weights` 對 shared-tensor 別名去重再報告
   （§4.2），避免下一個人再被 646 嚇到。

### 5.5 若 5.1–5.3 仍未定位（進階）

- 單 batch 過擬合對照：兩邊各拿同一個 batch 訓 500 iters，比 loss 下降軌跡，
  快速暴露 head/增強實作的數值差異。
- 逐模組數值對拍：同一輸入分別過兩邊的 position encoding / head forward，
  比對中間張量（backbone 已可用同一份權重）。

---

## 6. 根因分析：BEV 增強沒有折進 ego_pose，訓練時時序記憶被打亂（2026-07-27）

### 6.1 證據鏈（由 §5.2 的 ep10 test log 挖出）

| 觀察 | 數據 | 指向 |
| --- | --- | --- |
| AP 差距在 0.5/1/2/4m **所有閾值均勻存在** | car 每檔 −5~6 點、bicycle −6~7 點 | 不是定位精度問題 |
| 4m 總 match 數幾乎相同 | car 34355 vs 34510 | 偵測有出來，recall 上限一樣 |
| max-F1 操作點 recall 低、precision 持平 | car recall 0.632 vs 0.696；precision 0.765 vs 0.781 | TP 分數偏低、排序差 |
| **AVE（速度誤差）全類別 +43~66%** | car 0.276 vs 0.193、bus 0.267 vs 0.163、bicycle 0.859 vs 0.517 | 時序通道劣化 |
| AOE / ASE 幾乎持平 | — | 單幀幾何沒問題 |
| AWML 權重過 autoware-ml **推論管線** AVE 正常 | car 0.193 | 問題在**訓練**，不在推論 |

速度與跨幀物件關聯幾乎全靠 StreamPETR 的 memory/query propagation 學出來
—— 症狀集中指向「訓練時時序記憶通道壞掉，模型學會不信任 memory」。

### 6.2 根因（程式碼比對確認）

StreamPETR 每一步用 `ego_pose_inv(t) @ ego_pose(t-1)` 把上一幀的 memory
（top-256 propagated queries + 1024 memory keys 的 reference points、
egopose、velocity）warp 到當前幀座標。訓練管線兩邊都開了逐幀獨立抽樣的
BEV 增強（旋轉 ±0.3925 rad、縮放 0.95–1.05）：

- **AWML**：`GlobalRotScaleTransImage` 的 `_rotate_bev_along_z` /
  `_scale_xyz` / `_trans_xyz` 在改 `lidar2img`/`extrinsics` 的同時，把增強
  折進 `ego_pose ← ego_pose @ aug⁻¹`、`ego_pose_inv ← aug @ ego_pose_inv`
  （`projects/StreamPETR/stream_petr/datasets/pipelines/transform_3d.py:439-473`）
  —— 跨幀補償在增強後的座標系裡**仍然精確**。
- **autoware-ml**：`transforms/camera/geometry.py` 的 `GlobalRotScaleTrans`
  只轉了 `gt_boxes` 和相機矩陣，把增強矩陣存進 `global_aug_matrix` 後
  **全 repo 沒有任何地方消費它；`ego_pose`/`ego_pose_inv` 原封不動**
  （grep 驗證：`transforms/` 內對 ego_pose 零命中）。`RandomFlip3D` 的
  `bev_flip_matrix` 同樣無人消費（本 config 未開 BEV flip，實害來自
  rot/scale）。

後果：訓練時第 t 幀的世界被隨機轉了 θ_t（±22.5°）、縮放 s_t，但 memory
是用**原始 odometry pose** warp 的 —— 每一步 memory 都錯位「相鄰兩幀增強
的相對差」（Δθ 平均 ~15°、最大 45°；50 m 處 reference point 偏 ~13–20 m）。
單幀監督自洽（GT 跟著轉），只有時序通道整段是雜訊。推論時沒有增強、
對齊完美，但權重已經學會忽略 memory。這同時解釋：

- **(a) AVE 全類別劣化** —— 速度主要從時序通道學；
- **(b) car/bicycle recall 差** —— propagated queries 本該是高品質
  warm-start（對移動/小物件幫助最大），訓練時卻是空間雜訊；
- **(c) train loss 從 ep1 起整體偏高** —— 時序通道貢獻不了資訊。

次要（同類、程度小）：AWML `reset_origin=True` 把每個場景的 ego 平移
重定原點，autoware-ml 用原始地圖座標（~9e4 m）過 float32 ego_pose →
公分級對齊雜訊；autoware-ml 的 scale 增強沒有同步縮放 GT velocity
（AWML 的 `.scale()` 有）→ ±5% 速度目標不一致。

§5.2 的「訓練中 val 低報」與 memory 無關（兩邊都確認 train/val 切換會
重置 memory）—— 後續在修 §4.1 時查明真因：那些 epoch 的 validation 根本
沒有執行（變動長度 sampler × Lightning 凍結的 `val_check_batch`），詳見
§4.1。

### 6.3 修法與驗證

**修法**（對齊 AWML 的語義）：在 `transforms/camera/geometry.py` 的
`GlobalRotScaleTrans` 與 `RandomFlip3D`（以及 `camera_lidar/geometry.py`
的對應類別）中，把增強折進 ego pose：

```text
ego_pose     ← ego_pose @ aug⁻¹
ego_pose_inv ← aug @ ego_pose_inv
```

（或讓 datamodule 在 collation 前消費既有的 `global_aug_matrix` /
`bev_flip_matrix`。）順手可加：per-scene origin reset、scale 增強同步縮放
GT velocity。

**便宜的驗證**（不用跑滿 10 epochs）：修完後重訓 2–3 epochs，看

1. `train/loss` 是否收斂到 AWML 的軌跡（ep1 ≈ 22.5、ep2 ≈ 18.5）；
2. 對 ep2 checkpoint 跑 test，AVE_car 是否從 ~0.28 掉向 ~0.19。
兩者其一成立即可確認根因；滿訓後預期同尺 mAP 差距（ep10 −2.4 點）
大幅收斂。

---

## 7. 修復紀錄（2026-07-27，尚未重訓驗證）

三個問題都已修在 working tree（未 commit、未重訓 —— 驗證訓練待啟動）：

### 7.1 BEV 增強折進 ego_pose（§6 根因）

- `autoware_ml/transforms/geometry3d.py`：新增 `update_ego_poses()`
  （`ego_pose ← ego_pose @ aug⁻¹`、`ego_pose_inv ← aug @ ego_pose_inv`，
  無 ego_pose 的樣本自動跳過）；`transform_boxes()` 順手修正
  **GT velocity 未隨 scale 縮放**的次要問題（`v' = s·R·v`）。
- `transforms/camera/geometry.py` 與 `transforms/camera_lidar/geometry.py`
  的 `GlobalRotScaleTrans` / `RandomFlip3D` 共四處都接上 `update_ego_poses`。
- 測試：`tests/transforms/test_geometry3d.py` 新增「增強後 ego_pose 必須把
  *增強座標* 映回同一個 global 點」的性質測試（rot/scale/trans 與 flip 都測）、
  camera vs camera_lidar 一致性、velocity 縮放。

### 7.2 Validation 靜默跳過（§4.1）

- 新增 `autoware_ml/utils/lightning_loops.py`：`EpochEndValidationLoop`
  覆寫 `_should_check_val_fx` —— 除了原本的 modulo 條件外，**epoch 的真正
  最後一個 batch 也觸發 validation**（配合 `check_val_every_n_epoch`）。
- `utils/runtime.py` 的 `instantiate_trainer()` 統一掛上，所有訓練入口生效。
- 測試：`tests/utils/test_lightning_loops.py` 用「epoch 0 有 4 個 batch、
  之後每 epoch 3 個」的縮水 sampler 重現 bug —— 原生 Lightning 只在 epoch 0
  跑 val（`[0]`），掛上修正後每個 epoch 都跑（`[0,1,2]`）。若上游 Lightning
  修好，第一個測試會開始失敗，即可移除此 override。

### 7.3 權重載入報告別名誤報（§4.2）

- `autoware_ml/utils/checkpoints.py`：以 `(data_ptr, shape, dtype)` 偵測
  shared-tensor 別名；經別名載入的 key 改列為
  `Model keys initialized via shared-tensor aliases (N)`，不再混入
  「not initialized」清單；`enforce_full_coverage`（deploy 用）同步計入別名。
- 測試：`tests/utils/test_checkpoints.py` 新增同一 module 掛兩個屬性的模型
  的載入報告與 full-coverage 測試。

測試狀態：`tests/transforms/` + `tests/utils/test_checkpoints.py` +
`tests/utils/test_lightning_loops.py` 共 **157 passed**（容器內執行）。

### 7.4 待修（下一項）：dataloader worker thread pinning

**現象（2026-07-27 實測）**：`num_workers: 32`（2 GPU × 32 = 64 workers）時
共享機 load average 飆到 **~209**（約 48 核）、GPU 利用率掛零、~10 秒/batch
（正常 0.8 秒），且記憶體壓進 swap 39 GB。`SerializedSampleList`（commit
`742b64e`）已解決 copy-on-write 的 OOM（worker RSS 只剩 300–800 MB），但
**CPU 超載是另一道坎**：每個 worker 的 OpenCV/BLAS 還會各開自己的執行緒，
worker 數 × 執行緒數把 CPU 撐爆。

**AWML 為什麼 2×32 跑得動**：mmengine 的 `env_cfg` 做了 thread pinning ——
`opencv_num_threads=0`（`cv2.setNumThreads(0)`）+ `OMP_NUM_THREADS=1`，
每個 worker 被限制成單執行緒。autoware-ml 目前沒有等效機制。

**修法**：在 datamodule 的 dataloader 加 `worker_init_fn`（或程序啟動時的
環境設定）：`cv2.setNumThreads(0)`、`torch.set_num_threads(1)`、
`OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`。修好後才能把 `num_workers`
往上調（AWML parity 是 32），並以「load average ≈ worker 數、GPU 利用率
> 90%、it/s 不低於 2×8 的 1.22」驗收。在那之前 **`num_workers: 8` 是這台
機器的實測安全值**（1.22 it/s、~52 分/epoch）。

重訓驗證方式見 §6.3：跑 2–3 epochs 看 train loss 是否貼上 AWML 軌跡
（ep1 ≈ 22.5、ep2 ≈ 18.5）、ep2 checkpoint 的 AVE_car 是否從 ~0.28 掉向
~0.19；同時確認每個 epoch 都有 val 進 MLflow。

---

## 8. 修復驗證結果（2026-07-28）：全面反超 AWML

修復後重訓（run `4934d680…`，num_workers 8、~65 分/epoch），三個修復
全部驗證通過：

1. **Validation 每個 epoch 都執行**（10/10 epoch 的 val metrics 進 MLflow，
   train.log 每個 epoch 都有 `validation metrics:` 摘要行）。
2. **train loss 收斂到 AWML 水準**：ep1 23.5（修復前 24.9）→ 最終 **15.37**
   （修復前 17.17；AWML 15.07）；loss_cls 最終 0.379（修復前 0.448；AWML 0.367）。
3. **時序通道恢復**：AVE 全面回到 AWML 水準或更好（見下表）。

### 最終對照（ep10、fresh test、同 evaluator、同 val frames）

| class | 修復後 autoware-ml | AWML | 修復前 autoware-ml |
| --- | ---: | ---: | ---: |
| car | **0.571** | 0.560 | 0.506 |
| truck | **0.467** | 0.415 | 0.392 |
| bus | **0.631** | 0.600 | 0.592 |
| bicycle | **0.455** | 0.411 | 0.351 |
| pedestrian | **0.387** | 0.382 | 0.363 |
| traffic_cone | **0.227** | 0.194 | 0.194 |
| barrier | **0.224** | 0.200 | 0.200 |
| **mAP** | **0.423** | 0.395 | 0.371 |
| mAP（0–51.2m 過濾版） | **0.431** | 0.403 | 0.377 |

七類全部 ≥ AWML；整體 **+2.8 點**（修復前 −2.4）。0–51.2m 過濾版 0.431
與 AWML 自家 T4Metric 的 0.433 已在雜訊範圍內持平。

### AVE（速度誤差，@optimal 2m，越低越好）—— ego_pose 修復的直接證據

| class | 修復後 | AWML | 修復前 |
| --- | ---: | ---: | ---: |
| car | **0.186** | 0.193 | 0.276 |
| truck | **0.286** | 0.299 | 0.441 |
| bus | **0.157** | 0.163 | 0.267 |
| bicycle | **0.489** | 0.517 | 0.859 |
| pedestrian | 0.316 | **0.299** | 0.428 |

car 的 ATE 0.605 / AOE 0.102 也略優於 AWML（0.629 / 0.106）。

### Test split 驗證（held-out，非 checkpoint 選擇依據）

為排除「對 val 過擬合」的質疑，兩份 ep10 權重再對
`t4dataset_j6gen2_base_infos_test.pkl` 各跑一次（同 evaluator）：

| class | autoware-ml（修復後） | AWML | Δ |
| --- | ---: | ---: | ---: |
| car | **0.572** | 0.553 | +0.019 |
| truck | **0.465** | 0.446 | +0.019 |
| bus | **0.569** | 0.530 | +0.039 |
| bicycle | **0.365** | 0.358 | +0.008 |
| pedestrian | **0.382** | 0.371 | +0.011 |
| traffic_cone | **0.201** | 0.186 | +0.016 |
| barrier | **0.192** | 0.106 | +0.086 |
| **mAP** | **0.392** | 0.364 | **+0.028** |
| mAP（0–51.2m 過濾版） | **0.399** | 0.370 | +0.029 |

領先幅度與 val split 完全一致（+2.8 點）、七類再次全部領先、
AVE_car 打平（0.151 vs 0.152）—— 結論可泛化，不是 val 挑點的偏差。
（AWML 的 barrier 在 test 上退化到 0.106，值得 AWML 側留意。）

**結論：§6 的根因診斷正確，parity 達成並反超（val +2.8 / test +2.8）。**
剩餘的開放項目只有 §7.4 的 worker thread pinning（效能優化，不影響精度）。
若要對外正式宣稱優於 AWML，可再補：反向交叉評估（autoware-ml 權重過
T4Metric）與第二個 seed 的重複實驗。

---

## 9. 歸因分析（2026-07-28）：為什麼 autoware-ml 訓練得比 AWML 好

修復後 autoware-ml 領先 +2.8 mAP（val/test 一致）。逐一驗證可能解釋後的結論：

### 9.1 已排除的解釋

| 假設 | 驗證 | 結論 |
| --- | --- | --- |
| **交叉評估偏差**（AWML 權重過 autoware-ml 管線吃了 resize/解碼差異的虧） | AWML 在**自家容器、自家管線、自家 T4Metric** 跑 test split：mAP **0.402**；與交叉評估 0.364 的差恰好 = 已知量尺 offset（+0.038，val 上同為 +0.038） | **排除** —— +2.8 是真實訓練差距 |
| **partial-ignore 實作差異**（barrier/traffic_cone） | 逐行比對：3D cls 負樣本遮罩、DN 遮罩、2D 頭（zero-weight vs clamp -100，淨效果相同）、assigner、預設值全部等價 | **排除** |
| Seed 運氣 | 單 run 雜訊 ~±0.5 | 蓋不住 +2.8 |

### 9.2 證實的 AWML 側訓練品質問題（差距的來源，貢獻無法精確分解）

1. **DN 的 z 目標不一致（最大的已證實 code bug，估 ≤1 點）**：AWML 的 DN
   分支用 **bottom-center** z 加噪與回歸（`streampetr_head.py:554,564,1032`），
   匹配分支卻用 **gravity-center** z（`:1085-1088`），兩者以 ~10:1 的樣本比
   （scalar=10）監督**同一組共享 reg branches** —— z 回歸被持續的梯度衝突
   拉扯、帶著 ~h/2 的系統偏差，decode 補償只符合匹配分支慣例。因 evaluator
   用 BEV(xy) 距離匹配，傷害是間接的（共享分支學習品質 + memory 傳遞的
   reference point z 劣化）。autoware-ml 全程 gravity-center 一致。
   **→ 已修並重訓驗證（2026-07-29）**：AWML `prepare_for_dn` 改用
   `torch.cat((t.gravity_center, t.tensor[:, 3:]), dim=1)`（與匹配分支同一
   轉換式）。重訓後同尺實測：**val +0.1 點（0.3947→0.3959，雜訊級）、
   test +0.9 點（0.3640→0.3728，主要來自 barrier/bicycle/truck/bus）**；
   AWML 自評 T4Metric val 0.429（舊 0.433，seed 雜訊內）。DN loss 幾乎不變
   （z 只是 10 個回歸維度之一、weight 1.0），證實 BEV 匹配的 metric 把
   z 偏差遮蔽掉大半 —— 是真 bug，但對 mAP 的實際貢獻在估計上限的下緣。
2. **fp16 + dynamic loss scale 的不穩定**：AWML 的 loss 到處 `torch.nan_to_num`
   （`streampetr_head.py:971-972,1044-1045`），溢位跳步是系統性拖累；
   autoware-ml 用 bf16 無此問題。
3. **GT 衛生**：autoware-ml 載入時以 `box_is_physical` 丟棄退化標註
   （非有限值/非正尺寸/速度>150m/s，`transforms/boxes3d/annotations.py:39-60`）；
   AWML 只在 bbox loss 濾非有限值，**退化框仍進 Hungarian 匹配（吃掉 query
   當 cls 正樣本）與 DN 群組**，瘋狂速度值留在 loss 造成尖峰、又觸發 fp16
   scale 掉落，與 #2 惡性循環。

### 9.3 barrier 在 test 崩盤（0.106 / 自評 0.113，差距的 ~1/3）

AWML **自家評估同樣崩**（T4Metric test barrier 11.3 vs val 19.2）——
是模型本身的弱點，非評估管線造成。沒有 barrier 特定的 bug；最合理解讀：
barrier 是稀有、場景集中的類別（再被 partial-ignore 場景稀釋），test AP
由少數場景主導、方差大；AWML 較吵的優化（§9.2 三項疊加）訓出的稀有類
表徵較不穩健，在 test 場景上翻車而 val 恰好撐住。下嚴格結論需要第二個
AWML seed。

### 9.4 總結（2026-07-30 更新：DN 與 bf16 兩輪 ablation 已完成）

四方同尺對照（AWML 各版權重皆過 autoware-ml evaluator）：

| | autoware-ml | AWML bf16+zfix | AWML z-fix | AWML 原版 |
| --- | ---: | ---: | ---: | ---: |
| val | **0.423** | 0.391 | 0.396 | 0.395 |
| test | **0.392** | 0.358 | 0.373 | 0.364 |

判決更新：

1. **fp16 vs bf16：排除。** AWML 換 bf16（無 GradScaler，鏡像 autoware-ml
   的 bf16-mixed）沒有改善，反而略降 —— fp16 不穩定不是差距來源。
2. **DN z 不一致：真 bug，但貢獻 ≈ 0。** val +0.1；test 的 +0.9 落在
   run 間雜訊內（見下）。仍建議 commit（正確性問題）。
3. **三個 AWML run 給出變異數基準**：val 散佈僅 ±0.25（0.391–0.396）——
   autoware-ml 的 0.423 高出 +2.8，**遠超 run 間變異，領先是確定的**；
   test 散佈 ±0.75（0.358–0.373），test 上的小差異不可過度解讀。
4. **barrier 的 test 崩盤是系統性的**：三個 AWML run 的 test barrier =
   0.106 / 0.126 / 0.108（val 三次都正常 0.20–0.22；autoware-ml test
   0.192）—— 跨 run 復現的泛化弱點，不是單次翻車。
5. **剩餘 ~2.8 點未歸因**：具名候選只剩 GT 衛生（`box_is_physical`）；
   更可能是原生重寫在未識別處有實質優勢。autoware-ml 側無「作弊」因素。

成本評估：已投入兩輪 ablation 重訓（各 ~13 小時），邊際收益遞減。
工程結論（不輸且穩定領先）已足夠；剩餘原因的深入調查見 §9.5。

### 9.5 剩餘原因的深入調查（2026-07-30）：找到主因候選

對「尚未審計」的訓練路徑做了完整比對（loss 平均因子、weight decay 分組、
GridMask、resize/flip 移植、2D 輔助頭目標生成、位置編碼、query 初始化），
加上資料面實證。結果：**大多數等價，找到兩個真實差異 + 一個量化的資料差異**。

#### 已驗證等價（本輪新增，之後不用再查）

loss 正規化與平均因子（cls/bbox/DN/2D 各項公式一致，僅 AWML 的 bbox
avg_factor 有跨 GPU reduce_mean、aml 用 per-rank —— ≤0.2 點）、weight decay
分組（兩邊都對 norm/bias 施加 wd=0.01，無差異）、GridMask（超參數與演算法
完全相同，prob 0.7 無 ramp）、2D 輔助頭目標（同樣從增強後的 3D 框投影，
同角點/中心/深度數學）、LID 位置編碼與 depth bins（逐 bit 等價）、query
初始化與 propagated-query 分數來源。

#### 真實差異 #1（主因候選）：增強的取樣粒度 —— 每相機獨立 vs 每幀一次

- **autoware-ml**：`ResizeCropFlipRotImage` 對**每個相機獨立取樣** resize
  抖動、隨機 crop_x、以及 50% flip（`transforms/camera/resize.py:225-232`，
  flip 在 `:257`）—— 一幀裡五個視角可能混合翻轉/未翻轉、各自不同的抖動，
  每個視角的內參各自正確更新。
- **AWML**：每幀只取樣**一次**、套用到所有相機
  （`transform_3d.py:211`，逐視角迴圈之前）。

autoware-ml 的增強嚴格更豐富。在 10 epochs、中小型資料集的設定下，
估計貢獻 **1–2 mAP** —— 是剩餘差距最大的具名候選。

#### 真實差異 #2：AWML 的 flip 內參有 1 像素誤差

AWML 翻轉時內參平移用 `x' = fW − x`（`transform_3d.py:347-350`），但 PIL
實際像素翻轉是 `x' = fW − 1 − x`；autoware-ml 用像素正確的
`fW − 1 − x`（`resize.py:284-292`）。AWML 約一半的訓練幀帶著全相機系統性
1px 幾何誤差（僅訓練期；測試不翻轉）。估 **0.2–0.5 mAP**。

#### 資料差異（量化完成）：velocity-NaN 標註的處理

train set 有 **21,898 個 velocity 為 NaN 的標註**（0.7%；car 0.81%、
traffic_cone 0.91% —— 軌跡端點幀），box 幾何本身全部正常：

- autoware-ml：`box_is_physical` **整框丟棄**（分類與回歸都看不到）。
- AWML：保留為 **cls 正樣本但無 bbox 監督**（bbox loss 的 `isnotnan`
  跳過整列；匹配成本不含速度所以匹配仍正常）。

方向未證（丟棄=乾淨 vs 保留=多 0.7% 的分類召回訊號），是唯一可用
「一行 mask + 重訓」驗證的剩餘假設。

#### barrier test 崩盤的結構性成因（資料面）

| | val | test |
| --- | --- | --- |
| 有 barrier 的場景 | 10/137 | 22/228 |
| top-3 場景占 barrier 數 | 70% | 59% |
| 距離中位數 / 在 51.2m 內 | 42.3m / 58% | 44.9m / 56% |

barrier 稀有、場景高度集中、近半在評估範圍邊緣 —— AP 由少數遠距場景
決定。遠距小物件最依賴時序累積與強增強帶來的穩健性，與「AWML 三個 run
在 test barrier 全崩（0.106/0.126/0.108）、autoware-ml 穩住（0.192）」
的觀察自洽。

#### 增強假設的 ablation 結果（2026-07-30）：**推翻**

AWML 套用「每相機獨立取樣 + 1px flip 修正」後重訓（`..._bf16_exp2`，
bf16 + z-fix + 增強修正）：

| | val | test |
| --- | ---: | ---: |
| AWML exp2（含增強修正） | **0.3868** | **0.3504** |
| AWML bf16+zfix（對照組） | 0.3912 | 0.3580 |

**沒有幫助，反而略降**（AWML 自評同方向：0.4276 → 0.4217）。
「每相機獨立增強是主因」的假設推翻 —— autoware-ml 的優勢不是來自
增強多樣性。1px flip 修正雖屬正確性修復，效果也被雜訊蓋住。

#### velocity-NaN GT 處理：**無差異**（原以為是差異，實測推翻）

- autoware-ml：`sanitize_velocity` 把 NaN → 0 後才做 `box_is_physical`
  判斷（`transforms/boxes3d/loading.py:122-123`），所以框**保留**、
  velocity 記為 0。
- AWML：mmdet3d `NuScenesDataset.parse_ann_info` 做
  `nan_mask = np.isnan(gt_velocities[:,0]); gt_velocities[nan_mask] = [0,0]`，
  T4Dataset 經 `super()` 走同一路徑（`t4dataset.py:122`）。
- 資料驗證：21,898 筆 NaN velocity **全部 vx/vy 同時為 NaN**，故
  mmdet3d 只檢查 `[:,0]` 也能全數攔到，無漏網。

兩邊行為相同；AWML loss 裡的 `usable = torch.isfinite(...)` 對 velocity
而言是永不觸發的防禦碼。

#### 最終歸因（取代 §9.4 第 5 點）

**四個 AWML run 的同尺結果（全部過 autoware-ml evaluator）：**

| | val | test |
| --- | ---: | ---: |
| **autoware-ml（修復後）** | **0.4230** | **0.3922** |
| AWML zfix（fp16） | 0.3959 | 0.3728 |
| AWML 原版 | 0.3947 | 0.3640 |
| AWML bf16+zfix | 0.3912 | 0.3580 |
| AWML exp2（+增強修正） | 0.3868 | 0.3504 |
| AWML 四 run 平均 | 0.3921 | 0.3613 |
| **autoware-ml − AWML 平均** | **+0.0309** | **+0.0309** |

- **領先幅度在兩個 split 上完全一致（+3.1 點）**，且**高於四個 AWML run
  的最佳值**（val +2.7 / test +1.9）—— 遠超 run 間變異
  （AWML val 散佈 0.9 點、test 散佈 2.2 點）。
- 逐類貢獻（test）：barrier +1.14、bus +0.72、car +0.36、truck +0.33、
  traffic_cone +0.25、pedestrian +0.17、bicycle +0.12（總和 3.09）；
  val 則分散於 truck +0.72、bicycle +0.63、traffic_cone +0.55、bus +0.54。
  **優勢是全面的、非單一類別**，符合「系統性訓練品質差異」而非個別 bug。
- **所有具名假設已出清**：交叉評估偏差、partial-ignore、DN z、fp16/bf16、
  增強粒度、velocity-NaN GT，全部排除或實測無效；loss 平均因子、wd 分組、
  GridMask、2D 輔助頭、位置編碼、query 初始化皆逐項等價。
- 剩餘已知次要項（插值核 PIL BICUBIC vs cv2 bilinear、bbox avg_factor 的
  跨 GPU reduce_mean、autoware-ml 的自注意力 fp32 島）估計合計 0.5–0.8 點，
  **填不滿 3.1 點** —— 差距的主體仍未歸因。

**AWML 兩處修改（2026-07-30，已重訓驗證 → 見下方 ablation 結果）**：
`transform_3d.py` 的 `ResizeCropFlipRotImage.__call__` 改為**每相機獨立
取樣**（test 模式取樣是確定性的，行為不變；已驗證五視角 train 時內參
各異、test 時一致）；`_img_transform` 的 flip 平移修為 `crop_width − 1`
（單元驗證：翻轉後投影與實際像素對齊誤差 0.31px，修正前 ~1.3px）。

> **踩坑筆記：容器時鐘。** `awml_petr` 容器跑 UTC、host 跑 JST(+9)，
> 所以 mmengine 的 work_dir 名稱（例如 `20260729_172914`）是**容器時間**，
> 比 host 檔案 mtime 早 9 小時。判斷「某次 run 有沒有吃到某個程式碼修改」
> 時必須換算，否則會誤判（本調查一度誤判 exp2 沒吃到修改）。
> 換算檢查：`docker exec <container> date` vs host `date`。

---

## 附錄 A：在本機瀏覽器看 MLflow UI（SOP）

Metrics 都存在 sqlite（`mlruns/mlflow.db`），文字 log 裡沒有 —— 要看曲線
一律走 MLflow UI。host 上沒裝 mlflow，要在容器裡啟動（容器是 host network
模式，容器內開的 port 直接就是 server 的 port）。

### 1. 在 server 上啟動 MLflow UI

```bash
docker exec autoware-ml-yihsiang bash -lc \
    "cd /workspace && nohup mlflow ui \
        --backend-store-uri sqlite:///mlruns/mlflow.db \
        --host 127.0.0.1 --port 5001 \
        > /workspace/mlruns/mlflow_ui.log 2>&1 &"
```

- **用 5001，不要用 5000**：5000 被別人（kang）的 MLflow 佔用，指向他自己的
  `/home/kang/projects/AWML/mlruns` —— 連上會看到別人的實驗、找不到自己的 run。
  啟動前可先確認 port 沒被占用：`ss -tln | grep 5001`。
- 綁 `127.0.0.1` 是刻意的：只有透過 SSH/VSCode 轉發才連得到，別人碰不到。
- 確認活著：`curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5001`
  回 `200` 即可。log 在 `mlruns/mlflow_ui.log`。

### 2. 把 port 轉發到本機

程序是在容器背景啟動的，**VSCode 偵測不到、不會自動轉發**，要手動加（擇一）：

- **VSCode（推薦）**：下方面板的「PORTS」頁籤（和 TERMINAL 同一排；找不到就
  `Ctrl+Shift+P` → "Forward a Port"）→ 點「Forward a Port」→ 輸入 `5001`。
- **純 SSH**：在**本機** terminal 執行並保持連線：

  ```bash
  ssh -L 5001:localhost:5001 yihsiang@<server位址>
  ```

然後開本機瀏覽器 **<http://localhost:5001**。>

### 3. 常見狀況

| 症狀 | 原因 / 解法 |
| --- | --- |
| 本機瀏覽器 "site can't be reached" | port 沒轉發 —— 回到步驟 2 手動加 |
| 連上了但看不到自己的 run | 你連到 5000（別人的 MLflow）—— 改用 5001 |
| server 上 curl 5001 沒回應 | UI 程序死了（server 重開等）—— 重跑步驟 1 |
| `mlflow: command not found` | 在 host 上跑了 —— mlflow 只在容器裡，用步驟 1 的 `docker exec` |

也可以不開 UI 直接查數字：

```python
from mlflow.tracking import MlflowClient
client = MlflowClient("sqlite:///mlruns/mlflow.db")  # 容器內 /workspace 下執行
for m in client.get_metric_history("<run_id>", "val/det3d/mAP"):
    print(m.step, m.value)
```

---

## 附錄 B：資料位置

| 內容 | 路徑 |
| --- | --- |
| autoware-ml metrics | `mlruns/mlflow.db`（MLflow UI：container 內 port 5001） |
| autoware-ml resolved config | `mlruns/.../bbd05f97…/artifacts/config/resolved.yaml` |
| autoware-ml 訓練 log | `mlruns/.../bbd05f97…/hydra/train.log` |
| AWML resolved config | `work_dirs/.../20260723_013849/vis_data/config.py` |
| AWML 訓練 log | `work_dirs/.../20260723_013849/20260723_013849.log` |
| 互動式對比圖表 | claude.ai artifact `b2cb7e13-c47c-4fa3-975c-0cfab1ae6408` |
