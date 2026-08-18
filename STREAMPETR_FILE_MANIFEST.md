# StreamPETR Branch — Delta Manifest（review 用，未追蹤、不進 commit）

**Rebase 後的結構**（2026-08-06）：branch base = `89678cc`（Amadeusz 的 feat(models): add StreamPETR）。
staged 的 **40 檔（+3577 / −114）** 是在其上的純增量 —— 所有原始 StreamPETR 整合（vovnet、grid_mask、原版 head/model、samplers、multiview datamodule 骨架、resize.py 修正、docs/index.md）都已在 base commit 裡，不再出現在 diff 中。

History: `89678cc (add StreamPETR) → 1fb5f82 (sweep transformation fix) → b48ae8a (#77) → [staged delta]`

---

## 0. 訓練配方：原版 → 現在（一頁看懂改了什麼）

| 項目 | 原版（89678cc, Amadeusz） | 現在（staged delta） | 性質 |
|---|---|---|---|
| **Neck** | `GeneralizedLSSFPN`（concat+BN+ReLU） | **`CPFPN`**（reference 原版 neck，權重相容 AWML/model-zoo checkpoint） | 換模型元件 |
| **預訓練初始化** | 無流程（練不起來的主因之一） | model-zoo nuScenes pretrain + `convert_streampetr_checkpoint` 轉檔（880/880 tensor byte-equal） | 新增 |
| **LR scheduler** | `CyclicCosineAnnealingLR`（per-epoch，warmup 1 epoch） | **`IterWarmupEpochCosineLR`**（500-iter linear warmup 起點 1/3 + per-epoch cosine，`interval: step`；與 AWML 逐 epoch <0.5% 誤差） | 換配方 |
| **LR / batch** | lr 5e-5、batch 8、GPU 數不定 | lr 1e-4 / backbone 1e-5，**釘死 2 GPU × batch 8 = 總 batch 16**（AWML auto_scale_lr 的展開值） | 換配方 |
| **Epochs** | 35 | **10**（AWML 生產 recipe） | 換配方 |
| **增強** | 全關（rand_flip false、rot 0、scale 1.0、無 camera shuffle） | **全開**：rand_flip、rot ±0.3925 rad、scale 0.95–1.05、每 frame shuffle 相機順序 | 換配方 |
| **pc_range** | ±54.0 m | **±51.2 m**（AWML baseline） | 換配方 |
| **2D 輔助頭** | 無 | **`FocalHead2D`**（Focal-PETR 式 5 個 loss，只在訓練時跑） | 新增元件 |
| **Partial-ignore** | 無 | **traffic_cone/barrier**：未標註 scene 的 frame，背景 query 不因預測這兩類被罰（3D main/DN + 2D 都處理） | 新增元件 |
| **Loss 正規化** | rank-local positive count（隨 GPU 數改變結果，偏高 ~8%） | **cross-rank 全域 mean**（DETR/mmdet `reduce_mean` 語義；DN loss 同步修正） | bug fix |
| **像素正規化** | `normalize_to_unit` 預設 true → **[0,1] 像素配 0-255 ImageNet stats（bug）** | `normalize_to_unit: false`（nuScenes + j6gen2 全部 pipeline） | **bug fix** |
| **test split** | `test_ann_file` 指向 **val** pkl（bug） | 指向 test pkl | **bug fix** |
| **Checkpoint 選擇** | val/loss min + early stopping (patience 20) | **val/det3d/mAP max**、early stopping 關閉（訓滿固定 epochs） | 換配方 |
| **Seed** | 未固定 | 0 | 換配方 |
| **評估** | eval_class_range 54 m **徑向**上限（把方形 GT 的四角裁掉但預測留著→保證 FP）；0-50/0-54 buckets；`gt_num_points` 沒 collate → min-points filter 靜默失效 | eval_class_range 121 m（方形內 no-op）；T4MetricV2 buckets（0-50/50-90/90-121/0-121）；`gt_num_points` 進 collation | eval 對齊 |
| **DataLoader 記憶體** | annotation 是 plain list → 多 worker copy-on-write 逐漸私有化整份 annotations（OOM） | `SerializedSampleList` fork-shared buffer | bug fix |
| **Validation 觸發** | streaming sampler 每 epoch 長度不同 → 短 epoch **靜默跳過 validation**（mAP checkpoint 失效） | `EpochEndValidationLoop`：真正的最後一個 batch 也觸發 validation | bug fix |
| **增強幾何** | 增強沒折進 ego_pose → streaming memory 跨 frame 錯位；velocity 沒乘 scale | `update_ego_poses` 折疊 + `v' = s·R·v` | bug fix |

**成果**（詳見 docs/models/streampetr.md）：此配方訓出 val mAP **0.39127** / test **0.36609**（aligned eval），對比 AWML 同配方 bf16 retrain 的 0.37521 / 0.35515。

---

## 1. 逐檔說明

### Configs（4 檔）

| 檔案 | 變更內容 |
|---|---|
| `configs/.../streampetr/base.yaml` (M, +25) | seed 0；`gt_num_points` + `traffic_cone_barrier_status` 進 collation；scheduler 換成 IterWarmupEpochCosine（`interval: step`）；mAP checkpoint + early_stopping null |
| `configs/.../streampetr/vov_480x640_t4dataset_j6gen2.yaml` (M, +160/−30) | **上表所有「換配方」項的落地**：pc ±51.2、CPFPN、FocalHead2D、partial-ignore、全增強、lr 1e-4/2GPU/10ep、eval 121m+V2 buckets、train_collation_map（2D GT keys）、test pkl 修正、頂部完整寫 pretrain 下載+轉檔流程 |
| `configs/.../streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier.yaml` (A, 33) | 從 AWML checkpoint 續 finetune：只覆寫 batch 1 / lr 6.25e-6 / 40 ep / devices 1 |
| `configs/.../streampetr/vov_320x800_nuscenes.yaml` (M, +6) | 只有 `normalize_to_unit: false` bug fix ×3 pipeline（其餘保持 Amadeusz 原樣，仍是 GeneralizedLSSFPN 的非 parity 配置） |

### 模型（新增元件）

| 檔案 | 內容 |
|---|---|
| `models/common/necks/cp_fpn.py` (A, 108) | CPFPN 原生 port：1×1 lateral（無 norm/act）+ nearest top-down + 只在最高解析度 3×3 refine；參數名對齊 mm ConvModule → 權重相容 |
| `models/detection3d/heads/focal2d.py` (A, 455) | FocalHead2D：共享 conv tower ×2 + 4 個 1×1 預測頭；QFL/L1/GIoU/center-L1/heatmap-centerness 五 loss；Hungarian matching；partial-ignore class 權重；正規化用全域 positive count |
| `models/detection3d/partial_ignore.py` (A, 86) | class 名 → label index 解析 + 狀態 flag 正規化（3D/2D 頭共用） |
| `models/detection3d/task_modules/assigners2d.py` (A, 153) | 2D Hungarian assigner + L1/GIoU/center 三種 cost |
| `models/detection3d/task_modules/boxes2d.py` (A, 85) | cxcywh↔xyxy 轉換 + IoU/GIoU overlaps |
| `losses/detection2d/losses.py` (A, 228) | QualityFocalLoss、GIoULoss、WeightedL1Loss、HeatmapGaussianFocalLoss |
| `transforms/camera/annotations2d.py` (A, 189) | `LoadAnnotations2DFromBoxes3D`：增強後 3D box → 每相機 2D box/center/depth/label 投影（含 gravity-center z 語義修正的註解） |

### 模型（原檔修改）

| 檔案 | 變更內容 |
|---|---|
| `models/detection3d/streampetr.py` (M, +41/−4) | 接上 `img_roi_head`（train-only）與 `scheduler_config`；loss() 合併 2D losses 並傳 partial-ignore status |
| `models/detection3d/heads/streampetr.py` (M, +202/−26) | ① partial-ignore：main loss 對 negative query 清零 cone/barrier 欄、DN loss 對 noised-into-background query 同理 ② loss 正規化 local→全域（`reduce_mean_count`，DN 的 collective 門控在 rank-uniform 條件上）③ `prepare_for_loss` 多回傳 batch id 供 DN partial-ignore 使用 |
| `models/detection3d/task_modules/streaming.py` (M, +22) | 新增 `reduce_mean_count`（單 GPU 時 bitwise 不變） |
| `losses/detection3d/focal.py` (M, +9/−2) | `SigmoidFocalLoss` 接受 per-query-per-class 權重（partial-ignore 用） |

### 資料管線

| 檔案 | 變更內容 |
|---|---|
| `datamodule/common/multiview_detection3d.py` (M, +32/−10) | `SerializedSampleList` 接入（scene groups 先建好快取，避免每次反序列化整表）；輸出 `traffic_cone_barrier_status`；`_collate_fn_for(split)` |
| `datamodule/common/serialization.py` (A, 67) | `SerializedSampleList`：單一 pickle buffer + offsets，fork-shared 防 OOM（本 branch 只接 multiview；其餘 datamodule 另走 `origin/fix/dataloader_oom`） |
| `datamodule/base.py` (M, +80/−9) | ① `train_collation_map`（train-only key，val/test 不再噴 missing-key warning）② numpy scalar 保 dtype（float64 timestamp 不被降成 float32 → 256 秒量化）③ collate 分流 |

### Transforms

| 檔案 | 變更內容 |
|---|---|
| `transforms/camera/loading.py` (M, +22/−2) | `shuffle_order`（AWML 每 frame 打亂相機順序，train-only） |
| `transforms/camera/geometry.py` (M, +8/−2)、`camera_lidar/geometry.py` (M, +8/−2) | 增強後呼叫 `update_ego_poses`（camera_lidar 版為一致性；無 ego_pose 樣本 no-op） |
| `transforms/geometry3d.py` (M, +30/−1) | `update_ego_poses` 實作 + **velocity scale bug fix**（`v' = s·R·v`）⚠️ 後者影響所有用 GlobalRotScaleTrans+scale 的模型 — PR 說明要主動講 |

### 訓練基礎設施

| 檔案 | 變更內容 |
|---|---|
| `utils/schedulers/iter_warmup_epoch_cosine.py` (A, 82) | 上表 scheduler；`total_steps` 由框架既有機制自動注入 |
| `utils/lightning_loops.py` (A, 54) + `utils/runtime.py` (M, +6) | `EpochEndValidationLoop`（單一 method override）+ 裝進 trainer |
| `utils/checkpoints.py` (M, +44/−2) | shared-tensor alias 偵測：backbone/neck 同時掛在 `img_backbone` 與 `image_feature_extractor`（ONNX 匯出包裝）下，沒有它 `--weights` 的 full-coverage 檢查會誤判失敗 |
| `tools/convert_streampetr_checkpoint.py` (A, 276) + `tools/__init__.py` | mm→native 權重名稱轉換、`--bgr-to-rgb` stem 翻轉、`--drop-pattern`；tolerant unpickler 讓轉檔不需安裝 mm 系列套件 |

### Tests（8 檔，+1028）

| 檔案 | 覆蓋 |
|---|---|
| `test_streampetr_partial_ignore.py` (A, 370) | partial-ignore 3D/2D/DN 路徑 + checkpoint convert |
| `test_streampetr_loss_distributed.py` (A, 243) | 全域 loss 正規化的多 rank 等價性（gloo 2-process） |
| `test_serialization.py` (A, 99) | SerializedSampleList 行為 |
| `test_lightning_loops.py` (A, 109) | 短 epoch 也觸發 validation |
| `test_checkpoints.py` (M, +57) | alias 偵測與 full-coverage |
| `test_camera.py` (M, +49) | 2D 投影的 gravity-center z 語義 |
| `test_geometry3d.py` (M, +63) | ego-pose 折疊（camera 與 camera_lidar 一致）+ velocity scale |
| `test_point_cloud.py` (M, +38) | train_collation_map 分流 |

### 文件與雜項

| 檔案 | 變更內容 |
|---|---|
| `docs/models/streampetr.md` (M, +160/−13) | 完整 README：pretrain 轉檔 → train → eval → deploy 流程、parity 結果表（0.39127/0.36609 vs AWML 0.37521/0.35515）、eval 對齊筆記 |
| `zensical.toml` (+1) | StreamPETR 註冊進 Models nav |
| `.gitignore` (+4) | `work_dirs/`、`parity_out/`、`*.pt`、`*.npz` |

---

## 2. Code Review 結論（2026-08-06）

**整體判定：clean，無顯著 over-engineering。** DDP collective 的門控條件全部驗證過是 rank-uniform（有 distributed 測試背書）；新模組都是最小實作（cp_fpn 108 行、lightning_loops 單一 method override、serialization 67 行）；convert tool 的 tolerant unpickler 有明確理由（不然要裝整套 mm）。

發現的小問題 —— **前三項已修，修正放在 working tree 未 staged**（review 完 `git add` 即可；`git diff` 可看，共 +13/−13）：

1. ✅ **`depths_2d` 死資料已移除**：`LoadAnnotations2DFromBoxes3D` 不再產出、`train_collation_map` 不再收集（`FocalHead2D.loss` 從不消費；reference 的 depth branch 沒有 port）。
2. ✅ **focal2d.py import 已合併**：`task_modules.streaming` 的兩行 import 併成一個。
3. ✅ **annotations2d 邊界 case 已補**：center 自身不在相機前方時跳過該 box（物體橫跨相機平面時 center 像素會被 1e-6 分母撐爆）。
4. `partial_ignore.normalize_status_flags` 的遞迴 flatten 偏防禦性（collation 已宣告 list 策略），可接受不改。

修正後驗證：test_camera + test_streampetr_partial_ignore 共 30 tests 通過、pre-commit 相關 hooks 全過。

**PR 說明要主動講的行為變更**：`geometry3d.transform_boxes` 的 velocity scale fix 會輕微改變所有使用 GlobalRotScaleTrans（scale ≠ 1）模型的訓練行為 — 正確性修正，不是 regression。

---

## 3. 已移出本 branch 的東西（狀態追蹤）

| 項目 | 去向 |
|---|---|
| metrics_text_logger 整組 | 已在獨立 local branch `feat/metrics-text-logger`（abce1ea，based on origin/main，測試/hooks 全過，未 push） |
| 7 個非 multiview datamodule 的 serialization 接線 | cherry-pick `origin/fix/dataloader_oom` 或獨立 PR |
| `color_type` 參數、`score_t4metricv2_export.py`、docker/Dockerfile pixi retry | 已刪/已拆（備份在 `backup/streampetr_develop` 與 `origin/develop/streampetr_final_with_everything`） |
| 調查文件（5×framework reports、regression docs、onboarding 40 檔）、實驗工具（overfit_probe 等 3 支）、parity_out/、31.7MB .pt | 同上備份 branch |
