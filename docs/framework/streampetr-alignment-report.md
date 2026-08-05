# StreamPETR: autoware-ml vs AWML 對齊報告

**日期**: 2026-07-31（2026-08-03 補上 autoware-ml 2D GT 修正的 retrain 結果，§6.5）
**狀態**: 兩側 retrain 皆已完成。AWML aligned bf16 退步（§6.3–6.4）；autoware-ml 2D GT 修正**沒有帶來增益**（§6.5）
**工具**: single-batch overfit probe（`autoware_ml/tools/overfit_probe.py`、`tools/detection3d/overfit_probe.py`、`autoware_ml/tools/compare_overfit.py`）

---

## 0. 一頁總結

| | 內容 |
| --- | --- |
| **起點** | autoware-ml 落後 AWML ~5 mAP |
| **現況（對外請用這組）** | autoware-ml **val 0.4210 / test 0.3900**（§6.6，唯一同時不帶 2D GT bug、正規化又數學正確的 run）。對 AWML 四 run 平均領先 **+2.9 / +2.9**、對 AWML 最佳單 run 領先 **+2.5 / +1.7** |
| **本次目標** | 不動 autoware-ml（要出貨），找出 AWML 未對齊處並修正 |
| **關鍵發現** | **探針對那 +3 mAP 沒有訊號。** 探針能測的每一項現在都已對齊：輸入、forward/loss（step-0 差 0.5%）、**temporal memory（−0.3%）**、**DN（內部量逐項相同）**。它唯一測到的實質差距是 **autoware-ml 自身的 2D GT bug**（已修）。cross-attention 精度只佔 8.4%，不是主因 |
| **那 +3 到哪去了** | 不在任何單步計算裡。剩餘空間只有**完整訓練動態**：多 GPU 正規化、GridMask/aug 的隨機行為、sampler 的 epoch 序列構成。建議改用**行為診斷**（拿現有 checkpoint 做逐類別／距離分層／TP 指標比對，純推論不用 retrain），把「+3 分」變成可診斷的形狀 |
| **retrain 結果** | ❌ **AWML aligned bf16 退步到 0.4104**（baseline 0.4290–0.4331，T4Metric）。根因已定位：`avg_factor` 改錯方向，見 **§6.4** |
| **最大教訓** | **「對齊 autoware-ml」≠「改成正確的」** —— 已有**兩次** autoware-ml 才是錯的一方（§1.3 2D GT、§6.4 avg_factor）。每項對齊都須獨立論證數學正確性 |
| **2D GT retrain** | ⚠️ **修對了，但分數沒變好**：val 0.4230 → **0.4104**、test 0.3922 → **0.3869**（同評估器、同 ep10、單一變數）。差距幾乎全集中在 **barrier**（val −3.9、test −4.4）。詳見 **§6.5** |
| **loss 正規化** | ✅ **改對且分數也變好**：autoware-ml 改採跨卡全域正規化後 val 0.4104 → **0.4210**、test 0.3869 → **0.3900**（兩 split 同號）。增益**完全由 barrier 驅動**（+8.0 / +4.5），而 barrier 正是 2D GT 修正掉最多的那一類。詳見 **§6.6** |
| **待驗證** | 依 §6.4 的正確修法（雙邊全域）重跑 AWML；autoware-ml 兩次修正的淨效果需第二顆 seed 才能脫離雜訊（barrier 變異最大） |

探針的定位：它能在**幾分鐘內**回答「forward/loss 是否等價、輸入是否相同」，取代 13 小時的 retrain。但它**測的是單 batch 擬合，不是泛化**，所以最終效果仍需 retrain。

---

## 1. autoware-ml 側

### 1.1 先前已修（本次之前，帶著這些達到 0.423）

| 項目 | 說明 |
| --- | --- |
| **ego_pose 增強未折疊**（根因） | camera `GlobalRotScaleTrans` 沒把 per-frame BEV rot/scale 折進 `ego_pose`/`ego_pose_inv`，導致 StreamPETR 的 memory warp 每步錯位約 15°，模型學會忽略時間通道。修正後 AVE_car 0.276 → 0.186 |
| **validation 靜默跳過** | Lightning 凍結首個 epoch 的 `val_check_batch`，而 `GroupStreamingSampler` 每個 epoch 長度不同（3150–3351），10 個 epoch 只有 4 個真的跑到 val。新增 `EpochEndValidationLoop` |
| **checkpoint alias 去重** | backbone 註冊在兩個別名下，存檔重複計算 |

### 1.2 本次新增（工具）

| 檔案 | 說明 |
| --- | --- |
| `autoware_ml/tools/overfit_probe.py` | 單 batch overfit 探針。本次加上 **bitwise determinism**（見 §4.2）與 fingerprint 的 `tokens` 欄位 |
| `autoware_ml/tools/compare_overfit.py` | 雙側 trace 比對。本次加上 box z-origin 正規化、只印真正有差異的 key 的提示 |

### 1.3 ✅ 發現並已修正的 bug

**2D 輔助頭的 GT 垂直偏移半個物體高度**（2026-07-31 已修）

- 位置：`autoware_ml/transforms/camera/annotations2d.py:98-100`

  ```python
  gravity_centers = gt_boxes[:, :3].copy()
  gravity_centers[:, 2] += gt_boxes[:, 5] * 0.5   # 假設 z 是底面
  ```

  `_boxes3d_corners`（同檔 ~line 33）也用 `dz in (0.0, 1.0)` 從 z 往上長角點，同樣假設 z 是底面中心。
- 但 autoware-ml 的框在這個時點帶的是**重心 z** → 重複加了一次 `h/2`
- 這是 autoware-ml **內部的不一致**：3D head 用重心 z 是正確的（所以 0.423 沒問題），只有 2D 輔助投影這條路徑錯
- **證據**（見 §3.4）：`Δcy / (影像框高/2)` = **0.955 ± 0.024**，橫跨 70 個物體、深度 27–181 m
- **修正**：拿掉 `+=`，並把 `_boxes3d_corners` 改成以 z 為中心（`dz in (-0.5, 0.5)`）
- **驗證**：修正後 autoware-ml 的 2D GT 與 AWML 獨立推導的值吻合到 **mean 0.0000**（最大殘差 0.17 px，僅來自 clip/rounding）。修正前 center y 平均差 −5.026、最大 15.08
- **回歸測試**：`test_camera.py::test_load_annotations_2d_treats_box_z_as_gravity_center` 把慣例釘住（單一 2 m 立方體 + 針孔相機，中心必須投影在主點而非上緣）。147 個 transform 測試全過
- ✅ **retrain 已完成（2026-08-03 補測，§6.5）**：val 0.4230 → **0.4104**、test 0.3922 → **0.3869**。**修正在數學上是對的，但沒有換來 mAP** —— 已驗證的 0.423/0.392 是在**帶著這個 bug** 的情況下達成的，修掉之後反而各掉 1.3 / 0.5 點

> **重要**：這代表「AWML 對齊 autoware-ml」的原則**在這一項上不適用** —— AWML 一直是對的，照著 autoware-ml 改會把 bug 複製過去。

---

## 2. AWML 側 —— 本次修正

### 2.1 Cross-attention 精度不誠實（最實質）

**檔案**: `projects/StreamPETR/stream_petr/models/utils/attention.py`

- **問題**：`FlashAttention.forward` 無條件呼叫 `q.half()` / `kv.half()`。它甚至**偵測得到** fp32 輸入（`fp16 = q.dtype in [float16, bfloat16]`）並把**輸出**轉回 fp32，但**運算本身仍是 fp16**
- **修正**：`attn_dtype = q.dtype if fp16 else torch.float16` —— 尊重呼叫者 dtype。fp32 輸入仍走舊 fp16 路徑（flash_attn 只有 16-bit kernel）；另加 `STREAMPETR_ATTENTION_DTYPE={fp16,bf16,fp32}` 與 `STREAMPETR_FP32_ATTENTION=1` 供探針做隔離實驗
- **實質影響：小。** 真實配置下 cross-attn dtype 只佔擬合差距的 **8.4%**（見 §3.3）

> ⚠️ **這個 bug 對 AWML 歷史四次 run 完全沒有影響。** AWML 原本的 recipe 是 **fp16 AMP**，在 fp16 autocast 下 cross-attn 的 in_proj 產出本來就是 fp16 → **`q.half()` 是恆等操作**。它**只污染了 7/29 那次 bf16 ablation**（bf16 通得過 `fp16` 判斷，被降回 fp16），所以那次「bf16 沒有幫助（val 0.391 vs fp16 0.396）」的結論不成立。
>
> 修這一項的理由是**正確性**（設定的精度必須到得了 attention，否則所有 precision ablation 都會靜默失效），不是因為它是 mAP 差距的主因 —— 它不是。

**架構釐清**：AWML 的 **self-attention 原本就已經是 fp32 island**（stock `petr_transformer.py` 把 `self_attn` 包在 `autocast(enabled=False)`，與 autoware-ml 同一招）。真正硬寫 fp16 的**只有 cross-attention**。對齊後的數值佈局 = autoware-ml：backbone/FFN bf16、self-attn fp32、cross-attn bf16。

### 2.2 Loss 正規化的多 GPU 不一致 —— ⚠️ **我第一次改錯方向，見 §6.4**

**檔案**: `streampetr_head.py`（matched + DN 兩處）、`focal_head.py`

- **問題（成立）**：同一個 `loss_single` 裡，`loss_cls` / `loss_iou` 用**本地**正樣本數正規化，而 bbox / centerness / centers2d 用 `reduce_mean` 的**跨 GPU 平均** → 多卡時各 loss 項權重不一致。這個不一致是真的
- **第一次的修正（錯誤）**：把 bbox 也改成 `max(num_total_pos, 1)`，理由是「對齊 autoware-ml，它一律用本地計數」
- **為何錯**：`reduce_mean` **才是數學上正確的全域正規化**。本地正規化讓每張 GPU 票票等值（不管身上有幾個物體），而「每物體平均損失」要求每個物體等值。詳細推導與實測見 **§6.4**
- **正確的修法**：bbox 改回 `reduce_mean`，並設 `sync_cls_avg_factor=True` 讓 cls 也走全域正規化 —— 這才真正解決了原本的不一致（往正確的方向統一，而不是往錯的方向）
- **注意**：`reduce_mean` 在單卡上是**恆等變換**，所以探針完全看不到這項。這正是它逃過所有探針驗證、直到 2-GPU retrain 才暴露的原因

### 2.3 影像 resize 插值核

**檔案**: `transform_3d.py`

- **問題**：AWML 用 PIL `Image.resize`（bicubic，且 PIL 會依縮小倍率放大濾波器支撐 → **會** antialias）；autoware-ml 用 `cv2.resize` 預設 INTER_LINEAR（固定 2×2 鄰域 → **不** antialias）
- 實際縮放 2880×1860 → 743×480 = **0.258 倍**，走樣嚴重
- **實測**（該幀 6 張影像）：std 比值 cv2/PIL = **1.0413**，與 fingerprint 觀察到的 1.0383 吻合；mean 幾乎不變（0.1%）
- **修正**：AWML 改用 `cv2.resize(..., INTER_LINEAR)`。fingerprint 的 `img` 差距 **0.03084 → 0.00202**（進入容差內）

### 2.4 GT hygiene filter（保險，實測無作用）

**檔案**: `dataset.py`，新增 `StreamPETRDataset.parse_ann_info` override

- 加入等同 autoware-ml `box_is_physical` 的過濾（finite、dims > 0、速度 ≤ 150），僅作用於 StreamPETR，不影響共用 `T4Dataset` 的其他專案
- **實測結果：整份 train pkl 共 3,140,324 個 instance，丟掉 0 個**（無 non-finite、無非正尺寸、無超速）
- → **先前調查中的「GT hygiene」假說在此資料集上是空的**，不可能解釋任何差距。filter 純作保險保留

### 2.5 其他

| 項目 | 說明 |
| --- | --- |
| **新 config** | `..._j6gen2_partialignore_aligned_bf16.py`：繼承 fp16 主 config，只改 `dtype="bfloat16"` + 關 loss_scale。合併已驗證 |
| **過時註解** | 主 config 中「bf16 沒好處」的註解已改寫，註明失效原因 |
| **探針** | 新增 `tools/detection3d/overfit_probe.py`（含 determinism、tokens） |

### 2.6 Working tree 中先前既有、會一起進 retrain

| 項目 | 備註 |
| --- | --- |
| DN gravity-center z 修正 | DN 分支原本用底面中心 z，matched 分支用重心 → 共用迴歸分支被 10:1 的 DN 列拉偏 |
| 1px flip 修正 | PIL FLIP 映射 `x → (width-1)-x`，homography 原本用 `width` |
| **per-camera 增強取樣** | ⚠️ 對齊 autoware-ml，但 exp2 唯一一次實測**最低分**（0.3868）。是唯一有反向證據的對齊項 |

---

## 3. 探針的關鍵量測

### 3.1 資料集順序不同（第一次比對完全無效）

兩邊含**完全相同的 3208 個樣本、137 個場景，每個場景內部的幀順序與內容都一致**，但：

- **AWML** 依 `scene_token` 排序場景
- **autoware-ml** 保持 pkl 原始順序

→ `awml[0..1]` 實際等於 `aml[857..858]`。用相同 `--start-index` 比對，是拿兩個無關場景在比（gt 框數 16 vs 2，錄製時間差 13 天）。

**教訓**：fingerprint 原本沒有樣本識別碼，「是不是同一幀」要靠 gt_counts/timestamp 反推。已在兩側 fingerprint 加上 `tokens`。

### 3.2 對齊後：forward + loss 等價

| | 值 |
| --- | --- |
| step-0 loss（autoware-ml / AWML） | 11.4350 / 11.4972，差 **−0.5%** |

→ 相同權重、相同輸入下 **forward 與 loss 計算等價，模型數學沒有差異**。這是最穩固的結論（step-0 在同設定下可重現到小數四位）。

### 3.3 Attention 精度**不是**主因（僅 8.4%）

**先前的錯誤歸因**：早期探針跑 `--precision fp32`，量到把 AWML cross-attn 改 fp32 可讓總差距從 +15.3% → −1.8%，據此判定 attention 精度是主因。**那是探針造成的假象** —— 在 fp32 模式下，autoware-ml 的 cross-attn 真的是 fp32，而 AWML 被 `.half()` 強制成 fp16，**這個 fp32 vs fp16 的對比在真實訓練中不存在**（真實是 bf16 vs fp16，都是 16-bit）。

**真實訓練配置下的三個 arm**（determinism 在 fp16 下也已驗證逐位元相同）:

| Arm | 配置 | step-0 | 尾段平均 |
| --- | --- | --- | --- |
| **A** | autoware-ml bf16-mixed（真實） | 10.8186 | **1.4723** |
| **B** | AWML fp16 AMP（真實） | 11.5996 | **1.3017** |
| **C** | AWML fp16 + **只把 cross-attn 換 bf16** | 11.4765 | **1.3161** |

- 誠實的差距（A−B）: **+0.1706（+13.1%）**
- cross-attn dtype 單獨貢獻（C−B）: **+0.0144（+1.1%）→ 只解釋 8.4%**
- 注意符號：換成 bf16 讓 AWML 擬合**稍微變差**，所以「對齊成 bf16」本身不是改善

**理論也支持這個結果**：fp16 的 mantissa（10 bits）**比 bf16（7 bits）多**；bf16 的動態範圍優勢在此用不到（head_dim 32、scale 1/√32，logits 約 O(10)，離 fp16 上限 65504 很遠）；flash_attn 與 SDPA **內部都用 fp32 累加**。本來就不該期待大效應。

### 3.3b 真正的差距：91% 在 2D 輔助頭，而那是 autoware-ml 自己的 bug

| 群組 | A−B（真實配置） | C−B（只變 cross-attn） |
| --- | --- | --- |
| **2D 輔助頭** | **+0.1547（91%）** | −0.0140 |
| 3D decoder bbox | +0.0131（8%） | +0.0239 |
| 3D 分類 | +0.0028（2%） | +0.0045 |

2D 輔助頭的差距在**所有**精度組合下都穩定在 +0.12～0.15（fp32、fp32-attn、bf16/fp16）→ 這是**結構性**的，就是 §1.3 那個 h/2 的 2D GT bug，不是數值問題。

**修掉那個 bug 後的實測（同為真實 bf16 配置）**:

| | step-0 | 尾段平均 | 對 AWML 的差距 |
| --- | --- | --- | --- |
| autoware-ml（修正前） | 10.8186 | 1.4723 | +0.1706（+13.1%） |
| **autoware-ml（修正後）** | **10.0979** | **1.3957** | **+0.0940（+7.2%）** |

| 群組 | 修正前 | 修正後 |
| --- | --- | --- |
| **2D 輔助頭** | +0.1547 | **+0.0512** |
| 3D decoder bbox | +0.0131 | +0.0397 |
| 3D 分類 | +0.0028 | +0.0031 |

`loss_iou2d` 從 0.6470 降到 0.5636（AWML 0.5194）。→ **2D 輔助頭的差距關掉了約 2/3**，證實該 bug 就是那 91% 的主要成分。殘餘 +0.0512 來自 0.17 px 的 clip 差異、兩邊 2D head 實作差異，以及此比較本身帶有 bf16 vs fp16 的精度差。

> ### ⚠️ 最關鍵的結論
>
> **3D 路徑在探針裡是平手的**（bbox +0.013、cls +0.003），唯一的實質差距是 autoware-ml 自己的 2D GT bug —— **而 autoware-ml 的 mAP 反而高 3 分**。
>
> 也就是說：**這支單 batch 探針對那 +3 mAP 完全沒有訊號。** 它測得到的（2D aux）與 mAP 差距無關；答案在它測不到的地方 —— 多 GPU loss 正規化、被關掉的 augmentation / DN / GridMask / temporal memory 路徑、或 ego_pose 修正在完整訓練下的動態。
>
> **繼續用單 batch 探針切 3D 路徑是死路。**

### 3.4 2D GT 偏移的裁定

同樣 70 個物體、同樣標籤、同樣深度、x 吻合到 0.06 px，但 **cy 平均差 −5.03 px**，且**隨 1/depth 變化**（cv 0.226）而非固定偏移（cv 0.550）。

決定性檢定：**`Δcy / (影像框高/2)` = 0.955 ± 0.024**（70 物體，深度 27–181 m）→ 偏移量精確等於每個物體高度的一半。

哪邊錯——用 pkl 自己的矩陣獨立投影（深度 180.91）：

| | CAM_BACK_LEFT | CAM_BACK_RIGHT |
| --- | --- | --- |
| 用 pkl 的 z **原值**投影 | 215.90 | 220.26 |
| 用 **z + h/2** 投影 | 213.59 | 217.97 |
| **AWML** 實際值 | **215.90** ✅ | **220.26** ✅ |
| **autoware-ml** 實際值 | 213.59 ❌ | 217.97 ❌ |

兩相機皆吻合到小數兩位 → **AWML 正確，autoware-ml 多加了一次 h/2**（詳見 §1.3）。

補充：pkl 的 `instances` **只有 `bbox_3d`**，沒有任何預存 2D 標註 —— 兩邊都是自己從 3D 框算 2D GT。

---

## 4. 過程中被推翻的假說（避免重複調查）

| 假說 | 結論 |
| --- | --- |
| 影像統計差異來自 JPEG 解碼 / resize | ❌ 是 §3.1 的場景不匹配 |
| GT hygiene（`box_is_physical`） | ❌ 實測 3,140,324 個 instance 丟 0 個，假說為空 |
| fp16 vs bf16 已排除 | ❌ 舊 ablation 被 `.half()` 汙染，從未真正測到（§2.1） |
| bbox `avg_factor` / `reduce_mean` | ⚠️ 單卡下恆等，探針測不到；但多卡下是真差異，已修（§2.2） |
| autoware-ml 的 fp32 self-attn island | ⚠️ `--precision fp32` 下是 no-op；且 AWML self-attn 本來就是 fp32（§2.1） |
| per-camera 增強粒度 | ❌ exp2 實測最低（0.3868） |
| 「75% 差距在 2D 輔助頭 / 63% 是 loss_iou2d」 | ❌ 噪聲期讀數，作廢。零噪聲下實為 54% / 42%（§3.3） |

### 4.2 探針本身的重大修正：determinism

**這是讓探針從「無用」變成「利器」的關鍵。**

- 修正前：兩次**完全相同**的 autoware-ml run，尾段平均差 **0.115（7.9%）** —— 與被分析的差距同一量級，任何歸因都不成立
- 光加 `torch.use_deterministic_algorithms(True)` **不夠**：
  - **autoware-ml**：SDPA 的 flash / mem-efficient backend 反向傳播非決定性，該旗標**只警告不覆蓋**。必須額外強制 math backend（`enable_flash_sdp(False)` / `enable_mem_efficient_sdp(False)` / `enable_math_sdp(True)`）
  - **AWML**：直接用 `flash_attn` 套件（自訂 CUDA op），torch 的旗標**看不到它也不會警告** —— 那次跑一個警告都沒印，看起來成功實際完全無效。flash_attn 2.7.3 支援 `deterministic=True`，掛在 `torch.are_deterministic_algorithms_enabled()` 上
- **驗證**：兩側各跑兩次，200 步紀錄**逐位元相同**。散布 0.115 → **0**
- 副作用：強制 autoware-ml 走 math SDPA 改動其 attention 數值，step-0 差距從 +0.5% → +1.6%。所以「forward/loss 等價」精確說是**吻合到約 1.6%**，殘差來自兩邊固有不同的 attention kernel

---

## 5. 目前**仍未**對齊的項目

### 5.1 已修正（原列為「刻意不對齊」）

| 項目 | 狀態 |
| --- | --- |
| **autoware-ml 的 2D GT h/2 bug** | ✅ 2026-07-31 已在 autoware-ml 側修正（§1.3）。修正後兩邊 2D GT 吻合到 0.0000。retrain 已完成（§6.5）：**val 0.4104 / test 0.3869，比帶 bug 的 0.4230 / 0.3922 低** —— 正確性有了，精度沒有 |

### 5.2 無法對齊（固有）

| 項目 | 說明 |
| --- | --- |
| **attention kernel 實作** | `flash_attn` 套件 vs PyTorch SDPA。只對齊了 dtype，累加順序差異固有（step-0 那 ~1.6% 殘差的來源）。**注意：這不代表 autoware-ml 沒有 flash attention** —— 見下方澄清 |

> #### ⚠️ 澄清：autoware-ml 的 cross-attention **已經是 flash attention**
>
> 上一列容易被誤讀成「AWML 有 flash、autoware-ml 沒有」。實際上兩邊的結構是對應的：
>
> | | self-attention | cross-attention |
> | --- | --- | --- |
> | 原版 StreamPETR / AWML | mmcv `MultiheadAttention`，`petr_transformer.py:716` 以 `autocast(enabled=False)` 包成 **fp32 島** | `PETRMultiheadFlashAttention` → `FlashMHA` → **`flash_attn` 套件** |
> | autoware-ml | `nn.MultiheadAttention`，`heads/streampetr.py:86` 同樣 `autocast(enabled=False)` + `.float()` → **fp32 島** | `nn.MultiheadAttention`（`need_weights=False`）在 bf16 autocast 下 → SDPA → **flash 後端** |
>
> **兩邊都只有 cross-attn 走 flash，self-attn 都刻意不走。**
>
> 實測（2026-08-03，容器內、真實 shape B=8 / 8 heads / head_dim 32 / query 644 / image tokens 6×30×40=7200）：
> autocast bf16 下 profile 到的 CUDA kernel 是
> `pytorch_flash::flash_fwd_kernel<...bfloat16_t>` 與
> `pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<...>` ——
> `pytorch_flash` 即 PyTorch vendored 的 FlashAttention-2，與 `flash_attn` 套件**同一套演算法**。
>
> backend 可用性也一致：bf16 / fp16 下 flash 與 mem-efficient 皆可用；**fp32 下 flash 不可用**
> （flash kernel 只有 16-bit 版本）—— 這正是兩邊的 fp32 self-attn 島都自然落到 mem-efficient 的原因，
> 不是 autoware-ml 漏配。
>
> **結論：不建議改用 `flash_attn` 套件。** ①效能上拿不到東西（同一套 kernel）；②它唯一多出來的
> varlen/unpadding 路徑用不到（StreamPETR 的 `key_padding_mask` 傳 `None`）；
> ③`nn.MultiheadAttention` 可 ONNX export，自訂 CUDA op 不行，而 autoware-ml 是要出貨的一邊；
> ④會多一個綁 CUDA/torch 版本的硬編譯相依；⑤會動到數值 → 又要一次 13 小時 retrain 重新驗證。
> 真正未對齊的是**同一演算法兩個實作之間固有的累加順序差異**，換套件消除不了。
| **評估器差異** | autoware-ml MeanAP 把超出範圍的預測算 FP、用 precision envelope、LiDAR 原點；T4Metric 過濾預測、raw precision、base_link 原點 → 讀同一份權重差 3.4–3.8 pt。**不可跨框架直接比 raw mAP**，比較必須用同一評估器 |
| **群組層面殘差** | 真實配置（aml bf16 / AWML fp16）、2D GT 修正後的殘差：2D aux **+0.0512**、3D decoder bbox **+0.0397**、3D cls **+0.0031**（總 +7.2%）。2D aux 那項在修正前是 +0.1547，修掉 h/2 bug 後關閉約 2/3。剩下的量級已接近 bf16-vs-fp16 精度差與 attention kernel 累加順序差可解釋的範圍，未再細分 |

### 5.3 探針視野之外（只有 retrain 能回答）

| 項目 | 說明 |
| --- | --- |
| **多 GPU loss 正規化** | 單卡上是恆等（`reduce_mean` 無 dist 時原樣回傳），只有 2-GPU 才生效。**這正是 §6.4 那個錯誤修改逃過所有探針驗證的原因** |
| **bf16 訓練效果** | smoke 只證明能跑、無 NaN（30 步 loss 12.25→3.23，所有 loss 項無 NaN/Inf） |
| **GridMask / augmentation 的實際隨機行為** | 兩邊 RNG 實作不同 → 抽樣序列必然不同，無法乾淨比對 |
| **Sampler 的整個 epoch 序列構成** | 探針手動取 2 幀，測不到跨 epoch 的序列組成 |
| **場景串接順序** | 已確認不同（§3.1）。訓練有 shuffle，主要影響重現性與工具對齊，未必影響精度，未動 |

**DN 與 temporal memory 已於 2026-07-31 用探針開關逐一測過（見 §5.4），兩者皆對齊**，已從本表移除。

---

### 5.4 DN 與 temporal memory 盲區檢查（2026-07-31）

探針有 `--keep-dn` / `--keep-memory` 開關，把先前刻意關閉的兩條核心路徑逐一打開，各跑兩側（aml `--precision bf16` / AWML `--precision fp16`，即真實訓練配置）：

| 路徑 | aml 尾段 | AWML 尾段 | 差距 |
| --- | ---: | ---: | ---: |
| 基準（DN、memory 都關） | 1.3957 | 1.3017 | +7.2% |
| **開 temporal memory** | 1.4603 | 1.4643 | **−0.3%** |
| **開 DN** | 1.5523 | 1.3885 | +11.8% |

**Temporal memory / ego_pose warp：對齊。** 打開後兩邊收斂到 −0.3%，也佐證 §1.1 的 ego_pose 修正是完整的。

**DN：也是對齊的。** 逐項核對 step-0 的內部量，**全部相同**：

| | aml | AWML |
| --- | --- | --- |
| `num_total_pos`（DN query 數） | 40 | 40 ✓ |
| `cls_scores` shape | [40, 7] | [40, 7] ✓ |
| foreground / background | 12 / 28 | 12 / 28 ✓ |
| `cls_avg_factor` | 8.836 | 8.836 ✓ |
| `label_weights` numel / sum / 零比例 | 280 / 224.0 / 0.2 | 280 / 224.0 / 0.2 ✓ |

噪聲產生程式碼也**逐行相同**（`diff = scale/2 + trans`、`rand_prob = rand_like*2−1`、`center += rand_prob*diff*noise_scale`、pc_range 正規化、`clamp(0,1)`、`norm(rand_prob) > split` 轉背景、`single_pad`），兩邊 `noise_scale` 都是 1.0、`dn_weight` 都是 1.0。

> #### ⚠️ 一個我當下誤讀、必須記下來的教訓
>
> 我最初看到 **step-0 的 DN loss 差 43%（分類項差 82%）**，判定為「實作差異」。**那是錯的。**
>
> step-0 的值距離它自己的尾段平均有 **+7.7 / +12.3 個標準差** —— 因為 step 0 還沒訓練過這個 batch，loss 本來就在高點。拿這種點去跨框架比對毫無意義。改看尾段平均，比值只有 **cls 1.073 / bbox 1.209**，而且兩邊 RNG 不同、DN 噪聲抽樣必然不同（每步 cv 高達 21%）。
>
> **教訓：跨框架比對只能用尾段平均，永遠不要用單一 step 的快照** —— 這跟 §4.2 的 determinism 是同一類錯誤（拿雜訊當訊號）。

---

## 6. 驗證與結果

### 6.1 已完成的驗證

| 檢查 | 結果 |
| --- | --- |
| 修改後探針（預設路徑）vs 修改前 trace | **逐位元相同** —— 三項程式修改在已驗證路徑上可證明為 no-op |
| 修改後探針（fp32-attn 路徑）vs 修改前 | **逐位元相同** |
| bf16 smoke（30 步，走真 bf16 cross-attn） | loss 12.25 → 3.23，**所有 loss 項無 NaN/Inf** |
| `aligned_bf16` config 合併 | 正確（bf16、loss_scale off、clip/lr/pkl 全保留） |
| 兩側 determinism | 各跑兩次，200 步**逐位元相同** |

### 6.2 Retrain（已完成 → 結果見 §6.3 / §6.5）

```bash
docker exec -it awml_petr bash -lc "cd /workspace && \
  bash tools/detection3d/dist_script.sh \
    projects/StreamPETR/configs/t4dataset/t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore_aligned_bf16.py \
    2 train"
```

**判讀基準**（皆為同一評估器）:

| Run | val mAP | test mAP |
| --- | --- | --- |
| AWML zfix | 0.3959 | 0.3728 |
| AWML orig | 0.3947 | 0.3640 |
| AWML bf16（無效的那次） | 0.3912 | 0.3580 |
| AWML exp2（per-camera + 1px） | 0.3868 | 0.3504 |
| **AWML 散布** | **0.9 pt** | **2.2 pt** |
| **autoware-ml**（帶 2D GT bug） | **0.4230** | **0.3922** |
| **autoware-ml + 2D GT 修正**（§6.5） | **0.4104** | **0.3869** |
| **AWML aligned bf16（本次）** | _未跑_ | _未跑_ |

> `AWML aligned bf16` 這列是「AWML 權重過 autoware-ml evaluator」的交叉評估，
> **沒有跑**。該 run 的結果改以 AWML 自家 T4Metric 判讀，見 §6.3
> （best val 0.4104 —— 與上一列 autoware-ml 的 0.4104 **只是巧合，評估器不同、
> 不可對照**）。

判讀：

- val 明顯衝出 **0.396**（歷史最高）往 0.42 靠 → 歸因成立
- 小幅變動則落在噪聲內 —— 待測效應 ~3 pt 只略高於 test 端 2.2 pt 的散布，**可能需要第二顆 seed 才能下定論**

### 6.3 結果：**退步了**（2026-07-31）

同評估器（T4Metric，AWML 自己的評估器）對照：

| Run | best val mAP | 峰值 epoch |
| --- | --- | --- |
| baseline zfix（run 1） | **0.4331** | 10 |
| baseline zfix（run 2） | 0.4290 | 10 |
| bf16（舊的無效 ablation） | 0.4276 | 10 |
| bf16_exp2（per-cam aug + 1px） | 0.4217 | 10 |
| **aligned_bf16（本次）** | **0.4104** | **9**（ep10 反轉降到 0.4061） |

**退步 −1.9～−2.3 分，是四次中最低**，而 AWML 自己的 run-to-run 散布只有 ~0.4 分 → 在噪聲之外，是真的退步。

軌跡形狀也不同：ep2 還跟得上（0.2524 vs 0.2507/0.2559），從 **ep4 起落後 3 分**（0.2657 vs 0.3016/0.2950）且從未追回，最後 **ep10 反轉下降**（兩次 baseline 都在 ep10 達到最高）。

逐項拆解（都是 T4Metric）:

| 增量 | mAP | 代價 |
| --- | --- | --- |
| baseline | 0.4290 | — |
| + bf16 only | 0.4276 | −0.14（噪聲內） |
| + per-cam aug & 1px（exp2） | 0.4217 | **−0.59** |
| + 本次三項新對齊 | 0.4104 | **−1.13** |

---

### 6.4 ⚠️ 根因：`avg_factor` 改錯方向（**這一節是本報告最重要的教訓**）

#### 診斷方法

我只改了 `loss_bbox` 的 avg_factor，**沒動** `loss_cls` 的。所以比較這兩者的變化形狀，就能把「正規化改動」與「真實擬合變差」分開。

Epoch 10 訓練 loss 分量：

| run | `loss_cls`（正規化未變） | `enc_loss_iou`（未變） | `loss_bbox`（**被改**） | total |
| --- | --- | --- | --- | --- |
| baseline 1 | 0.3740 | 1.1093 | 0.6831 | 14.514 |
| baseline 2 | 0.3766 | 1.1094 | 0.6842 | 14.567 |
| exp2 | 0.3871 | 1.1288 | 0.7069 | 14.944 |
| **aligned** | **0.3951** | **1.1330** | **0.7615** | 15.666 |

兩段增量的**形狀**決定性地不同：

| 增量 | `loss_cls` | `loss_bbox` | 判讀 |
| --- | --- | --- | --- |
| baseline → exp2 | +3.50% | +3.48% | **均勻** → 真的是資料變難擬合（per-camera aug） |
| exp2 → aligned | +2.07% | **+7.72%** | **選擇性** → 不是資料效應，是**正規化**效應 |

#### 為什麼 `reduce_mean` 才是對的

具體例子，2 張 GPU 拿到不同數量的 GT：

| | GT 框數 | bbox 誤差總和 | 每框平均 |
| --- | --- | --- | --- |
| GPU 0 | 2 | 8.0 | 4.0（難的幀） |
| GPU 1 | 8 | 16.0 | 2.0（好的幀） |

「每框平均誤差」的正確答案 = (8+16)/(2+8) = **2.4**

- **`reduce_mean`**（兩卡 avg_factor 都 = mean(2,8) = 5）: GPU0 = 8/5 = 1.6、GPU1 = 16/5 = 3.2 → DDP 平均梯度 = **2.4** ✅
- **本地計數**: GPU0 = 8/2 = 4.0、GPU1 = 16/8 = 2.0 → DDP 平均 = **3.0** ❌ 高估 25%

**核心**：本地正規化讓**每張 GPU 票票等值**，不管它身上有幾個物體 —— GPU 0 的 2 個框影響力等於 GPU 1 的 8 個框。而「每物體平均損失」的定義要求**每個物體等值**，那正是 `reduce_mean` 做的事。副作用是系統性高估（Jensen：平均比值 ≥ 比值的平均），且 `n` 變異越大高估越多。

#### 機制的可檢驗推論 —— 數據吻合

同一改動同時作用在 3D 與 2D 頭，但兩者正樣本數量級差很多：

| | 每幀正樣本（探針那幀） | 1/n 變異 | **實測放大** |
| --- | --- | --- | --- |
| `loss_bbox`（3D） | **2** | 大 | **+7.72%** |
| `enc_loss_bbox`（2D） | **35** | 小 | **+1.65%** |

資料集變異度也支持：每幀有效 GT 數 **平均 45.4、std 31.3（CV 0.69）**，p5=12 → p95=92 差 8 倍。

而 loss 是拿去反傳的，所以這不是報表假象：**bbox 梯度相對 cls 實質增重約 8%**，多任務權衡被改掉了 —— 這正是會值 1～2 mAP 的那種改動。

#### 結論與正確修法

> **`reduce_mean` 是對的，我把它改成了有偏的版本。** 原本那個「內部不一致」是真的，但**正確的統一方向是全域，不是本地**：
>
> 1. `loss_bbox` 的 avg_factor **改回 `reduce_mean`**（三處）
> 2. **設 `sync_cls_avg_factor=True`** → `cls_avg_factor` 也走 `reduce_mean`，cls 與 bbox 都全域正規化、內部一致
>
> 這比 AWML 原版和我的版本都好，且真正解決了原本的不一致。

#### 🔴 這一節的教訓（請不要再犯）

1. **「對齊 autoware-ml」不等於「改成正確的」。** 這已經是**第二次** autoware-ml 才是錯的一方（第一次是 §1.3 的 2D GT h/2 bug）。每一項對齊都必須獨立論證它在數學上正確，不能用「aml 的 mAP 較高」當作它每個實作細節都對的理由
2. **單卡探針對多卡正規化完全盲目**（`reduce_mean` 在單卡是恆等變換）。任何「探針看不到」的改動都必須另外找驗證手段，不能只靠 code review 就送進 retrain
3. **檢查改動的影響形狀，不只看總 loss。** 這次能定位就是因為只改了 bbox 沒改 cls —— 選擇性放大 vs 均勻上升，一眼就能區分正規化效應與資料效應
4. **獨立問題**：per-camera aug 從 `loss_cls` 的均勻 +3.5% 確認是真的讓資料更難擬合（mAP 0.4290 → 0.4217）。它是「更強的增強」，理論上該幫助泛化卻沒有 —— 可能需要更長 schedule 才划算，屬另一個決定

---

### 6.5 autoware-ml 側的 2D GT 修正 retrain（2026-08-03 補測）

§1.3 的 h/2 修正在 2026-07-31 01:40 進入 working tree，retrain 於同日 14:48 啟動、
08-01 00:36 完成（run `859067df…`，10 epochs、2 GPU）。

**這是乾淨的單變數對照。** 比對兩次 run 之間所有異動檔案的 mtime，訓練路徑上
**只有 `transforms/camera/annotations2d.py`（+ 其單元測試）** 在 baseline run
（`4934d680…`，07-28 01:39 啟動）之後改過；`tools/` 下的探針不進訓練，
`heads/streampetr.py` / `heads/focal2d.py` 的 07-31 15:41 異動是**純註解**且在本次
run 啟動之後。

評估方式與 §8 的 baseline 完全一致：ep10 權重、`autoware-ml test` fresh run、
autoware-ml 自家 evaluator、同一組 val / test frames。

#### 結果：修對了，但分數沒變好

| | baseline（帶 bug） | **2D GT 修正後** | Δ |
| --- | ---: | ---: | ---: |
| **val mAP** | 0.4230 | **0.4104** | **−0.0126** |
| val mAP（0–51.2m） | 0.431 | 0.4183 | −0.013 |
| **test mAP** | 0.3922 | **0.3869** | **−0.0053** |
| test mAP（0–51.2m） | 0.399 | 0.3943 | −0.005 |

訓練期 checkpoint callback 記到的 best val 是 0.41180（ep10，全程單調上升、
無 ep10 反轉）；fresh test run 得 0.4104，兩者差 0.14 點，屬正常重跑抖動。

#### 逐類：差距幾乎全在 barrier

| class | val baseline | **val 修正後** | Δ | test baseline | **test 修正後** | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| car | 0.571 | 0.572 | +0.001 | 0.572 | 0.578 | +0.006 |
| truck | 0.467 | 0.449 | −0.018 | 0.465 | 0.460 | −0.005 |
| bus | 0.631 | 0.609 | −0.022 | 0.569 | 0.578 | +0.009 |
| bicycle | 0.455 | 0.446 | −0.009 | 0.365 | 0.366 | +0.001 |
| pedestrian | 0.387 | 0.390 | +0.003 | 0.382 | 0.381 | −0.001 |
| traffic_cone | 0.227 | 0.222 | −0.005 | 0.201 | 0.197 | −0.004 |
| **barrier** | 0.224 | **0.185** | **−0.039** | 0.192 | **0.148** | **−0.044** |
| **mAP** | **0.4230** | **0.4104** | **−0.0126** | **0.3922** | **0.3869** | **−0.0053** |

- **test split 上，barrier 一類就解釋掉全部的下降**：其餘六類淨和 **+0.006**
  （即 +0.0009 mAP），barrier 一類貢獻 −0.0063。
- val split 上 barrier 佔 44%，其餘由 bus（−0.022）與 truck（−0.018）分擔，
  但這兩類在 test 上是 **+0.009 / −0.005** —— **方向不一致，是雜訊的形狀**。
- barrier 則是**兩個 split 同號、同量級（−3.9 / −4.4）** —— 這一項不像雜訊。

#### 時序通道未受影響（符合預期）

AVE（@optimal 2m，越低越好）在兩份權重之間基本持平，證實這次改動沒有碰到
ego_pose / memory 那條路徑：

| class | val baseline | val 修正後 | test 修正後 |
| --- | ---: | ---: | ---: |
| car | 0.186 | 0.185 | 0.151 |
| truck | 0.286 | 0.295 | 0.302 |
| bus | 0.157 | 0.148 | 0.132 |
| bicycle | 0.489 | 0.486 | 0.603 |
| pedestrian | 0.316 | 0.316 | 0.314 |

#### 判讀

1. **對外的領先結論仍然成立，但幅度縮水。** 對 AWML 四 run 平均（val 0.3921 /
   test 0.3613）仍領先 **+1.8 / +2.6** 點，對 AWML 最佳單 run（val 0.3959 /
   test 0.3728）領先 **+1.5 / +1.4** 點。原本的 +3.1 / +3.1 已不適用，
   **對外引用請改用 0.4104 / 0.3869**（那才是不帶已知 bug 的數字）。
2. **統計強度不足以宣稱「退步」。** 兩側各只有一顆 seed。以 AWML 量到的 run 間
   散布為尺（val 0.9 點、test 2.2 點），test 的 −0.5 **完全在雜訊內**，
   val 的 −1.3 只是略微超出。能確定的是：**這個修正沒有帶來增益**。
3. **唯一有跨 split 一致訊號的是 barrier。** 值得注意的是 §8 曾記到 AWML 的
   barrier 在 test 上塌到 0.106，而 autoware-ml 帶 bug 時是 0.192 —— barrier 這
   一類對 2D 輔助頭的目標特別敏感。一個合理但**未經驗證**的解釋是：
   `loss_bbox2d` / `loss_centers2d` 的權重是在偏移過的目標上調出來的，
   目標一改，2D 輔助訊號與 3D 主頭之間的權衡就跟著變 —— 這與 §6.4 的
   avg_factor 是**同一類問題**（改動本身正確，但 recipe 是在舊行為上調校的）。
4. **不建議回退。** 舊行為是可證明的錯誤（§3.4 用 pkl 自己的矩陣獨立投影裁定），
   保留它等於把一個已知 bug 當成超參數。正確的下一步是**在修正後的目標上重調
   2D 輔助頭的 loss 權重**，而不是把 bug 放回去。

> **第三次印證同一件事**：§1.3 的 bug 是真的、§6.4 的 `reduce_mean` 是對的、
> 這次的修正也是對的 —— **但「正確」與「分數更高」是兩件獨立的事**。
> 任何動到 loss 目標或正規化的修正，都必須預期它會使既有的權重 recipe 失準，
> 並把「重調權重」算進成本裡。

---

### 6.6 autoware-ml 側改採全域 loss 正規化（2026-08-03，決定 + 實作，retrain 待跑）

使用者裁定：既然多卡訓練下本地計數在數學上是錯的（§6.4），autoware-ml 應該改。
上游標準做法已逐一查證：

| 參考實作 | bbox 正規化 | cls 正規化 |
| --- | --- | --- |
| **DETR 原版**（facebookresearch） | `num_boxes` **all_reduce ÷ world_size**（全域） | CE 對全部 query 取平均（分母固定，天然無偏） |
| **mmdetection** Deformable DETR / DINO 官方 config | `reduce_mean`（全域） | **`sync_cls_avg_factor=True`**（全域） |
| **StreamPETR 原版**（exiawsh） | `reduce_mean`（全域） | 本地（沒設 sync flag）—— AWML 繼承的不一致即源自此 |

→ **全域正規化是社群共識**；StreamPETR 原版的 cls 本地計數是它自己的疏漏，不是設計。

**實作**（全部進 working tree，330 個相關測試全過）:

- `task_modules/streaming.py` 新增 `reduce_mean_count`：跨卡平均計數，**無 DDP 時為恆等**
  （單卡行為逐位元不變，已驗證路徑不受影響）
- `heads/streampetr.py` matched 分支：cls 與 bbox 共用同一個全域計數
- `heads/streampetr.py` DN 分支：集合通訊**上提到 `loss()`**，掛在跨 rank 一致的旗標上
  （`with_dn and training`），**不掛在本 rank 有無 GT 上** —— 沒 GT 的 rank 以計數 0 參與，
  避免 mixed-GT step 死鎖
- `heads/focal2d.py`：五個 2D loss 共用同一個全域計數

**與 AWML 那次失敗改動（§6.3–6.4）的關鍵差別**：AWML 那次只改了 bbox 沒改 cls，
**打破了 cls:bbox 的組內比例**（選擇性 +7.7%）。這次三處都是 cls 與 bbox **共用同一個
因子** → 組內比例不變，只有組間（3D matched vs DN vs 2D aux）依各自被移除的偏差量
（約 8% / 中間 / 1.7%）平移。理論上溫和得多，但仍是 recipe 擾動。

**驗證**（`test_streampetr_loss_distributed.py`，gloo 2-rank 真集合通訊）:

1. `reduce_mean_count`：rank 計數 2 與 8 → 平均 5（§6.4 的算例），且不改輸入
2. **分片不變量**：同兩幀（2 框 vs 8 框）分到兩個 rank 的 DDP 平均 loss ==
   合成單一 batch 的 loss（`rtol=1e-5`）—— 這正是本地計數會違反、全域計數保證的性質
3. **DN 均勻性**：rank 0 有 GT、rank 1 無 GT，`loss()` 兩邊都完成不死鎖
   （join 帶 timeout，死鎖會變成測試失敗而非 pytest 卡死）

另外訓練 log 的 loss 絕對值會系統性下降（3D 約 −8%），與舊 run 比 loss 曲線時要記得這個位移。

#### 結果（2026-08-04 補測）：**這次改對了，分數也變好了**

retrain run `92068f7b…`（08-03 13:35 啟動、08-04 00:19 完成，10 epochs、2 GPU）。

**歸屬已用 mtime 驗證**，是乾淨的單變數對照：`transforms/camera/annotations2d.py`（2D GT 修正）mtime 07-31 01:40 → 兩個 run 都含；`task_modules/streaming.py` / `heads/streampetr.py` / `heads/focal2d.py`（本次正規化）mtime 08-03 13:20–13:23 → **只有本 run 含**。

> ⚠️ **基準必須是 §6.5 的 0.4104 / 0.3869，不是最初的 0.4230 / 0.3922** —— 後者是**帶 2D GT bug** 的 run。拿它當基準會把結論的正負號讀反（我第一次就讀錯了）。

| | §6.5（僅 2D GT 修正） | **本次（+ 全域正規化）** | Δ |
| --- | ---: | ---: | ---: |
| **val mAP** | 0.4104 | **0.4210** | **+0.0106** |
| **test mAP** | 0.3869 | **0.3900** | **+0.0031** |

兩個 split 同號。訓練期 callback 記到的 best val 是 0.42213（ep10，全程單調上升、無反轉），fresh run 得 0.4210，差 0.12 點屬正常重跑抖動。

#### 逐類：**barrier 一類解釋掉全部增益**

| class | val §6.5 | **val 本次** | Δ | test §6.5 | **test 本次** | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| car | 0.572 | 0.5658 | −0.006 | 0.578 | 0.5684 | −0.010 |
| truck | 0.449 | 0.4441 | −0.005 | 0.460 | 0.4665 | +0.007 |
| bus | 0.609 | 0.6254 | +0.016 | 0.578 | 0.5537 | −0.024 |
| bicycle | 0.446 | 0.4479 | +0.002 | 0.366 | 0.3660 | ±0.000 |
| pedestrian | 0.390 | 0.3821 | −0.008 | 0.381 | 0.3813 | ±0.000 |
| traffic_cone | 0.222 | 0.2161 | −0.006 | 0.197 | 0.2017 | +0.005 |
| **barrier** | 0.185 | **0.2653** | **+0.080** | 0.148 | **0.1925** | **+0.045** |
| **mAP** | **0.4104** | **0.4210** | **+0.0106** | **0.3869** | **0.3900** | **+0.0031** |

- **barrier 在兩個 split 都是最大變動、且同號**（+8.0 / +4.5）。其餘六類方向混雜、量級小 —— 是雜訊的形狀。
- barrier 一項對 val mAP 的貢獻是 +0.080/7 = **+0.0114**，已超過實測到的總增益 +0.0106 → 這次的 mAP 變化**完全由 barrier 驅動**。
- 值得注意的是：**barrier 正是 §6.5 那次 2D GT 修正掉最多的類別**（val −3.9 / test −4.4），而這次正規化修正把它拉回 +8.0 / +4.5。兩次改動在同一個類別上一來一回。

#### 判讀

1. **這次是「改對且分數也變好」**，打破了 §6.4/§6.5 建立的「改對不保證變好」模式。與那兩次的差別在於本次三處都是 **cls 與 bbox 共用同一因子**，組內比例不變（見上方對照）。
2. **統計強度仍然有限**：單一 seed，且 barrier 是稀有、遠距、集中於少數場景的類別（§8 記載 val 只有 10/137 個場景含 barrier），本身變異最大。**+1.06 val 落在 AWML 量到的 run 間散布（0.9 點）邊緣**，test 的 +0.31 遠在雜訊內。能確定的是：**沒有退步，且方向為正。**
3. **可推翻的假說**：先前一度懷疑「aml 的 bbox 有效權重比 AWML 高 8% 是它領先的原因」，並打算讓 AWML 調高 `loss_bbox_weight` 複製。本次移除了 aml 那 8% 偏差而 mAP **沒有下降**（反而微升），→ **有效權重比不是 aml 領先的來源**，該實驗取消。
4. **對外數字更新為 val 0.4210 / test 0.3900** —— 這是唯一同時不帶 2D GT bug、且正規化在數學上正確的一組。對 AWML 四 run 平均（val 0.3921 / test 0.3613）領先 **+2.9 / +2.9**；對 AWML 最佳單 run（val 0.3959 / test 0.3728）領先 **+2.5 / +1.7**。

---

## 7. 給 AWML 上游的回報清單

依實質性排序（已依 §3.3 的實測結果重排）：

1. **loss 正規化在多 GPU 下組內不一致** —— 同一 `loss_single` 內 `loss_cls`/`loss_iou` 用**本地**計數、bbox/centerness/centers2d 用**跨 GPU 平均**（§2.2）。此不一致源自 StreamPETR 原版（§7.1），非 AWML 引入。
   **回報時務必說明正確的修法方向是「兩者都改成全域」**（設 `sync_cls_avg_factor=True`，與 DETR 原版／mmdet 官方 config 一致），**不是**把 bbox 改成本地 —— 我們實測後者會掉 1.9 mAP（§6.3–6.4）。這也是探針結構上看不到的唯一一類差異（單卡下 `reduce_mean` 是恆等）
2. **DN 分支 z 慣例與 matched 分支不一致** —— DN 用底面中心、matched 用重心，10:1 列數比拉偏共用迴歸分支
3. **cross-attention 無條件降轉 fp16，設定的精度到不了它**（§2.1）—— 實質影響小（8.4%），但屬**正確性**問題：它會讓所有 precision ablation 靜默失效（已造成一次錯誤結論）
4. **flash_attn 反向非決定性且 torch 旗標無法偵測** —— 影響任何可重現性工作（§4.2）
5. **`FlashAttention.forward` 的 padded 分支有 UnboundLocalError** —— `if not fp16: output = output.float()` 在賦值前引用 `output`；fp32 + `key_padding_mask` 會直接崩。StreamPETR 傳 `None` 所以從未觸發（重構時順手修掉）
6. resize 插值核、1px flip、場景排序（次要）

**不需回報**：2D GT h/2 偏移 —— 那是 autoware-ml 的 bug，AWML 是正確的一方（§1.3，已於 2026-07-31 在 autoware-ml 側修正）。

### 7.1 問題歸屬：哪些是原版 StreamPETR 的、哪些是 AWML 移植時引入的（2026-08-03 查證）

對照 exiawsh/StreamPETR 原版原始碼逐項確認：

| 問題 | 原版 StreamPETR | 歸屬 |
| --- | --- | --- |
| **cls 本地 / bbox 全域不一致**（上面第 1 項） | **有** —— bbox 用 `reduce_mean`，cls 的 `sync_cls_avg_factor` 預設 False 且 config 從未設 True | 繼承鏈：DETR 原版做對（`num_boxes` all-reduce）→ mmdet 移植成 flag 預設 False → mmdet 自家 Deformable DETR / DINO config 都設 True → **PETR/StreamPETR 沒開** → AWML 繼承 |
| cross-attn 靜默降轉 fp16（第 3 項） | **沒有** —— 原版是 `assert q.dtype in [fp16, bf16]`，錯 dtype 大聲炸；bf16 誠實跑 bf16 | **AWML 引入**（`q.half()`）。原版的 assert 是更好的工程 —— 7/29 bf16 ablation 的污染在原版下不會發生 |
| padded 分支 `UnboundLocalError`（第 5 項） | **沒有** —— output 在條件前已賦值 | **AWML 引入** |
| DN 分支 z 慣例不一致（第 2 項） | **沒有** —— 原版 DN 用 `gravity_center` | **AWML 引入** |

→ 回報給 AWML 時：第 1 項可以連帶建議往上游 StreamPETR 回報（設 `sync_cls_avg_factor=True` 即可修）；第 2、3、5 項是 AWML 自己的 diff，上游無此問題。

---

## 8. 附錄：檔案清單

### autoware-ml（`/home/yihsiang/autoware-ml`）

新增工具（untracked）:

- `autoware_ml/tools/overfit_probe.py`
- `autoware_ml/tools/compare_overfit.py`
- `docs/framework/overfit-probe.md`

先前調查的修正（staged，未 commit）:

- `autoware_ml/transforms/geometry3d.py`、`transforms/camera/geometry.py`、`transforms/camera_lidar/geometry.py`（ego_pose 折疊）
- `autoware_ml/utils/lightning_loops.py`、`utils/runtime.py`（val loop）
- `autoware_ml/utils/checkpoints.py`（alias 去重）
- `docs/framework/streampetr-awml-parity.md`（完整調查紀錄）

本次修正（未 commit）:

- `autoware_ml/transforms/camera/annotations2d.py` + `tests/transforms/test_camera.py`（§1.3 的 2D GT h/2 bug，07-31 01:40；retrain 結果見 §6.5）
- `autoware_ml/models/detection3d/task_modules/streaming.py`（新增 `reduce_mean_count`）、
  `heads/streampetr.py`、`heads/focal2d.py`（**loss 正規化改為跨卡全域**，2026-08-03，見 §6.6）
- `autoware_ml/tests/models/test_streampetr_loss_distributed.py`（新增，gloo 2-rank 實測分片不變量與 DN 集合通訊均勻性）

### AWML（`/home/yihsiang/AWML`）

本次修改:

- `projects/StreamPETR/stream_petr/models/utils/attention.py`（dtype + determinism + fp32 路徑）
- `projects/StreamPETR/stream_petr/models/dense_heads/streampetr_head.py`（avg_factor ×2 處）
- `projects/StreamPETR/stream_petr/models/dense_heads/focal_head.py`（avg_factor）
- `projects/StreamPETR/stream_petr/datasets/pipelines/dataset.py`（GT hygiene）
- `projects/StreamPETR/stream_petr/datasets/pipelines/transform_3d.py`（cv2 resize）
- `projects/StreamPETR/configs/t4dataset/..._aligned_bf16.py`（新，untracked）
- `tools/detection3d/overfit_probe.py`（新，untracked）

**全部未 commit** —— retrain 跑的是 working tree 狀態。

---

## 9. 環境注意事項

- `awml_petr` 容器跑 **UTC**，host 跑 **JST (+9)**。mmengine 的 work_dir 時間戳是容器時間，比 host mtime 慢 9 小時 —— 判斷某次 run 是否吃到某個程式修改前**必須換算**（此坑曾導致一次誤判）
- 探針 trace 存於兩邊的 `parity_out/`（untracked）

---

## 10. Summary (EN)

**autoware-ml fixed**: ego-pose augmentation folding (the original ~5 mAP root cause), silently-skipped validation epochs, checkpoint alias deduplication — reaching 0.423 val / 0.392 test — and, found this round, a 2D auxiliary ground-truth bug that placed every target half an object height too high in the image (now agreeing with AWML to 0.0000 and pinned by a regression test). The retrain that bug demanded has since run (§6.5): as a clean single-variable comparison it gives **0.4104 val / 0.3869 test**, i.e. correcting the targets cost 1.3 and 0.5 mAP rather than gaining any, with barrier the only class whose drop is consistent across both splits (−3.9 / −4.4). The test-split delta sits well inside the measured run-to-run spread, so this is not established as a regression — but it is established as no improvement. The honest number to quote externally is now 0.4104 / 0.3869, still ahead of AWML's four-run mean by +1.8 / +2.6 rather than the earlier +3.1 / +3.1.

**AWML fixed**: the DN branch's z convention, cross-attention silently downcasting everything to fp16 (which invalidated the earlier bf16 ablation), nondeterministic FlashAttention backward, an UnboundLocalError in the padded attention branch, the resize interpolation kernel, and a 1-px flip miscalibration — plus a GT hygiene filter that provably drops nothing on this dataset. A multi-GPU loss-normalization inconsistency was also real, but the first attempt fixed it in the wrong direction and cost 1.9 mAP; it has been reverted to AWML's original global normalization pending a rerun in the correct direction (§6.4).

**And the normalization fix, done correctly on the autoware-ml side, paid off**: putting classification and regression on one global cross-rank count took val from 0.4104 to **0.4210** and test from 0.3869 to **0.3900** (§6.6). The entire gain is `barrier`, +8.0 val and +4.5 test — the same class the 2D ground-truth fix had cost the most. Two corrections, one class, one round trip.

**The hardest-won lesson**: "align AWML to autoware-ml" is not the same as "make it correct". Twice now autoware-ml turned out to be the wrong side — the 2D ground-truth offset and the loss normalization — so every alignment has to justify its own mathematics rather than borrowing authority from autoware-ml's higher mAP. The single-GPU probe is also structurally blind to multi-GPU normalization, which is exactly how the wrong-direction normalization change reached a 13-hour retrain unchallenged.

**What separated the change that worked from the one that did not**: AWML's failed attempt altered only the bbox term's averaging factor and left classification on the old one, which silently reweighted bbox against cls by ~8% and cost 1.9 mAP. autoware-ml's version put cls and bbox on the *same* global factor at all three sites, so within-group ratios were untouched and only the absolute scale moved — and it gained 1.1 val / 0.3 test. The rule that falls out: **when you change a normalization, change it for every term that shares a regression branch, or you have quietly changed the loss weights instead.**

Note also that the final quoted figures rest on a single seed per configuration, and the entire measured gain of the normalization fix comes from `barrier` — a rare, distant class present in only 10 of 137 validation scenes and therefore the noisiest one in the set. The direction is consistent across both splits, which is the same evidence standard used to take the earlier `barrier` regression seriously, but a second seed is what would turn "no regression, direction positive" into a measured improvement.

**What the probe could and could not show**: it proved forward and loss are equivalent at identical weights and inputs, and it located the 2D ground-truth bug. It also showed that cross-attention precision accounts for only 8.4% of the fitting gap, not the majority as first believed — that earlier reading was an artifact of running the probe in fp32. The two paths the probe had deliberately disabled were then switched back on one at a time (§5.4) and both proved aligned: temporal memory converges the two sides to −0.3%, and every inspectable DN internal is identical (query count, foreground/background split, averaging factor, label weights) with line-for-line identical noise generation. So **every mechanism the probe can reach is now aligned, and the probe carries no signal about the accuracy difference**; whatever remains lives in full-run training dynamics — multi-GPU normalization, the random behaviour of GridMask and augmentation, and the sampler's epoch-level sequence construction.

**A third measurement trap, recorded so it is not repeated**: the DN comparison first looked like a 43% implementation gap (82% on the classification term). It was not. Those were step-0 snapshots, which sit 7.7 and 12.3 standard deviations above their own tail means simply because step 0 has not yet trained on the batch. On tail averages the ratios fall to 1.07 and 1.21, within what differing RNG draws explain at a per-step coefficient of variation of 21%. Cross-framework comparison is only ever valid on tail averages — the same category of error as trusting a nondeterministic trace (§4.2).

**Where to debug next**: not by diffing implementations further — that avenue is exhausted, and it produced two wrong "fixes" along the way. Ask instead what the better model does better. Take the existing checkpoints, run them through one evaluator, and break the difference down by class, distance band, lidar-point count, and the TP error terms. That is inference-only, costs hours rather than a retrain, and it is how the original ego-pose root cause was actually found (the tell was AVE degraded 43–66% across every class while AOE and ASE stayed at parity). A behavioural signature is what makes a 13-hour retrain worth spending; without one, picking a mechanism is guessing.

**Still not aligned**: the attention kernel implementations themselves (only their dtype was aligned), and the evaluator offset of 3.4–3.8 mAP that makes raw cross-framework mAP comparison invalid.
