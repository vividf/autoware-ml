# StreamPETR 影像 Resize:PIL vs cv2 — 完整背景與決策

> 目的:回答「為什麼原版 StreamPETR 用 PIL、兩者差在哪、我們該統一用哪個」。
> 結論先講:**統一 cv2(INTER_LINEAR)**;node 的 CUDA kernel 加 antialias 開關做世代過渡。

---

## 1. 為什麼原版 StreamPETR 用 PIL?— 是血統,不是選擇

原版 StreamPETR 的訓練 pipeline 其實是**混血**:

```
mmcv LoadMultiViewImageFromFiles(cv2 讀圖,numpy/BGR)
        ↓
ResizeCropFlipRotImage(把 numpy 轉成 PIL:Image.fromarray → PIL resize/crop/flip/rotate)
        ↓
回到 numpy,mm 後續 transform(cv2 慣例)
```

`ResizeCropFlipRotImage` 這段 PIL 程式碼不是 StreamPETR 作者為了數值品質挑的,而是**逐層繼承**來的:

| 世代 | 專案 | 影像慣例 |
|---|---|---|
| 祖 | **Lift-Splat-Shoot**(NVIDIA, 2020) | torchvision/PIL(`Image.open`、PIL resize/crop/rotate + ida matrix 記帳) |
| 父 | **BEVDet / BEVDepth** | 直接沿用 LSS 的 `img_transform`(PIL) |
| 子 | **PETR / StreamPETR** | 從 BEVDet 抄 `ResizeCropFlipRotImage`,塞進 mm(cv2)pipeline 裡 |

證據就在程式碼:AWML `transform_3d.py` 的 `Image.fromarray(np.uint8(imgs[i]))` — 圖已經是 cv2 讀的 numpy,到增強這步才轉 PIL,做完再轉回來。ida_mat 的翻轉/裁切數學也全是 LSS 的 PIL 座標慣例(例如 `FLIP_LEFT_RIGHT` 對應 `x → width−1−x`)。

換句話說:**mm 生態全線是 cv2,PIL 只是 LSS 血統的一段飛地**。原作者從未在論文或 repo 裡論證過 PIL resize 的必要性。

## 2. 兩者的實際差異在哪?

resize 之外的操作兩派等價(crop 是無損切片;flip 無損,但 PIL 慣例的平移項要用 `width−1`,AWML 已修;rotate 在這些 recipe 裡是 0)。**唯一有數值意義的分歧是 resize 的濾波行為**:

| | PIL `Image.resize(BILINEAR)` | cv2 `INTER_LINEAR` |
|---|---|---|
| 濾波核 | Triangle filter,**support 隨縮小倍率放大**(卷積重採樣) | 固定 2×2 鄰域取樣 |
| 縮小時 | **有抗鋸齒**:所有貢獻像素都參與,輸出平滑 | **無抗鋸齒**:高頻 aliasing 殘留,輸出較銳/較躁 |
| 放大時 | 等同標準雙線性 | 等同標準雙線性(兩者一致) |
| 座標映射 | half-pixel(`(x+0.5)·s−0.5`) | half-pixel(相同) |

**本專案實測量級**:T4 相機 2880×1860 → 743×480(~0.26×)時,兩者輸出像素 std 差 **~4%**。
train/deploy 錯配的代價在 same-weights 跨 stack 實驗量到 **~−0.8 mAP**(方向一致跨全部 7 類)。

## 3. 優缺點總表

### PIL(抗鋸齒 resize)

| 優點 | 缺點 |
|---|---|
| 縮小時訊號較乾淨(理論上對訓練略有利,無公開證據顯示對 detection mAP 有實質差異) | **部署地獄**:GPU/embedded 生態沒有現成等價物 — node 現在那顆 100+ 行的 CUDA kernel 就是為了模仿 PIL 手寫的,而且是 O(scale²) taps 的慢路徑 |
| 與 AWML model-zoo 舊 checkpoint 的訓練分佈一致(node 現行 kernel 服務的就是它們) | 比 cv2 慢(dataloader 熱路徑;Pillow-SIMD 才有 SIMD) |
| RGB 原生(不會有 BGR 陷阱) | Python 生態限定;要在 mm/cv2 pipeline 裡來回轉格式 |
| | 訓練框架(autoware-ml + aligned AWML)已經**不是**它了 — 選 PIL 等於重做 parity |

### cv2(INTER_LINEAR)

| 優點 | 缺點 |
|---|---|
| **就是 GPU kernel 的自然形態**:固定 4-tap 雙線性 = npp / TensorRT resize / torchvision `antialias=False` / DALI 的預設語義,deploy 零特製 | 縮小無抗鋸齒,輸出含 aliasing(對「一致訓練」的模型無實害,模型學的就是這個分佈) |
| 快(SIMD、固定 taps);node kernel 從 ~8×8 taps 降到 2×2,**preprocess 直接變快** | 與 model-zoo **舊** checkpoint 的訓練分佈不一致(部署舊模型時 kernel 不能切過來) |
| autoware-ml 原生 pipeline、aligned AWML、已驗證的 parity 數字(0.391/0.366)全部錨定在 cv2 | cv2 讀圖預設 BGR(本 repo 已在 load 端轉 RGB,風險已封) |
| 訓練↔部署同一套語義,skew 結構性消失 | |

### 順帶排除的第三選項

「cv2 INTER_AREA / torchvision `antialias=True`」≈ 抗鋸齒但非 PIL 精確等價 — 這是**第三種語義**,
選它 = 全鏈重訓 + kernel 重寫,又沒有證據 mAP 更好。不考慮。

## 4. 決策:統一 cv2(INTER_LINEAR)

理由排序:

1. **一致性壓倒濾波品質**:實測告訴我們,mAP 的敵人不是「有沒有抗鋸齒」,是「訓練與部署不一致」(~4% std → ~0.8 mAP)。兩派各自「內部一致」時都能訓出好模型;所以決策準則是哪一派的一致性最便宜。
2. cv2 的一致性成本最低:訓練端**已經**全是 cv2(autoware-ml 原生 + AWML aligned patch,commit `96ae7728` 註解明言 "autoware-ml is the parity reference");部署端只要把 kernel 的 adaptive support 釘成 1(`preprocess_kernel.cu` L81-82 一行)就與 cv2 等價,還變快。
3. PIL 的一致性成本最高:要改回 PIL,等於放棄已驗證的 parity 錨點、重訓/重驗所有 cv2 配方,只為了保住一顆終將退役的 kernel。

### 執行事項

- [ ] **node kernel 加開關**:`resizeAndExtractRoi_kernel` 的 support 計算改為
      `antialias ? max(scale, 1.0) : 1.0`,由 ROS param / model metadata 控制。
      - 舊 model-zoo checkpoint(PIL 訓)→ `antialias=true`(現行為,保持正確)
      - cv2 訓的新模型(parity 那顆 0.391、畢業路線產出)→ `antialias=false`
- [ ] **golden-tensor 測試進 CI**:同一張圖走 autoware-ml val pipeline vs node kernel,
      對齊後 preprocess tensor 差異應在 uint8/浮點噪音級(~1e-3 像素單位);
      錯配時會出現 ~4% std 系統差 — 這個測試同時是開關接對沒有的守門員。
- [ ] model-zoo PIL 世代 checkpoint 全數退役後,刪除 antialias 分支,kernel 回歸單純雙線性。

## 5. 相關檔案索引

| 檔案 | 角色 |
|---|---|
| `AWML/projects/StreamPETR/stream_petr/datasets/pipelines/transform_3d.py` | 原版 PIL 增強(L218 `Image.fromarray`);resize 已被 patch 成 cv2(L341-348,commit `96ae7728`);flip 的 width−1 修正(L359-363) |
| `autoware-ml/autoware_ml/transforms/camera/resize.py` | native cv2 全套實作(cv2.resize / numpy crop / warpAffine) |
| `autoware_universe/perception/autoware_camera_streampetr/lib/network/preprocess_kernel.cu` | node 的 PIL-mimic CUDA kernel(L43-45 註解、L81-82 adaptive support)← 開關要加在這 |
| `autoware-ml/STREAMPETR_NATIVE_PRETRAIN_PLAN.md` | 畢業計畫(native cv2 全鏈的脈絡) |
