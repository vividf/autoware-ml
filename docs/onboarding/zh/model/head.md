# Head

> **本文涵蓋內容：** head — 最重要、最需要理解的子模組，因為它**擁有三件事**：
> 預測分支（`forward`）、訓練損失（`loss`），以及解碼／NMS（`predict`）。模型
> 包裝器（wrapper）只是把工作委派給它。
> 先備知識：[model_architecture.md](model_architecture.md)。

---

## 1. Head 擁有 loss 與 decode

回想一下模型的包裝器（`CenterPointDetectionModel`）：

```python
def compute_metrics(self, batch, outputs):
    return self.bbox_head.loss(outputs, batch["gt_boxes"], batch["gt_labels"])   # ← head.loss
def predict_outputs(self, batch, outputs):
    return self.bbox_head.predict(outputs)                                        # ← head.predict
def build_eval_output(self, batch, outputs):
    return detection_eval_output(self.bbox_head.predict(outputs), batch)          # ← head.predict again
```

所以模型本身幾乎不含任何邏輯。**目標生成（target generation）、loss 以及框（box）解碼
全部都在 head 裡。** 這是刻意的設計：模型包裝器保持可重用、與任務無關（task-agnostic），
而任務特定的複雜度則被局限（localized）在單一類別中。

---

## 2. `CenterHead` 從頭到尾（`heads/centerpoint.py:57`）

### Construction — 分支與 loss

```python
class CenterHead(nn.Module):
    def __init__(self, in_channels, num_classes, shared_channels, point_cloud_range, voxel_size,
                 out_size_factor, ..., use_velocity=True):
        self.shared_conv = ConvModule(in_channels, shared_channels)      # shared tower
        self.heatmap = self._build_head(shared_channels, num_classes, init_bias=heatmap_init_bias)
        self.reg     = self._build_head(shared_channels, 2)   # center offset (x,y)
        self.height  = self._build_head(shared_channels, 1)   # z
        self.dim     = self._build_head(shared_channels, 3)   # l,w,h (log-encoded)
        self.rot     = self._build_head(shared_channels, 2)   # sin,cos(yaw)
        self.vel     = self._build_head(shared_channels, 2) if use_velocity else None
        self.loss_heatmap = GaussianFocalLoss()               # the head OWNS its losses
        self.loss_bbox    = nn.L1Loss(reduction="none")
```

`_build_head` 就是 `ConvModule → 1×1 Conv`。分類分支的 bias 被初始化為
`heatmap_init_bias = -2.19`（這是 focal loss 針對稀少正樣本設計的先驗值）。

### `forward` — 密集特徵圖

```python
def forward(self, x):                       # :145
    shared = self.shared_conv(x)
    outputs = {"heatmap": self.heatmap(shared), "reg": self.reg(shared),
               "height": self.height(shared), "dim": self.dim(shared), "rot": self.rot(shared)}
    if self.vel is not None:
        outputs["vel"] = self.vel(shared)
    return outputs                          # dict of dense (B,C,H,W) maps
```

### `get_targets` — 將 GT 編碼進密集網格中（`:159`）

對每一個 GT box，將其中心投影到 BEV 特徵網格上，在類別 heatmap 中潑灑（splat）一個
高斯分佈（半徑來自 `gaussian_radius`），並儲存編碼後的迴歸（regression）目標：

```python
encoded_box = [center_x_frac, center_y_frac, z, log(l), log(w), log(h), sin(yaw), cos(yaw)(, vx, vy)]
```

它會回傳一個 `CenterPointTargets` dataclass，內含 `heatmap`、`anno_boxes`、`indices`、
`mask`。注意這裡的 box 表示方式：尺寸（dimensions）是**以 log 編碼**的，yaw 則是
**sin/cos**，且只有正樣本格（在 `indices` 位置，由 `mask` 標記）會對 box loss 有貢獻。

### `loss` — 合併 heatmap 與 box（`:228`）

```python
def loss(self, outputs, gt_boxes, gt_labels):
    targets = self.get_targets(gt_boxes, gt_labels, outputs["heatmap"].shape[-2:], outputs["heatmap"].device)
    loss_heatmap = self.loss_heatmap(outputs["heatmap"], targets.heatmap)     # GaussianFocalLoss

    pred_boxes = torch.cat([outputs["reg"], outputs["height"], outputs["dim"], outputs["rot"]
                            (+ [outputs["vel"]])], dim=1)
    pred_boxes = _transpose_and_gather_feat(pred_boxes, targets.indices)      # gather at positive cells
    bbox_mask  = targets.mask.unsqueeze(-1).expand_as(targets.anno_boxes).float()
    loss_bbox  = (self.loss_bbox(pred_boxes, targets.anno_boxes) * bbox_mask).sum() / bbox_mask.sum().clamp_min(1.0)

    total = loss_heatmap + self.loss_bbox_weight * loss_bbox
    return {"loss": total, "loss_heatmap": loss_heatmap, "loss_bbox": loss_bbox}
```

回傳的 dict 會直接一路往上流經 `compute_metrics` → `_shared_step`，後者會記錄
`train/loss`、`train/loss_heatmap`、`train/loss_bbox`，並對 `"loss"` 做反向傳播（back-prop）。

### `predict` — 將密集特徵圖解碼成框（box）（`:251`）

```python
def predict(self, outputs):
    heatmap = outputs["heatmap"].sigmoid()
    pooled  = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    heatmap = heatmap * (pooled == heatmap)     # peak-based NMS (keep local maxima)
    # ...per sample: topk peaks → threshold → decode (x,y,z,l,w,h,yaw[,vel]) → circle_nms per class...
    # → list of {bboxes_3d, scores_3d, labels_3d}
```

`predict` 在推論（inference，`predict_outputs`）以及評估（evaluation，`build_eval_output`
→ `detection_eval_output` 將解碼後的框與 GT 配對供 metric suite 使用）兩種場合都會用到。

---

## 3. Head 家族（`models/detection3d/heads/`）

| Head | 檔案 | 風格 | Loss／matching |
| ---- | ---- | ----- | --------------- |
| `CenterHead` | `heads/centerpoint.py` | dense, anchor-free (heatmap) | GaussianFocalLoss + masked L1; peak + circle NMS |
| `TransFusionHead` | `heads/transfusion.py` | query-based (transformer decoder) | Hungarian assignment (`HungarianAssigner3D`), bbox coder, focal + regression |
| `StreamPETRHead` | `heads/streampetr.py` | query-based, temporal/streaming (camera) | set-prediction, temporal query propagation |
| `FocalHead2D` | `heads/focal2d.py` | 2D dense | focal-style 2D detection |

以 query 為基礎的 head（`TransFusionHead`、`StreamPETRHead`）使用 `task_modules/` —
assigner（`HungarianAssigner3D/2D`）、bbox coder（`TransFusionBBoxCoder`、
`NMSFreeBBoxCoder3D`）以及匹配代價（match cost）。分類 head 則位於
`models/common/heads/`（`LinearClsHead`）。

無論風格為何，**約定（contract）都是一樣的**：head 暴露 `forward`（預測）、
`loss`（訓練），以及 `predict`/解碼，而模型包裝器則將工作委派給它們。

---

## 4. Loss 從何而來

Loss 是 `autoware_ml/losses/` 中的 `nn.Module`，**在 head 內部建構並擁有**
（而不是在模型內）。CenterHead 在 `__init__` 中建構了 `GaussianFocalLoss()` 和
`nn.L1Loss(reduction="none")`。完整的 loss 目錄請參見：
[../training/loss_design.md](../training/loss_design.md)。

---

## Common debugging cases

| 症狀 | 原因 | 修正 |
| ------- | ----- | --- |
| head 處的 `in_channels` 不匹配 | head 的 `in_channels` ≠ neck 的輸出 channel 數 | 設定 head 的 `in_channels` = neck 各 `out_channels` 之總和（例如 384） |
| heatmap loss 主導、box 學不起來 | `loss_bbox_weight` 太低，或所有 box 都超出範圍 | 檢查 `loss_bbox_weight`、`point_cloud_range`、GT 過濾 |
| 評估時沒有偵測結果 | `score_threshold` 太高，或 `out_size_factor` 錯誤 | 降低 threshold；驗證網格計算（`voxel_size`、`out_size_factor`） |
| NaN loss | log 編碼的尺寸為零或負值，或 GT 有問題 | 清理 GT box；檢查 `dim` 目標值 |
| 類別索引超出範圍 | `num_classes` ≠ dataset 的類別數量 | 接上 `num_classes: ${dataset.detection3d.num_classes}` |
| 偵測結果重複/合併 | circle NMS 的 `nms_min_radius` 太大或太小 | 調整 `nms_min_radius` |

---

## Common modification scenarios

| 我想要… | 這樣做 |
| ---------- | ------- |
| 新增一個預測分支（例如 IoU） | 在 head 的 `__init__` 中新增一個 `_build_head` 分支，並將它納入 `forward`/`loss`/`predict` |
| 更改 loss | 在 head 的 `__init__` 中建構不同的 loss；調整 `loss()` |
| 重新調整各 loss 的權重 | 調整 `loss_bbox_weight`（config → head 參數） |
| 新增一個類別 | 更新 dataset group 的 `class_names`/`num_classes`；head 會透過 config 讀取它們 |
| 換成以 query 為基礎的 head | 將 `bbox_head._target_` 改為 `TransFusionHead`；接上其 `task_modules` |

---

**Next (Phase 4):** [../training/training_loop.md](../training/training_loop.md) — 這個 head
回傳的 loss 如何變成一次權重更新。
