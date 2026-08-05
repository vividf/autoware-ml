# 損失設計 (Loss Design)

> **本文涵蓋內容：** 損失（loss）存放於何處、如何被計算與回傳、損失目錄（catalog），以及將其與訓練迴圈連結起來的 `"loss"` 合約（contract）。
> 先備知識：[../model/head.md](../model/head.md)、[training_loop.md](training_loop.md)。

---

## 1. 唯一的規則：損失存放於 head 中，透過 `compute_metrics` 顯露出來

沒有全域的「loss registry」或 loss-runner。整個流程刻意設計得很簡單：

```text
head owns the loss module(s)  →  model.compute_metrics() calls head.loss(...)  →  returns {"loss": ...}
   →  BaseModel._shared_step asserts "loss", logs it  →  training_step returns metrics["loss"]  →  Lightning backprops
```

具體來說（以 CenterPoint 為例）：

```python
# model wrapper
def compute_metrics(self, batch, outputs):
    return self.bbox_head.loss(outputs, batch["gt_boxes"], batch["gt_labels"])
```

```python
# head owns the loss objects (heads/centerpoint.py:130)
self.loss_heatmap = GaussianFocalLoss()
self.loss_bbox    = nn.L1Loss(reduction="none")
```

因此，若要理解某個模型的損失，**請閱讀其 head 的 `loss()` method** — 這是唯一的真實來源（source of truth）。

---

## 2. `"loss"` 合約

`compute_metrics` 回傳一個 dict，其中**必須**包含 `"loss"` 這個 key（此規則在 `base.py:260` 強制執行）。它可以包含任意數量的額外純量（scalar），這些純量會被記錄（logged）但不會被反向傳播（back-propagated）。CenterHead 回傳：

```python
return {"loss": total_loss, "loss_heatmap": loss_heatmap, "loss_bbox": loss_bbox}
```

`total_loss` 是 Lightning 實際優化的對象；`loss_heatmap`/`loss_bbox` 則被記錄為 `train/loss_heatmap`、`train/loss_bbox`，供診斷之用。各組成部分之間的權重配置只是 `loss()` 內部單純的 Python 運算組合：

```python
total_loss = loss_heatmap + self.loss_bbox_weight * loss_bbox    # loss_bbox_weight is a head arg (config)
```

若要調整權重，只需在 config 中變更 `loss_bbox_weight` → 傳入 head 建構子（constructor）。多工（multi-task）模型也是以相同方式加總各任務的損失（詳見 `docs/contributing/adding-models.md` 中的「Multiple Outputs」模式）。

---

## 3. 近距離觀察一個損失：`GaussianFocalLoss` (`losses/detection3d/gaussian_focal.py:13`)

```python
class GaussianFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        self.alpha, self.beta = alpha, beta

    def forward(self, prediction, target):
        prediction = prediction.sigmoid().clamp(min=1e-4, max=1-1e-4)
        pos_mask    = target.eq(1).float()          # exact peaks are positives
        neg_mask    = target.lt(1).float()
        neg_weights = (1 - target).pow(self.beta)   # negatives near a peak are down-weighted

        pos_loss = -torch.log(prediction)     * (1 - prediction).pow(self.alpha) * pos_mask
        neg_loss = -torch.log(1 - prediction) * prediction.pow(self.alpha) * neg_weights * neg_mask
        return (pos_loss.sum() + neg_loss.sum()) / pos_mask.sum().clamp_min(1)   # normalize by #positives
```

這裡呈現的兩個概念會反覆出現在各種損失中：**調變（modulation）**（`(1-p)^alpha` /
`p^alpha` 讓學習聚焦在困難樣本上）以及**依正樣本數量做正規化（normalization）**
（`clamp_min(1)` 可避免當一個 frame 沒有物件時發生除以零的情況）。損失就是單純接收
`(prediction, target)` 的 `nn.Module`，沒有任何框架特定的內容。

---

## 4. 損失目錄 (`autoware_ml/losses/`)

`__init__.py` 檔案是空的；請在使用它們的 head 中，以完整的 module path 來引用這些損失。

| 任務 | 損失 | 檔案 | 備註 |
| ---- | ---- | ---- | ----- |
| detection3d | `GaussianFocalLoss` | `losses/detection3d/gaussian_focal.py` | CenterPoint heatmap 損失 |
| detection3d | `SigmoidFocalLoss` | `losses/detection3d/focal.py` | 用於 transformer 風格 heads 的分類（classification） |
| detection2d | `QualityFocalLoss`, `GIoULoss`, `WeightedL1Loss`, `HeatmapGaussianFocalLoss` | `losses/detection2d/losses.py` | 2D detection |
| segmentation3d | `BoundaryLoss`, `LovaszLoss`, `LovaszSoftmaxLoss` | `losses/segmentation3d/{boundary,lovasz}.py` | IoU 替代（surrogate）+ boundary 損失 |

Box regression 經常直接使用現成（stock）的 `torch.nn` 損失（例如 `nn.L1Loss(reduction="none")`，
在 head 內部進行遮罩（mask）與正規化），而非自訂類別。

---

## 5. 為何採用此設計（相較於 loss registry）

- **局部化的複雜度（Localized complexity）。** 目標生成（target generation）、編碼（encoding）與損失三者必須完全一致（例如以 log 編碼的維度、sin/cos 表示的 yaw、正樣本格點（positive-cell）的蒐集方式）。將它們保留在同一個 `loss()` method 中，代表只需在一處閱讀與修改 — 不會有「損失模組」與由字串連接的「target assigner」之間的跨檔案耦合。
- **深層模組（Deep module）。** 模型只對外暴露一個極小的介面（`compute_metrics → {"loss"}`），並將所有 target/編碼相關機制隱藏在其後。
- **沒有隱藏的依賴關係。** 框架中沒有任何步驟會對損失做後處理；head 回傳的內容，就是最終被記錄與優化的內容。

---

## 6. 分散式歸約（Distributed reduction）

損失是以 `sync_dist=True` 記錄的（在 `_shared_step` 的 train/val/test 呼叫中），因此 Lightning
會跨 GPU 平均這個純量 — 這是正確的做法，因為在 batch size 相同的情況下，「平均值的平均」等於
「全域平均值」。（像 mAP 這類指標*並非*線性的，torchmetrics 對它們的歸約方式不同 — 詳見
[../evaluation/evaluation_pipeline.md](../evaluation/evaluation_pipeline.md)。）

---

## 常見除錯情境

| 症狀 | 原因 | 修正方式 |
| ------- | ----- | --- |
| `compute_metrics() must return a dict containing a 'loss' key` | head 回傳了裸的（bare）tensor 或錯誤的 key | 回傳 `{"loss": ...}` |
| Loss 為 `nan`/`inf` | 對機率為零的值取 `log()`、對非正值的 box 維度做 log 編碼、fp16 溢位（overflow） | 進行 clamp（如 `GaussianFocalLoss` 的做法）；淨化（sanitize）GT；嘗試 `bf16-mixed` |
| 某一個組成項佔主導地位 | 權重不平衡 | 調整 `loss_bbox_weight`（或 head 中各組成項的權重） |
| Loss 下降但 mAP 沒有變化 | loss/target 編碼不一致，或指標解碼（decode）方式與訓練目標不同 | 確認 head 中 `get_targets` 與 `predict` 的一致性 |
| 輔助損失（Auxiliary loss）被忽略 | 未被納入 `"loss"` 中 | 將其加進回傳的 `"loss"` 總和裡 |
| 各組成項的損失沒有被分別記錄 | 只回傳了 `"loss"` | 同時回傳額外的 key（`loss_heatmap` 等） |

---

## 常見修改情境

| 我想要… | 這麼做 |
| ---------- | ------- |
| 重新調整損失項的權重 | 透過 config 變更 head 中的權重參數（`loss_bbox_weight` 等） |
| 替換損失 | 在 head 的 `__init__` 中建構不同的損失 `nn.Module`；調整 `loss()` |
| 新增輔助損失 | 在 `head.loss`（或 `compute_metrics`）中計算，將其加進 `"loss"`，並回傳以供記錄 |
| 新增一個損失類別 | 在 `losses/<task>/` 底下新增一個 `nn.Module(prediction, target)`；並在 head 中使用它 |
| 任務間權重平衡的多工損失 | 在模型的 `compute_metrics` 中，以可調整的權重加總各任務的損失 |

---

**下一步（第 5 階段）：** [../evaluation/evaluation_pipeline.md](../evaluation/evaluation_pipeline.md)
— validation/test 如何將預測結果轉換為 mAP/NDS/IoU。
