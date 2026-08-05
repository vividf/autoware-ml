# Heads

> **What this covers:** the head — the most important sub-module to understand, because it
> **owns three things**: the prediction branches (`forward`), the training loss (`loss`), and
> the decode/NMS (`predict`). The model wrapper just delegates to it.
> Prerequisite: [model_architecture.md](model_architecture.md).

---

## 1. The head owns loss and decode

Recall the model wrapper (`CenterPointDetectionModel`):

```python
def compute_metrics(self, batch, outputs):
    return self.bbox_head.loss(outputs, batch["gt_boxes"], batch["gt_labels"])   # ← head.loss
def predict_outputs(self, batch, outputs):
    return self.bbox_head.predict(outputs)                                        # ← head.predict
def build_eval_output(self, batch, outputs):
    return detection_eval_output(self.bbox_head.predict(outputs), batch)          # ← head.predict again
```

So the model contributes almost no logic. **Target generation, loss, and box decoding all live
in the head.** This is deliberate: the model wrapper stays reusable/task-agnostic, and the
task-specific complexity is localized in one class.

---

## 2. `CenterHead` end to end (`heads/centerpoint.py:57`)

### Construction — branches + losses

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

`_build_head` is `ConvModule → 1×1 Conv`. The classification branch's bias is initialized to
`heatmap_init_bias = -2.19` (the focal-loss prior for rare positives).

### `forward` — dense maps

```python
def forward(self, x):                       # :145
    shared = self.shared_conv(x)
    outputs = {"heatmap": self.heatmap(shared), "reg": self.reg(shared),
               "height": self.height(shared), "dim": self.dim(shared), "rot": self.rot(shared)}
    if self.vel is not None:
        outputs["vel"] = self.vel(shared)
    return outputs                          # dict of dense (B,C,H,W) maps
```

### `get_targets` — encode GT into the dense grid (`:159`)

For each GT box, project its center to the BEV feature grid, splat a Gaussian into the class
heatmap (radius from `gaussian_radius`), and store the encoded regression target:

```python
encoded_box = [center_x_frac, center_y_frac, z, log(l), log(w), log(h), sin(yaw), cos(yaw)(, vx, vy)]
```

It returns a `CenterPointTargets` dataclass with `heatmap`, `anno_boxes`, `indices`, `mask`.
Note the box representation: dimensions are **log-encoded**, yaw is **sin/cos**, and only
positive cells (at `indices`, flagged by `mask`) contribute to the box loss.

### `loss` — combine heatmap + box (`:228`)

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

The returned dict flows straight up through `compute_metrics` → `_shared_step`, which logs
`train/loss`, `train/loss_heatmap`, `train/loss_bbox` and back-props `"loss"`.

### `predict` — decode dense maps into boxes (`:251`)

```python
def predict(self, outputs):
    heatmap = outputs["heatmap"].sigmoid()
    pooled  = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    heatmap = heatmap * (pooled == heatmap)     # peak-based NMS (keep local maxima)
    # ...per sample: topk peaks → threshold → decode (x,y,z,l,w,h,yaw[,vel]) → circle_nms per class...
    # → list of {bboxes_3d, scores_3d, labels_3d}
```

`predict` is used both at inference (`predict_outputs`) and for evaluation (`build_eval_output`
→ `detection_eval_output` pairs decoded boxes with GT for the metric suites).

---

## 3. The head family (`models/detection3d/heads/`)

| Head | File | Style | Loss / matching |
| ---- | ---- | ----- | --------------- |
| `CenterHead` | `heads/centerpoint.py` | dense, anchor-free (heatmap) | GaussianFocalLoss + masked L1; peak + circle NMS |
| `TransFusionHead` | `heads/transfusion.py` | query-based (transformer decoder) | Hungarian assignment (`HungarianAssigner3D`), bbox coder, focal + regression |
| `StreamPETRHead` | `heads/streampetr.py` | query-based, temporal/streaming (camera) | set-prediction, temporal query propagation |
| `FocalHead2D` | `heads/focal2d.py` | 2D dense | focal-style 2D detection |

The query-based heads (`TransFusionHead`, `StreamPETRHead`) use `task_modules/` — assigners
(`HungarianAssigner3D/2D`), bbox coders (`TransFusionBBoxCoder`, `NMSFreeBBoxCoder3D`), and
match costs. Classification heads live in `models/common/heads/` (`LinearClsHead`).

Regardless of style, the **contract is the same**: the head exposes `forward` (predict),
`loss` (train), and `predict`/decode, and the model wrapper delegates to them.

---

## 4. Where losses come from

Losses are `nn.Module`s in `autoware_ml/losses/`, **constructed and owned inside the head**
(not in the model). CenterHead builds `GaussianFocalLoss()` and `nn.L1Loss(reduction="none")`
in `__init__`. Detailed loss catalog: [../training/loss_design.md](../training/loss_design.md).

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `in_channels` mismatch at head | head `in_channels` ≠ neck output channels | set head `in_channels` = sum of neck `out_channels` (e.g. 384) |
| Heatmap loss dominates / boxes never learn | `loss_bbox_weight` too low, or all boxes out of range | check `loss_bbox_weight`, `point_cloud_range`, GT filtering |
| No detections at eval | `score_threshold` too high, or `out_size_factor` wrong | lower threshold; verify grid math (`voxel_size`, `out_size_factor`) |
| NaN loss | log-encoded dims with zero/negative sizes, or bad GT | sanitize GT boxes; check `dim` targets |
| Class indices out of range | `num_classes` ≠ dataset class count | wire `num_classes: ${dataset.detection3d.num_classes}` |
| Duplicated/merged detections | circle NMS `nms_min_radius` too large/small | tune `nms_min_radius` |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Add a prediction branch (e.g. IoU) | add a `_build_head` branch in the head's `__init__`, include it in `forward`/`loss`/`predict` |
| Change the loss | construct a different loss in the head's `__init__`; adjust `loss()` |
| Reweight losses | tune `loss_bbox_weight` (config → head arg) |
| Add a class | update the dataset group's `class_names`/`num_classes`; head reads them via config |
| Switch to a query-based head | change `bbox_head._target_` to `TransFusionHead`; wire its `task_modules` |

---

**Next (Phase 4):** [../training/training_loop.md](../training/training_loop.md) — how the loss
this head returns becomes a weight update.
