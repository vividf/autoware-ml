# Loss Design

> **What this covers:** where losses live, how they're computed and returned, the loss
> catalog, and the `"loss"` contract that ties them to the training loop.
> Prerequisites: [../model/head.md](../model/head.md), [training_loop.md](training_loop.md).

---

## 1. The one rule: losses live in the head, surface through `compute_metrics`

There is no global "loss registry" or loss-runner. The flow is deliberately simple:

```text
head owns the loss module(s)  →  model.compute_metrics() calls head.loss(...)  →  returns {"loss": ...}
   →  BaseModel._shared_step asserts "loss", logs it  →  training_step returns metrics["loss"]  →  Lightning backprops
```

Concretely (CenterPoint):

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

So to understand a model's loss, **read its head's `loss()` method** — that's the single
source of truth.

---

## 2. The `"loss"` contract

`compute_metrics` returns a dict that **must** contain a `"loss"` key (enforced at
`base.py:260`). It may contain any number of extra scalars, which are logged but not
back-propagated. CenterHead returns:

```python
return {"loss": total_loss, "loss_heatmap": loss_heatmap, "loss_bbox": loss_bbox}
```

`total_loss` is what Lightning optimizes; `loss_heatmap`/`loss_bbox` are logged as
`train/loss_heatmap`, `train/loss_bbox` for diagnostics. The weighting between components is a
plain Python combination inside `loss()`:

```python
total_loss = loss_heatmap + self.loss_bbox_weight * loss_bbox    # loss_bbox_weight is a head arg (config)
```

To reweight, change `loss_bbox_weight` in config → head constructor. Multi-task models sum the
per-task losses the same way (see the "Multiple Outputs" pattern in
`docs/contributing/adding-models.md`).

---

## 3. A loss up close: `GaussianFocalLoss` (`losses/detection3d/gaussian_focal.py:13`)

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

Two ideas visible here recur across the losses: **modulation** (`(1-p)^alpha` /
`p^alpha` focuses learning on hard examples) and **normalization by the positive count**
(`clamp_min(1)` avoids divide-by-zero when a frame has no objects). Losses are plain
`nn.Module`s taking `(prediction, target)` — nothing framework-specific.

---

## 4. The loss catalog (`autoware_ml/losses/`)

`__init__.py` files are empty; reference losses by their full module path in the head that
uses them.

| Task | Loss | File | Notes |
| ---- | ---- | ---- | ----- |
| detection3d | `GaussianFocalLoss` | `losses/detection3d/gaussian_focal.py` | CenterPoint heatmap loss |
| detection3d | `SigmoidFocalLoss` | `losses/detection3d/focal.py` | classification for transformer-style heads |
| detection2d | `QualityFocalLoss`, `GIoULoss`, `WeightedL1Loss`, `HeatmapGaussianFocalLoss` | `losses/detection2d/losses.py` | 2D detection |
| segmentation3d | `BoundaryLoss`, `LovaszLoss`, `LovaszSoftmaxLoss` | `losses/segmentation3d/{boundary,lovasz}.py` | IoU-surrogate + boundary losses |

Box regression frequently uses stock `torch.nn` losses (e.g. `nn.L1Loss(reduction="none")`,
masked and normalized inside the head) rather than a custom class.

---

## 5. Why this design (vs a loss registry)

- **Localized complexity.** Target generation, encoding, and loss must agree exactly (e.g.
  log-encoded dims, sin/cos yaw, positive-cell gathering). Keeping them in the same `loss()`
  method means one place to read and change — no cross-file coupling between a "loss module"
  and a "target assigner" wired by strings.
- **Deep module.** The model exposes a tiny interface (`compute_metrics → {"loss"}`) and hides
  all the target/encoding machinery behind it.
- **No hidden dependencies.** There's no framework step that post-processes losses; what the
  head returns is exactly what gets logged and optimized.

---

## 6. Distributed reduction

Losses are logged with `sync_dist=True` (in `_shared_step`'s train/val/test calls), so
Lightning averages the scalar across GPUs — correct because a mean-of-means equals the global
mean for equal batch sizes. (Metrics like mAP are *not* linear and are reduced differently by
torchmetrics — see [../evaluation/evaluation_pipeline.md](../evaluation/evaluation_pipeline.md).)

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `compute_metrics() must return a dict containing a 'loss' key` | head returned a bare tensor or wrong key | return `{"loss": ...}` |
| Loss `nan`/`inf` | `log()` of zero prob, log-encoding of non-positive box dims, fp16 overflow | clamp (as `GaussianFocalLoss` does); sanitize GT; try `bf16-mixed` |
| One component dominates | weight imbalance | tune `loss_bbox_weight` (or the per-component weights in the head) |
| Loss decreases but mAP flat | loss/target encoding mismatch, or metric decode differs from train targets | verify `get_targets` vs `predict` consistency in the head |
| Auxiliary loss ignored | not folded into `"loss"` | add it into the returned `"loss"` total |
| Loss not logged per-component | only returned `"loss"` | return extra keys (`loss_heatmap`, …) too |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Reweight loss terms | change the weight args in the head (`loss_bbox_weight`, …) via config |
| Swap a loss | construct a different loss `nn.Module` in the head's `__init__`; adjust `loss()` |
| Add an auxiliary loss | compute it in `head.loss` (or `compute_metrics`), add into `"loss"`, return it for logging |
| Add a new loss class | add an `nn.Module(prediction, target)` under `losses/<task>/`; use it from a head |
| Task-balanced multi-task loss | sum per-task losses in the model's `compute_metrics` with tunable weights |

---

**Next (Phase 5):** [../evaluation/evaluation_pipeline.md](../evaluation/evaluation_pipeline.md)
— how validation/test turn predictions into mAP/NDS/IoU.
