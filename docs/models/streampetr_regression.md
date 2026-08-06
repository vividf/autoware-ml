---
icon: lucide/target
---

# StreamPETR AWML Recipe Parity ("regression")

> Commit `98eec9e31f6b47898bccf96c0a155ce00201d86d` — *feat: streampetr regression*
> (vividf, 2026-07-23) · 27 files, +2758 / −24

## What this commit is about

The native `autoware-ml` StreamPETR (`StreamPETRDetectionModel`) already ran, but
it trained a **thinner recipe** than the production model in the old **AWML**
(`mmdetection3d`-based) repository. Retraining inside `autoware-ml` therefore did
**not reproduce** the AWML accuracy numbers.

This commit closes that gap. "Regression" here means *reproducing* AWML's
result — pulling the native training recipe back onto the AWML baseline so a
model retrained in `autoware-ml` lands on the same numbers as
`projects/StreamPETR/configs/default/vov_flash_480x640_baseline.py` in AWML.

To get there it ports the four training-only ingredients the native model was
missing, adds tooling to migrate/verify AWML checkpoints, and leaves the
inference/deployment path untouched.

**Inference and ONNX/TensorRT deployment are unchanged.** Everything added here
runs only under `self.training`, so the exported graph and runtime cost are
identical to before.

---

## The five changes, by concern

### 1. Auxiliary 2D detection head (Focal-PETR style) — *training only*

AWML shapes the image features with an extra 2D head that is dropped at
inference. This is now ported natively:

| File | Role |
| --- | --- |
| [autoware_ml/models/detection3d/heads/focal2d.py](../../autoware_ml/models/detection3d/heads/focal2d.py) | `FocalHead2D`: per-token class / centerness / LTRB box / projected-3D-center predictions on each camera's neck feature map. Five losses; **never run at inference**. |
| [autoware_ml/losses/detection2d/losses.py](../../autoware_ml/losses/detection2d/losses.py) | `QualityFocalLoss`, `GIoULoss`, `WeightedL1Loss`, `HeatmapGaussianFocalLoss`. |
| [autoware_ml/models/detection3d/task_modules/assigners2d.py](../../autoware_ml/models/detection3d/task_modules/assigners2d.py) | `HungarianAssigner2D` + `BBoxL1Cost2D` / `IoUCost2D` / `Center2DL1Cost`. |
| [autoware_ml/models/detection3d/task_modules/boxes2d.py](../../autoware_ml/models/detection3d/task_modules/boxes2d.py) | 2D box helpers (`cxcywh↔xyxy`, IoU/GIoU). |
| [autoware_ml/transforms/camera/annotations2d.py](../../autoware_ml/transforms/camera/annotations2d.py) | `LoadAnnotations2DFromBoxes3D`: projects the already-augmented 3D GT boxes onto every camera to produce 2D boxes / centers / depths / labels. |

In [streampetr.py](../../autoware_ml/models/detection3d/streampetr.py) the model gains an optional
`img_roi_head`; when present **and** training, its outputs and losses are added
to the total loss. Otherwise nothing changes.

### 2. `CPFPN` neck — weight-compatible with AWML checkpoints

[autoware_ml/models/common/necks/cp_fpn.py](../../autoware_ml/models/common/necks/cp_fpn.py) is a native port of the reference
StreamPETR `CPFPN` (plain 1×1 laterals, nearest-neighbor top-down, one 3×3
refine on the finest level). It mirrors the `mm` `ConvModule` parameter layout,
so AWML checkpoint weights load **name-for-name**. The existing
`GeneralizedLSSFPN` (concat + BN + ReLU) is *not* weight-compatible, which is why
a dedicated neck was needed for parity and checkpoint conversion.

### 3. Partial-ignore for `traffic_cone` / `barrier`

Some T4Dataset scenes are annotated for every class **except** `traffic_cone`
and `barrier`. Training on those frames must not punish background predictions
of the un-annotated classes as false positives.

| File | Role |
| --- | --- |
| [autoware_ml/models/detection3d/partial_ignore.py](../../autoware_ml/models/detection3d/partial_ignore.py) | New module: `resolve_partial_ignore_labels` (class-name → index) and `normalize_status_flags` (tensor/list/scalar → per-sample bool). |
| [autoware_ml/losses/detection3d/focal.py](../../autoware_ml/losses/detection3d/focal.py) | `SigmoidFocalLoss` now accepts **per-query-per-class** weights `(N, C)`, not just per-query `(N,)`, so individual class columns can be masked. |
| [heads/streampetr.py](../../autoware_ml/models/detection3d/heads/streampetr.py) | Zeroes the ignored class columns on **negative (background) queries** only — matched queries keep full supervision. Applied to both the main matching loss and the denoising (DN) queries noised into background. |
| [datamodule/common/multiview_detection3d.py](../../autoware_ml/datamodule/common/multiview_detection3d.py) | Emits the per-frame `traffic_cone_barrier_status` flag (missing → `True`). |

The flag flows datamodule → model → head. When it is absent or all-`True`, the
loss math is identical to before (no behavioral change for fully-annotated data).

### 4. Iteration-warmup + epoch-cosine LR schedule

[autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py](../../autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py) adds
`IterWarmupEpochCosineLR` (linear warmup over N *iterations*, then cosine decay
over *epochs*) to match AWML. The model now forwards a `scheduler_config`
(e.g. `interval: step`) to Lightning so the step-wise warmup ticks correctly.

### 5. Data-loader fidelity tweaks

[autoware_ml/transforms/camera/loading.py](../../autoware_ml/transforms/camera/loading.py) — `LoadMultiViewImagesFromFiles` gains:

- `shuffle_order`: per-sample camera-order shuffle (AWML's `shuffle_cameras=True`
  train-time regularization); every emitted per-camera array follows the shuffled
  order so the sample stays internally consistent.
- `color_type` (`rgb`/`bgr`): explicit channel order matched to the normalization
  stats.

Configs also flip `normalize_to_unit: false` so pixels stay in `[0, 255]`
(the `img_norm_cfg` mean/std are 0–255-scale ImageNet stats).

---

## Migration & verification tooling

| File | Role |
| --- | --- |
| [autoware_ml/tools/convert_streampetr_checkpoint.py](../../autoware_ml/tools/convert_streampetr_checkpoint.py) | Renames every parameter from the `mm` AWML layout (`Petr3D`/`StreamPETRHead`/`VoVNet`/`CPFPN`) into the native module names, optionally flips the stem conv for BGR→RGB, and can `--drop-pattern` the old class heads (so a 10-class nuScenes checkpoint feeds a 7-class T4 model, à la `strict=False`). Emits `{"state_dict": ...}` for `autoware-ml train --weights`. |
| [autoware_ml/tools/streampetr_parity_check.py](../../autoware_ml/tools/streampetr_parity_check.py) | Replays one real AWML-dumped frame (fp32, eval, no DN/GridMask/dropout) through the native model and compares image features, positional embeddings, per-layer head outputs, and every loss term against the reference. |
| `work_dirs/parity/streampetr_parity_reference.pt` | The 31 MB reference dump consumed by the parity checker (binary). |

---

## Configs

New / changed under [autoware_ml/configs/tasks/detection3d/streampetr/](../../autoware_ml/configs/tasks/detection3d/streampetr/):

| Config | Purpose |
| --- | --- |
| `_awml_parity.yaml` | Shared AWML-recipe knobs (composable, **not runnable alone**): pc_range ±51.2, full train augmentation, the 2D FocalHead, partial-ignore, the iter-warmup/epoch-cosine scheduler, seed 0, and mAP-based checkpoint selection. |
| `_reset_scheduler.yaml` | Sets `model.scheduler: null` so a later config can replace the scheduler node wholesale (OmegaConf merges dicts key-by-key; merging onto `null` replaces instead). |
| `vov_480x640_t4dataset_j6gen2_base.yaml` | Recipe-parity **base** training (35 epochs, bs 4, lr 5e-5). |
| `vov_480x640_t4dataset_j6gen2_finetune_cone_barrier.yaml` | Recipe-parity **fine-tune** with cone/barrier partial-ignore (40 epochs, bs 1, lr 6.25e-6). |
| `base.yaml` | Adds the new collation keys: `traffic_cone_barrier_status`, `gt_bboxes_2d`, `gt_labels_2d`, `centers_2d`, `depths_2d`. |
| `vov_320x800_nuscenes.yaml`, `vov_480x640_t4dataset_j6gen2.yaml` | Loader tweaks (`normalize_to_unit: false`); j6gen2 test set now points at `..._test.pkl` (was reusing val). |

> Note: `autoware-ml` has no `auto_scale_lr`. The config LRs are pre-scaled for
> the stated batch size — rescale by `total_batch_size / 8` for any other setup.

---

## How to use it

```bash
# 1. Convert an AWML checkpoint into native layout
python -m autoware_ml.tools.convert_streampetr_checkpoint \
    --input  work_dirs/streampetr_2_7/epoch_20.pth \
    --output streampetr_2_7_epoch_20_converted.pth \
    --bgr-to-rgb

# 2. (optional) confirm forward/loss parity against the AWML reference dump
python -m autoware_ml.tools.streampetr_parity_check \
    --reference  work_dirs/parity/streampetr_parity_reference.pt \
    --checkpoint streampetr_2_7_epoch_20_converted.pth

# 3a. Base training
autoware-ml train \
    --config-name tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_base \
    --weights nuscenes_vov99_baseline_320x800_converted.pth

# 3b. Fine-tune with cone/barrier partial-ignore
autoware-ml train \
    --config-name tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier \
    --weights streampetr_2_7_epoch_20_converted.pth
```

---

## Tests & build

- [autoware_ml/tests/models/test_streampetr_partial_ignore.py](../../autoware_ml/tests/models/test_streampetr_partial_ignore.py) — 13 unit
  tests covering partial-ignore label resolution, class-wise focal masking (main
  - DN queries), the 2D head forward/loss, 2D projection, CPFPN shapes, the LR
  schedule, checkpoint conversion (name mapping + drop patterns), and the camera
  shuffle consistency.
- [docker/Dockerfile](../../docker/Dockerfile) — unrelated build-robustness fix: `pixi`'s embedded
  `uv` has a fixed 30 s HTTP timeout, so large PyPI wheels time out on congested
  links. The `pixi install` step now caps download concurrency at 4 and retries
  up to 5× (the `uv` cache makes each retry progress).

---

## What did **not** change

- The inference forward path, the ONNX export, and the TensorRT engine build —
  the 2D head and partial-ignore logic are gated on `self.training`.
- Fully-annotated datasets: with no `traffic_cone_barrier_status` (or all-`True`),
  the loss is bit-for-bit the same as before this commit.
