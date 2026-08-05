# Augmentation & Transforms

> **What this covers:** the `BaseTransform` contract, `TransformsCompose`, the transform
> library layout, and how to read/write a transform. Transforms are the CPU, per-sample stage
> that **loads files and augments data**.
>
> Prerequisite: [dataset_pipeline.md](dataset_pipeline.md).

---

## 1. Why transforms exist (and why loading lives here)

A `Dataset` only returns *metadata* (paths, raw annotations). Everything else — reading the
LiDAR file, stacking sweeps, converting annotations to boxes, flipping/rotating/scaling,
cropping to range — is done by **transforms**. Two reasons:

1. **Composability.** A pipeline is an ordered list you assemble in config. Swap augmentations
   per experiment without touching Python.
2. **Split-specific behavior.** Train gets random augmentation; val/test/predict get only
   deterministic loading + cropping. Same dataset, different pipelines.

Transforms run **on CPU, in the DataLoader worker processes, per sample**. Heavy GPU,
per-batch work (voxelization) is *not* a transform — it's model-owned `DataPreprocessing`
(see [dataset_pipeline.md](dataset_pipeline.md#8-handoff-to-the-model-gpu-preprocessing)).

---

## 2. The contract: dict-in / dict-out (`transforms/base.py:28`)

Every transform is a `BaseTransform`. It reads some keys from the sample dict and returns
**only the keys it changed**; the composer merges them back.

```python
class BaseTransform(ABC):
    p: float | None = None          # apply probability (None = always)
    _required_keys: Sequence[str] = ()   # KeyError if any missing
    _optional_keys: Sequence[str] = ()   # triggers apply_defaults() if missing
    pre_transform: Any = None

    def __call__(self, input_dict, context=None):        # :48
        self._context = context
        self._validate_required_keys(input_dict)         # 1. required keys → KeyError if absent
        self._handle_optional_keys(input_dict)           # 2. fill optional defaults
        if not self._should_apply():                     # 3. probability gate
            return self.on_skip(input_dict)              #    (default: return unchanged)
        return self.transform(input_dict)                # 4. the real work

    @abstractmethod
    def transform(self, input_dict) -> dict: ...         # :153  YOU implement — return UPDATES
```

The four fixed steps (`__call__:48`) mean a transform author only writes `transform()`. The
base handles validation, optional-key defaults, and the probability gate uniformly.

`_should_apply()` (`:112`): `p is None` → always; `p<=0` → never; `p>=1` → always; else
`np.random.rand() < p`. So set `p` for stochastic augmentations, leave it `None` for loaders
and deterministic ops.

### Composition (`transforms/base.py:167`)

```python
class TransformsCompose:
    def __init__(self, pipeline=()):
        self.pipeline = list(pipeline)

    def __call__(self, input_dict, context=None):        # :182
        for transform in self.pipeline:
            output = transform(input_dict, context=context)
            if not isinstance(output, dict): raise TypeError(...)   # each transform MUST return a dict
            input_dict |= output                         # :203  merge updates
        return input_dict
```

The `input_dict |= output` merge is the whole contract: a transform returns the subset of
keys it touched, and they overwrite the running dict. This is why a loader can return just
`{"points": ...}` and an augmentation can return just `{"points": ..., "gt_boxes": ...}`.

---

## 3. Reading real transforms

### A loader — `LoadPointsFromFile` (`transforms/point_cloud/loading.py:27`)

```python
class LoadPointsFromFile(BaseTransform):
    _required_keys = ["lidar_path"]          # will KeyError without it

    def __init__(self, *, load_dim=5, use_dim=(0, 1, 2, 3)):
        self.load_dim, self.use_dim = load_dim, use_dim

    def transform(self, input_dict):
        load_dim = int(input_dict.get("num_pts_feats", self.load_dim))
        points = np.fromfile(input_dict["lidar_path"], dtype=np.float32).reshape(-1, load_dim)
        # optional single-source slicing / sensor-frame transform ...
        points = points[:, list(self.use_dim)] if not isinstance(self.use_dim, int) else points[:, :self.use_dim]
        return {"points": points.astype(np.float32)}    # only the key it produced
```

Takeaways: `_required_keys` declares its input contract; it reads a *path* from metadata and
returns a `points` array; `p` is `None` (a loader always runs).

### An augmentation — `GlobalRotScaleTrans` (`transforms/point_cloud/geometry.py:112`)

```python
class GlobalRotScaleTrans(BaseTransform):
    _required_keys = []

    def __init__(self, *, rot_range, scale_ratio_range, translation_std=None): ...

    def transform(self, input_dict):
        g3d.require_point_cloud(input_dict)
        rotation, rotation_angle, scale, translation = g3d.sample_rot_scale_trans(
            self.rot_range, self.scale_ratio_range, self.translation_std)
        g3d.transform_points(input_dict, rotation, scale, translation)   # points
        g3d.transform_normal(input_dict, rotation)                       # normals (if present)
        g3d.transform_boxes(input_dict, rotation, rotation_angle, scale, translation)  # gt_boxes
        return input_dict
```

**The load-bearing detail:** an augmentation transforms **points *and* boxes (and normals)
together**. `RandomFlip3D` (`geometry.py:75`) does the same — every flip is applied to
`points`, `normal`, *and* `gt_boxes`. If you write an augmentation that moves points but
forgets boxes, your labels silently desynchronize and training quietly degrades. All the math
is centralized in `autoware_ml/transforms/geometry3d.py` (`g3d`), and the camera / camera-lidar
variants reuse the exact same functions so LiDAR-only, camera, and fusion augmentations stay
consistent.

---

## 4. The transform library map (`autoware_ml/transforms/`)

Point `_target_` at the concrete implementation module (no `__init__` re-exports).

| Folder | Purpose | Representative transforms |
| ------ | ------- | ------------------------- |
| `common/` | modality-agnostic | `Copy`, `BuildPointFeatures`, `PermuteAxes` |
| `point_cloud/` | LiDAR | **loading:** `LoadPointsFromFile`, `LoadPointsFromMultiSweeps`; **geometry:** `GlobalRotScaleTrans`, `RandomFlip3D`, `RandomRotateTargetAngle`; **crop:** `PointsRangeFilter`, `CropBoxInner/Outer`, `SphereCrop`; **sampling:** `PointShuffle`, `RandomDropout`, `ElasticDistortion`, `GridSample`; **perturbation:** `RandomJitter`, `RandomShift`; **formatting:** `PreparePointCloudInput` |
| `boxes3d/` | 3D annotations | `LoadAnnotations3D` (→ `gt_boxes`,`gt_names`,`gt_labels`,`gt_num_points`), `MergeObjects3D`, filters `ObjectRangeFilter`/`ObjectNameFilter`/`ObjectMinPointsFilter`/`ObjectRangeMinPointsFilter` |
| `camera/` | images | `LoadImageFromFile`, `LoadMultiViewImagesFromFiles`, resize/crop/flip, normalize, `GridMask`, `UndistortImage` |
| `camera_lidar/` | fusion | `LidarCameraFusion`, `CalibrationMisalignment`, `ImageAug3D`, `BEVLoadMultiViewImageFromFiles` |
| `image/` | 2D-only | `PhotometricDistortion` |
| `segmentation3d/` | seg labels/aug | `LoadSegAnnotations3D`, `PreparePointSegInput`, `FrustumMix`, `InstanceCopy`, `RangeInterpolation` |
| `multi_task/` | typed multi-task stack | `MultiTaskTransformsCompose` + its own loading/geometry (operates on Pydantic samples) |

---

## 5. Configuring pipelines (per split)

Transforms are wired as a `TransformsCompose` with a `pipeline:` list. From the CenterPoint
NuScenes leaf config:

```yaml
datamodule:
  train_transforms:
    _target_: autoware_ml.transforms.base.TransformsCompose
    pipeline:
      - _target_: autoware_ml.transforms.boxes3d.loading.LoadAnnotations3D
        name_mapping: ${dataset.detection3d.name_mapping}
      - _target_: autoware_ml.transforms.point_cloud.sweeps.LoadPointsFromMultiSweeps
        sweeps_num: 10
        load_dim: 5
        use_dim: [0, 1, 2, 3, 4]
      - _target_: autoware_ml.transforms.point_cloud.geometry.RandomFlip3D
        flip_ratio_bev_horizontal: 0.5
        flip_ratio_bev_vertical: 0.5
      - _target_: autoware_ml.transforms.point_cloud.geometry.GlobalRotScaleTrans
        rot_range: [-1.571, 1.571]
        scale_ratio_range: [0.9, 1.1]
        translation_std: [0.5, 0.5, 0.2]
      - _target_: autoware_ml.transforms.point_cloud.crop.PointsRangeFilter
        point_cloud_range: ${point_cloud_range}
      - _target_: autoware_ml.transforms.boxes3d.filters.ObjectRangeFilter
        point_cloud_range: ${point_cloud_range}
      - _target_: autoware_ml.transforms.point_cloud.sampling.PointShuffle

  val_transforms:                       # ← NO random augmentation
    _target_: autoware_ml.transforms.base.TransformsCompose
    pipeline:
      - _target_: autoware_ml.transforms.boxes3d.loading.LoadAnnotations3D
        name_mapping: ${dataset.detection3d.name_mapping}
      - _target_: autoware_ml.transforms.point_cloud.sweeps.LoadPointsFromMultiSweeps
        sweeps_num: 10
      - _target_: autoware_ml.transforms.point_cloud.crop.PointsRangeFilter
        point_cloud_range: ${point_cloud_range}

  test_transforms: ${datamodule.val_transforms}    # reuse val's pipeline verbatim
```

Rules of thumb visible here:

- **Order matters.** Load annotations & points first; augment; then crop/filter; shuffle last.
- **Val/test/predict must not randomly augment.** Only deterministic loading + range cropping.
  `test_transforms: ${datamodule.val_transforms}` guarantees they match.
- **`predict_transforms` usually omits annotation loading** (no labels at inference).

---

## 6. Writing a new transform (recipe)

```python
# autoware_ml/transforms/point_cloud/my_aug.py
from typing import Any
import numpy as np
from autoware_ml.transforms.base import BaseTransform

class RandomIntensityScale(BaseTransform):
    """Scale the per-point intensity channel. Reads/writes `points`."""
    _required_keys = ["points"]          # fail loudly if points aren't loaded yet

    def __init__(self, *, p: float = 0.5, scale_range=(0.9, 1.1)):
        self.p = p                        # BaseTransform's gate handles the probability
        self.scale_range = scale_range

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        points = input_dict["points"]
        scale = np.random.uniform(*self.scale_range)
        points[:, 3] = points[:, 3] * scale     # column 3 = intensity
        return {"points": points}               # return only what you changed
```

Then add it to a `pipeline:` list in config. Checklist:

- Declare `_required_keys` so misordered pipelines fail with a clear `KeyError`.
- Return **only** changed keys.
- Set `p` for stochastic behavior; leave `None` for deterministic ops.
- If the aug moves geometry, transform **points and boxes together** (reuse `g3d` helpers).
- Keep it CPU/numpy (it runs in workers). GPU work belongs in `preprocessing/`.

---

## 7. Gotchas specific to transforms

- **In-place mutation.** Many transforms mutate arrays in place *and* return them. That's fine
  because each sample is independent per worker — but don't cache/share arrays across samples.
- **A produced key still gets dropped** if it isn't in `collation_map`. Producing `gt_names`
  in a transform doesn't make it reach the model unless collation keeps it.
- **`context` for mixing.** Copy-paste style augmentations (`FrustumMix`, `InstanceCopy`) use
  `self.context.sample_secondary(...)` to fetch another sample. Don't reach into the dataset
  directly.
- **`apply_defaults` must be implemented** if you declare `_optional_keys`, or the base raises
  `NotImplementedError` (`base.py:135`).

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `KeyError: Missing required key 'points'` | transform ordered before its loader | put loaders first in the pipeline |
| Boxes drift / mAP collapses with augmentation on | an aug moved points but not boxes | transform points **and** `gt_boxes` jointly (use `g3d`) |
| `TypeError: ... must return a dict` | `transform()` returned `None`/array | return a dict of updates |
| Augmentation active during eval | random transform in `val/test_transforms` | keep val/test deterministic; reuse val via `${...}` |
| Non-reproducible runs | RNG not seeded | training seeds via `L.seed_everything(..., workers=True)`; per-transform use `np.random` (seeded by workers) |
| Change to a transform has no effect | edited a different `_target_` than the config points at | verify the exact module path in the pipeline |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Add an augmentation | Write a `BaseTransform`, add it to `train_transforms.pipeline` |
| Tune augmentation strength | Edit the transform's args in config (`rot_range`, `scale_ratio_range`, `p`, …) |
| Turn augmentation off for an ablation | Remove entries from `train_transforms.pipeline` (or set `p: 0.0`) |
| Load an extra feature channel | `LoadPointsFromFile`/`LoadPointsFromMultiSweeps` `load_dim`/`use_dim`, then thread the channel through the model |
| Add a mixing augmentation | Use `self.context.sample_secondary(...)`; model after `FrustumMix`/`InstanceCopy` |

---

**Next (Phase 3):** [../model/model_architecture.md](../model/model_architecture.md) — how a
model consumes this batch and turns it into predictions and a loss.
