# Code Walkthrough — Config Flow

> How the single `cfg` object that drives a run is assembled from many small YAML files.
> This is the part of the framework most unlike plain PyTorch, so it gets a full, worked
> example: **`detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes`**.
>
> Reference: `docs/user-guide/configuration.md`. This document is the "trace one real config
> to the objects it builds" companion.

---

## Mental model: config = a tree that Hydra turns into objects

A run's config is one big nested dictionary (`DictConfig`). Two mechanisms build it:

1. **Composition** — Hydra merges many YAML files into one tree, guided by `defaults:` lists.
2. **Instantiation** — `hydra.utils.instantiate(cfg.<section>)` walks the tree and, wherever
   it sees a `_target_`, imports that class/function and calls it. Nested `_target_`s are
   built first (bottom-up), so a parent receives already-constructed children.

So a config is a *recipe*, and `instantiate` is the *oven*.

---

## The vocabulary you must know

| Token | Meaning | Example |
| ----- | ------- | ------- |
| `defaults:` | list of other configs to merge in, in order | `- /defaults/default_runtime` |
| `_self_` | where *this file's* values apply within the defaults order | usually last = this file wins |
| `# @package _global_` | merge this file's keys at the config root | top of task configs |
| `# @package nuscenes` | merge this file's keys under `cfg.nuscenes` | dataset group files |
| `_target_` | Python import path to instantiate | `autoware_ml.models.detection3d.centerpoint.CenterPointDetectionModel` |
| `_partial_: true` | build a `functools.partial`, don't call yet | optimizers/schedulers |
| `${a.b.c}` | interpolate another config value | `${voxel_size}` |
| `${oc.env:VAR,default}` | interpolate an env var | run dir |
| `???` | **mandatory** value; must be filled or Hydra errors | `dataset: ???` |
| `${resolver:arg}` | call a custom OmegaConf resolver | `${user_config_name:...}` |

---

## The composition chain for our example

```text
tasks/detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes.yaml   ← the LEAF (what you run)
   defaults:
     - /tasks/detection3d/centerpoint/base          ← the model/family BASE
         defaults:
           - /defaults/default_runtime              ← the runtime scaffold
               defaults:
                 - modules/callbacks
                 - modules/data_preprocessing
                 - modules/datamodule
                 - modules/deploy
                 - modules/logger
                 - modules/model
                 - modules/run
                 - modules/trainer
     - /datasets/nuscenes/detection3d               ← dataset group (@package nuscenes.detection3d)
     - /datasets/nuscenes/lidar                      ← dataset group (@package nuscenes)
     - _self_                                         ← the leaf's own overrides win last
```

Read it top-down as "the leaf pulls in the base, which pulls in the runtime scaffold, which
pulls in the module fragments." Read it as *precedence* bottom-up: later entries and `_self_`
override earlier ones.

---

## Layer 1 — the runtime scaffold (`defaults/default_runtime.yaml`)

```yaml
# @package _global_
defaults:
  - modules/callbacks          # ModelCheckpoint (monitor val/loss), EarlyStopping, LRMonitor
  - modules/data_preprocessing # DataPreprocessing() shell
  - modules/datamodule         # dataloader scaffolding
  - modules/deploy             # deploy.onnx.* / deploy.tensorrt.* defaults
  - modules/logger             # MLFlowLogger, tracking_uri sqlite:///mlruns/mlflow.db
  - modules/model              # model shell
  - modules/run                # hydra.run.dir (reads AUTOWARE_ML_HYDRA_RUN_DIR)
  - modules/trainer            # lightning.Trainer defaults (max_epochs, precision, ...)
```

Every task config inherits this, so every run has a trainer, a logger, callbacks, and a
deploy section *by default*. A task only overrides what it needs.

---

## Layer 2 — the model/family base (`tasks/detection3d/centerpoint/base.yaml`)

This is where the **CenterPoint architecture** is defined once, for all CenterPoint variants:

```yaml
# @package _global_
defaults:
  - /defaults/default_runtime
  - _self_                       # base's own values override the runtime scaffold

dataset: ???                     # MANDATORY: the leaf must supply a dataset group
point_cloud_range: ???           # MANDATORY: depends on range/voxel choice
voxel_size: ???                  # MANDATORY

datamodule:
  collation_map:                 # the strict whitelist (see data_flow.md)
    points: list
    gt_boxes: list
    gt_labels: list

model:
  _target_: autoware_ml.models.detection3d.centerpoint.CenterPointDetectionModel
  metrics: ${dataset.detection3d.metrics}          # pulled from the dataset group
  pts_voxel_encoder:
    _target_: autoware_ml.models.detection3d.encoders.pillar.PillarFeatureNet
    in_channels: 5
    voxel_size: ${voxel_size}                      # interpolation from the top level
    point_cloud_range: ${point_cloud_range}
    ...
  pts_middle_encoder:
    _target_: autoware_ml.models.detection3d.encoders.pillar.PointPillarsScatter
    in_channels: 32
    output_shape: ???                              # MANDATORY: depends on range/voxel → grid size
  pts_backbone:
    _target_: autoware_ml.models.detection3d.backbones.second.SECONDBackbone
    layer_strides: ???                             # MANDATORY
  pts_neck:
    _target_: autoware_ml.models.detection3d.necks.second_fpn.SECONDFPN
  bbox_head:
    _target_: autoware_ml.models.detection3d.heads.centerpoint.CenterHead
    num_classes: ${dataset.detection3d.num_classes}
    class_names: ${dataset.detection3d.class_names}
    out_size_factor: ???                           # MANDATORY
    nms_min_radius: ???                            # MANDATORY
  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true                                # → functools.partial(AdamW, lr=..., weight_decay=...)
    lr: 0.0001
    weight_decay: 0.01
  scheduler:
    _target_: autoware_ml.utils.schedulers.cyclic_cosine_annealing.CyclicCosineAnnealingLR
    _partial_: true
    warmup_epochs: 8
    decay_epochs: 22
    max_lr_factor: ???                             # MANDATORY

trainer:
  max_epochs: 30                                   # overrides the scaffold's default
  gradient_clip_val: 5.0
  gradient_clip_algorithm: norm

deploy:
  onnx:
    dynamo: false                                  # CenterPoint uses the legacy ONNX path
    opset_version: 17
    modules:                                       # CenterPoint exports TWO onnx modules
      pts_voxel_encoder_centerpoint: { ... }
      pts_backbone_neck_head_centerpoint: { ... }
  tensorrt:
    enabled: false

data_preprocessing:
  pipeline:
    - _target_: autoware_ml.preprocessing.detection3d.point_pillar.PointPillarPreprocessor
      voxel_size: ${voxel_size}
      point_cloud_range: ${point_cloud_range}
      max_num_points: 32
      max_voxels: 96000
```

Two things to notice:

- **`???` (mandatory-missing).** The base cannot know the voxel grid size until you pick a
  range/voxel size. So it declares `point_cloud_range`, `voxel_size`, `output_shape`,
  `layer_strides`, `out_size_factor`, `nms_min_radius`, `max_lr_factor` as `???` and forces
  the leaf to fill them. Forgetting one raises `MissingMandatoryValue`.
- **`_partial_: true` only on optimizer/scheduler.** Every *module* (`pts_voxel_encoder`,
  `pts_backbone`, …) is built into an `nn.Module` before the model constructor runs. But the
  optimizer/scheduler cannot be built yet — they need the model's parameters, which don't
  exist until the model is built. So they stay as *callables* (`functools.partial`) and are
  invoked later, inside `configure_optimizers()`. See
  [../training/optimizer_scheduler.md](../training/optimizer_scheduler.md).

---

## Layer 3 — the leaf (`voxel020_second_secfpn_51m_nuscenes.yaml`)

The leaf answers "which dataset, what range/voxel, what data loading":

```yaml
# @package _global_
defaults:
  - /tasks/detection3d/centerpoint/base    # inherit the architecture
  - /datasets/nuscenes/detection3d         # dataset group → fills cfg.nuscenes.detection3d
  - /datasets/nuscenes/lidar               # dataset group → lidar settings
  - _self_                                  # the leaf's values win last

batch_size: 16
num_workers: 8

dataset: ${nuscenes}                        # fill the base's `dataset: ???` with the nuscenes group

point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]   # fill the ???
voxel_size: [0.2, 0.2, 8.0]

model:                                       # fill the architecture's remaining ???
  pts_middle_encoder: { output_shape: [512, 512] }
  pts_backbone:       { layer_strides: [2, 2, 2] }
  bbox_head:          { out_size_factor: 4, nms_min_radius: 1.0 }
  scheduler:          { max_lr_factor: 10.0 }

datamodule:
  _target_: autoware_ml.datamodule.nuscenes.detection3d.NuscenesDetection3DDataModule
  data_root: ${dataset.data_root}
  train_ann_file: ${datamodule.data_root}/nuscenes_infos_train.pkl
  val_ann_file:   ${datamodule.data_root}/nuscenes_infos_val.pkl
  class_names: ${dataset.detection3d.class_names}
  train_dataloader_cfg: { batch_size: ${batch_size}, num_workers: ${num_workers}, shuffle: true }
  train_transforms:
    _target_: autoware_ml.transforms.base.TransformsCompose
    pipeline:
      - _target_: autoware_ml.transforms.boxes3d.loading.LoadAnnotations3D
        name_mapping: ${dataset.detection3d.name_mapping}
      - _target_: autoware_ml.transforms.point_cloud.sweeps.LoadPointsFromMultiSweeps
        sweeps_num: 10
      - _target_: autoware_ml.transforms.point_cloud.geometry.RandomFlip3D
      - _target_: autoware_ml.transforms.point_cloud.geometry.GlobalRotScaleTrans
      - _target_: autoware_ml.transforms.point_cloud.crop.PointsRangeFilter
        point_cloud_range: ${point_cloud_range}
      - _target_: autoware_ml.transforms.boxes3d.filters.ObjectRangeFilter
        point_cloud_range: ${point_cloud_range}
      - _target_: autoware_ml.transforms.point_cloud.sampling.PointShuffle
  val_transforms: { ... no random augmentation ... }
  test_transforms: ${datamodule.val_transforms}   # test reuses val's pipeline
```

Notice `test_transforms: ${datamodule.val_transforms}` — interpolation lets one pipeline be
reused, and guarantees test and val preprocess identically.

---

## Dataset groups and the `@package` directive

Why is it `dataset: ${nuscenes}` and not the dataset inline? Because the dataset lives in its
own file with a **package directive** that decides *where* its keys land:

```yaml
# datasets/nuscenes/detection3d.yaml
# @package nuscenes.detection3d      ← everything here goes under cfg.nuscenes.detection3d
defaults:
  - /datasets/nuscenes/base
  - _self_
class_names: [...]
num_classes: 10
name_mapping: {...}
metrics: [ { _target_: autoware_ml.metrics.detection3d.suite.Detection3DMetricSuite, ... } ]
```

So after composition, `cfg.nuscenes.detection3d.class_names` exists, and the leaf's
`dataset: ${nuscenes}` makes `cfg.dataset` point at that group. That is why the model reads
`num_classes: ${dataset.detection3d.num_classes}` and `metrics: ${dataset.detection3d.metrics}`
— the *same* dataset definition feeds the model, the datamodule, and the metrics, so they
cannot disagree about class names.

This indirection (config → variable → group) is deliberate: to retarget the whole run at a
different dataset, you swap one `defaults:` entry.

---

## From `cfg` to objects: what `instantiate(cfg.model)` does

```text
hydra.utils.instantiate(cfg.model)
  ├─ build cfg.model.pts_voxel_encoder  → PillarFeatureNet(in_channels=5, voxel_size=[0.2,0.2,8.0], ...)
  ├─ build cfg.model.pts_middle_encoder → PointPillarsScatter(in_channels=32, output_shape=[512,512])
  ├─ build cfg.model.pts_backbone       → SECONDBackbone(layer_strides=[2,2,2], ...)
  ├─ build cfg.model.pts_neck           → SECONDFPN(...)
  ├─ build cfg.model.bbox_head          → CenterHead(num_classes=10, out_size_factor=4, ...)
  ├─ build cfg.model.metrics            → [Detection3DMetricSuite(...)]
  ├─ leave cfg.model.optimizer          → functools.partial(AdamW, lr=1e-4, weight_decay=0.01)   (_partial_)
  ├─ leave cfg.model.scheduler          → functools.partial(CyclicCosineAnnealingLR, ...)        (_partial_)
  └─ call CenterPointDetectionModel(pts_voxel_encoder=..., pts_backbone=..., bbox_head=...,
                                     optimizer=<partial>, scheduler=<partial>, metrics=[...])
```

Children are built first, then handed to the parent constructor. This is why a model's
`__init__` receives fully-built `nn.Module`s, not configs.

---

## Overriding from the command line

```bash
# override an EXISTING key (no +)
autoware-ml train --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes \
    trainer.max_epochs=50 model.optimizer.lr=5e-4 batch_size=8

# ADD a new key (+)
autoware-ml train --config-name ... +trainer.limit_train_batches=10

# print the fully composed config WITHOUT running
autoware-ml train --config-name ... --cfg job
# print just one section
autoware-ml train --config-name ... --cfg job --package model
```

`--cfg job` is the single most useful debugging tool for configs: it shows the exact,
interpolation-resolved tree that would be instantiated.

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `MissingMandatoryValue` / `??? ` error | A base's `???` field wasn't filled by the leaf | Fill it in the leaf (e.g. `output_shape`, `voxel_size`) |
| `InterpolationKeyError: ${dataset...}` | The dataset group wasn't composed, or `dataset:` not set | Ensure the leaf has `/datasets/...` in `defaults` and `dataset: ${...}` |
| Override "could not be added" | Used a plain override on a missing key | Add `+` to create a new key |
| Override "already exists" with `+` | Used `+` on an existing key | Drop the `+` |
| Value silently not applied | `_self_` ordering: something after it overrode you | Check the `defaults:` order; `_self_` last = this file wins |
| Wrong class built | `_target_` typo / stale path | `--cfg job --package <section>` and read the `_target_` |
| Two sections disagree on class names | Not reading from the shared `dataset` group | Point them all at `${dataset.detection3d....}` |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Train the same model on a new dataset | New leaf: swap the `/datasets/...` `defaults` entries and `dataset: ${...}`, set range/voxel and ann files |
| Add a new variant (e.g. longer range) | New leaf inheriting `.../base`, fill the `???`s for that range |
| Change a hyperparameter for one experiment | CLI override (`model.optimizer.lr=...`), no file edit |
| Add a new configurable module to the model | Add a `_target_` block in the base; add the constructor arg to the model class |
| Retune metrics without restating the suite | Override `metric_ranges` / `metric_eval_class_range` variables — see [../evaluation/metrics.md](../evaluation/metrics.md) |

---

**Next:** [important_classes.md](important_classes.md) — the classes these configs instantiate.
