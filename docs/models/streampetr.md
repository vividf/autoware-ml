---
icon: lucide/cctv
---

# StreamPETR

<!-- cspell:ignore CPFPN fcos3d imgbackbone -->

StreamPETR is a camera-based 3D object detection model integrated under the `detection3d` task namespace. It uses a multiview image backbone, a feature pyramid neck, and a native query-based detection head with a streaming memory queue for temporal modeling.

## Summary

| Property     | Value                                       |
| ------------ | ------------------------------------------- |
| Task         | 3D object detection                         |
| Modality     | Camera                                      |
| Input        | Synchronized multiview images               |
| Output       | 3D bounding boxes and class scores          |
| Architecture | Multiview VoVNet/CPFPN + query decoder head |
| Datasets     | NuScenes, T4Dataset                         |

## Available Configurations

| Config Name                                            | Dataset   | Purpose                                                  |
| ------------------------------------------------------ | --------- | -------------------------------------------------------- |
| `detection3d/streampetr/vov_320x800_nuscenes`          | NuScenes  | Plain nuScenes reference configuration                   |
| `detection3d/streampetr/vov_320x800_nuscenes_pretrain` | NuScenes  | Stage 1: nuScenes pretrain (CPFPN + auxiliary 2D head)   |
| `detection3d/streampetr/vov_480x640_t4dataset_base`    | T4Dataset | Stage 2: full T4 base DB                                 |
| `detection3d/streampetr/vov_480x640_t4dataset_j6gen2`  | T4Dataset | Stage 3: j6gen2 fine-tune — the production configuration |

## Three-Stage Training Flow

The production model is trained natively in three stages; each stage's config
pins the exact recipe (GPU count, per-GPU batch, LRs) of the accepted run.
There is no `auto_scale_lr`: for a different total batch size rescale both
LRs linearly (each config's header states its base).

| Stage | Config                                                 | Database                | Epochs | Total batch / LR | Init from                      |
| ----- | ------------------------------------------------------ | ----------------------- | ------ | ---------------- | ------------------------------ |
| 1     | `detection3d/streampetr/vov_320x800_nuscenes_pretrain` | `nuscenes_base`         | 30     | 32 / 8e-4        | RGB-stem DD3D/FCOS3D VoVNet-99 |
| 2     | `detection3d/streampetr/vov_480x640_t4dataset_base`    | `t4dataset_base`        | 35     | 32 / 2e-4        | stage-1 `best.ckpt`            |
| 3     | `detection3d/streampetr/vov_480x640_t4dataset_j6gen2`  | `t4dataset_j6gen2_base` | 35     | 64 / 4e-4        | stage-2 `best.ckpt`            |

`--weights` accepts the previous stage's Lightning checkpoint directly — no
conversion between stages. Every stage trains the same 7 classes in the same
order (car, truck, bus, bicycle, pedestrian, traffic_cone, barrier): the T4
stages select the `seven_class` taxonomy of the database, which folds the
coarse taxonomy into the nuScenes detection classes and the list the deployed
camera detector decodes. Motorcycles train as bicycles, debris and
pushable/pullable objects as barriers, and animals are no targets.

The accepted recipe trains `traffic_cone`/`barrier` with partial-ignore:
frames whose scene lacks those annotations do not punish their background
predictions, annotated frames train them normally. Partial-ignore reads the
per-frame flag of the batch (`gt_traffic_cone_barrier_bbox_status`) and
refuses to run without it. The scenario lists carry the flag, but the record
table does not yet, so the T4 stages ship with `partial_ignore_classes: null`
and these two classes train as plain classes. Once the database fills the
flag, set `partial_ignore_classes=[traffic_cone,barrier]`.

The data is read from the record table of the database (built on first use
under `cache_root_path`) and split by the scenario lists under
`t4dataset_scenario_root_path`. Every dataloader streams whole scenes per
batch lane (`datamodule.streaming`), ordered by the sample time of the
records, and evaluation runs one lane per rank so every frame is scored.

The `img_backbone` LR is always `lr * 0.1`, set via
`model.optimizer_group_overrides.img_backbone.lr`. Validation runs every 5
epochs; `best.ckpt` is selected by `val/det3d/mAP`.

### Stage 1 — nuScenes pretrain

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_320x800_nuscenes_pretrain \
    --weights <rgb_stem_vovnet_backbone.pth>
```

The backbone init is the upstream StreamPETR DD3D/FCOS3D-pretrained
VoVNet-99 (`fcos3d_vovnet_imgbackbone-remapped.pth`, BGR); without it the
recipe underperforms. The detector keeps its backbone inside its image feature
extractor, so the file's `img_backbone.*` tensors are renamed to
`image_feature_extractor.img_backbone.*` (under the original names no tensor
matches and `--weights` fails) and the stem conv is flipped to RGB:

```python
import torch

state = torch.load("fcos3d_vovnet_imgbackbone-remapped.pth", map_location="cpu")
state = state.get("state_dict", state)
backbone = {
    f"image_feature_extractor.{key}": value
    for key, value in state.items()
    if key.startswith("img_backbone.")
}
stem = "image_feature_extractor.img_backbone.stem.stem_1/conv.weight"
backbone[stem] = backbone[stem][:, [2, 1, 0]].contiguous()  # BGR -> RGB
torch.save({"state_dict": backbone}, "rgb_stem_vovnet_backbone.pth")
```

The training log should report `Loaded matching weight tensors: 626/894` for
this init. Watch the first few hundred iterations: at lr 8e-4 with grad-clip 35
a loss spike/NaN means both LRs should back off (e.g. to 6e-4 / 6e-5).

Accepted run: val mAP **0.5031** at epoch 29 (7-class; the upstream 10-class
published result is 0.4697).

### Stage 2 — T4 base DB

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_base \
    --weights mlruns/detection3d/streampetr/vov_320x800_nuscenes_pretrain/<run_id>/artifacts/checkpoints/best.ckpt
```

Cross-stage checkpoint loads must report `Loaded matching weight tensors:
894/894` — full coverage, nothing dropped (the nuScenes and T4 stages share the
same 7-class head).

Accepted run: val mAP **0.4420** at epoch 34. That run used the older
5-class base infos (no cone/barrier at all — a 35-epoch supervision gap for
those classes); the T4 base database carries both classes, so a re-run's
numbers are not directly comparable.

### Stage 3 — j6gen2 fine-tune

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_base/<run_id>/artifacts/checkpoints/best.ckpt
```

For a pipeline validation run add `+trainer.fast_dev_run=true`. For a short
real run instead (e.g. `trainer.max_epochs=1`) also set
`trainer.check_val_every_n_epoch=1`: validation defaults to every 5 epochs,
and a run that finishes without ever validating fails at teardown. Full-length
training is unaffected.

Resuming an interrupted run: `--resume-checkpoint <last.ckpt>` (mutually
exclusive with `--weights`; repeat every other CLI override, they are not
restored from the checkpoint; add `--new-run` only to fork a new MLflow run).

### Evaluate

```bash
autoware-ml test \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt \
    trainer.devices=1
```

The headline number is the `0-121m` bucket mAP, directly comparable to AWML
T4MetricV2's `mAP_center_distance_bev` in the `bev_center_0.0-121.0`
evaluator (see the alignment notes below).

### Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/.../checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The export traces the three graphs the runtime node binds
(`extract_img_feat`, `position_embedding`, `pts_head_memory`) from the first
sample of the predict split. `extract_img_feat` takes images normalized with
the ImageNet statistics: the node normalizes the camera images itself, so the
`BatchNormalizeImages` layer of the training preprocessing is not part of the
graph. The current verification scope covers ONNX export. TensorRT engine
generation has not been validated yet.

## Results

Final accepted stage-3 run (batch 64 / lr 4e-4, trained on the 2.8 info
generation; an earlier batch-32 / lr 2e-4 run matched within ±0.5 pp on
every headline number). The three stage `best.ckpt` files are archived on
the training host under `~/ckpt_backup/`. These numbers predate the
record-table data pipeline and the removal of the head's `use_bottom_center`
shift, so the stages have to be retrained and re-measured on the current
code.

| Split | mAP (0-121m) | 0-50m  | 50-90m |
| ----- | ------------ | ------ | ------ |
| val   | 0.5160       | —      | —      |
| test  | **0.5025**   | 0.5336 | 0.1730 |

Test per-class mAP (0-121m): car 0.677, bus 0.651, truck 0.585, bicycle
0.456, pedestrian 0.464, traffic_cone 0.344, barrier 0.340. The 90-121m
bucket is 0 by construction: `point_cloud_range` is ±51.2 m and
`post_center_range` ±61.2 m, so no predictions exist beyond ~86 m.

The two weakest classes trace to the data, not the recipe: the accepted
run's stage-2 infos lacked cone/barrier entirely (a 35-epoch supervision
gap) and only part of the j6gen2 scenes carry them. The T4 base database
includes both classes, and partial-ignore keeps the scenes without them from
punishing their background once the per-frame flag reaches the batch — the
expected fix for these two classes on the next training round.

For reference, the earlier AWML-checkpoint-conversion route on the same split
scored val 0.3913 / test 0.3661, and AWML itself 0.3752 / 0.3552 — the
native three-stage flow exceeds both by a wide margin. Cross-checks backing
comparability: the metric stacks agree to ≤ 2.5e-8 per class on identical
predictions, and running one AWML checkpoint through both inference stacks
differs by −0.80 test mAP (camera-pipeline numerics, consistent in sign
across all 7 classes).

## Evaluation Alignment Notes

Two evaluation-side conditions make mAP comparable with AWML T4MetricV2:

1. **`gt_num_points` reaches the metric** — without it the evaluation-time
   min-points GT filter silently never engages. The shared detection eval
   output always attaches the point count of every GT box.
2. **No radial `eval_class_range` cap inside the pc_range square** — the GT
   is already limited to the ±51.2 m *square* by the pipeline
   `BBoxesRangeFilter`. An additional radial cap at 51.2/54 m removed the
   square's corners from the GT while corner *predictions* stayed and became
   guaranteed false positives — a semantics T4MetricV2 (radial on both GT
   and predictions) cannot reproduce. The `seven_class` taxonomy therefore
   evaluates every class up to 121 m (a no-op inside the square), and the
   T4 distance buckets mirror T4MetricV2's (0-50 / 50-90 / 90-121 / 0-121 m).

Validation during training streams one lane per rank. With several GPUs the
streaming sampler trims every rank to the shortest one, so `val/det3d/mAP`
(the checkpoint monitor) skips a few percent of the frames; `autoware-ml
test` on one device scores every frame.

### Ground truth against the accepted run

The accepted run read AWML-generated info files; the stages now read the
record tables of the database. The `seven_class` taxonomy and the database
box pipelines differ from those info files in a few places:

| Ground truth                                             | Accepted run (info files) | Now (database)    |
| -------------------------------------------------------- | ------------------------- | ----------------- |
| forklift                                                 | car                       | truck             |
| trailer without a truck to merge into                    | not a target              | truck             |
| construction sign                                        | not a target              | barrier           |
| truck–trailer merge distance                             | 2.0 m                     | 1.0 m             |
| ground-plane velocity                                    | as annotated              | clipped at 25 m/s |
| two-wheeler with `two_wheel_vehicle_state.without_rider` | kept                      | dropped           |

The first three follow the coarse taxonomy every other T4 detector of the
repository trains, the others are the shared box pipelines and attribute
filters of the T4 dataset configs.

## Implementation

| Path                                                       | Description                                       |
| ---------------------------------------------------------- | ------------------------------------------------- |
| `autoware_ml/models/detection3d/streampetr.py`             | StreamPETR model wrapper                          |
| `autoware_ml/models/detection3d/heads/streampetr.py`       | Query-based detection head                        |
| `autoware_ml/models/detection3d/heads/focal2d.py`          | Auxiliary 2D head (FocalHead2D)                   |
| `autoware_ml/models/detection3d/partial_ignore.py`         | Partial-ignore label handling                     |
| `autoware_ml/models/common/backbones/vovnet.py`            | Multiview image backbone                          |
| `autoware_ml/models/common/necks/cp_fpn.py`                | CPFPN neck (reference StreamPETR neck)            |
| `autoware_ml/models/detection3d/task_modules/`             | Shared assigners, costs, coders, streaming memory |
| `autoware_ml/datamodule/data_module.py`                    | `streaming` dataloaders (`GroupStreamingSampler`) |
| `autoware_ml/datamodule/base_dataset.py`                   | Scene grouping of the record table                |
| `autoware_ml/dataclasses/batch/frame_meta.py`              | Per-frame pose, sample time and scenario          |
| `autoware_ml/transforms/camera/annotations2d.py`           | 2D targets projected from the 3D boxes            |
| `autoware_ml/transforms/camera/geometry.py`                | `ImageAug3D`, `CameraGlobalRotScaleTrans`         |
| `autoware_ml/preprocessing/camera/normalize.py`            | Batched image normalization                       |
| `autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py` | Warmup + epoch-cosine LR schedule                 |
| `autoware_ml/configs/tasks/detection3d/streampetr/`        | Task configurations                               |
| `autoware_ml/configs/database/t4dataset/taxonomy/`         | `seven_class` taxonomy of the T4 stages           |

An AWML/mmdet3d checkpoint converter
(`autoware_ml/tools/convert_streampetr_checkpoint.py`) existed during the
migration and was removed once the native training flow replaced converted
checkpoints; recover it from git history if an mm-style checkpoint ever needs
importing again. Its output predates the image feature extractor, so prefix
its `img_backbone.*` and `img_neck.*` keys with `image_feature_extractor.`:
under the old names only the heads match, and `--weights` leaves the backbone
and the neck randomly initialized without an error.

## Acknowledgment

<!-- cspell:ignore exiawsh -->
The Autoware-ML StreamPETR implementation was ported from the official streampetr
project by exiawsh.

<!-- cspell:ignore Shihao -->
- Repository: <https://github.com/exiawsh/streampetr>
- License: Apache License 2.0
- Paper: Wang, Shihao, et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection" ICCV, 2023.
