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

Experiment configs live under `autoware_ml/configs/tasks/detection3d/streampetr/<dataset>/` and are passed to the CLI as the path relative to `tasks/`. The three leaves are the three stages of the production training flow.

| Leaf                                                                | Database                   | Weights                                               |
| ------------------------------------------------------------------- | -------------------------- | ----------------------------------------------------- |
| `nuscenes/vov99_cpfpn_320x800_b8_30e_nuscenes_54m_pretrain`         | nuScenes                   | RGB-stem DD3D/FCOS3D VoVNet-99 given with `--weights` |
| `t4dataset/vov99_cpfpn_480x640_b8_35e_t4dataset_51m_gen1_gen2_base` | T4Dataset gen1 + gen2 base | Fine-tuned from the stage-1 checkpoint (`--weights`)  |
| `t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base`   | T4Dataset j6gen2           | Fine-tuned from the stage-2 checkpoint (`--weights`)  |

The leaf name spells out its variant: the backbone and neck, the image size, the batch size, the epoch count and the detection range. The configs are layered so that each file owns one concern:

| File                          | Owns                                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `base.yaml`                   | Model architecture and head recipe, image normalization and grid mask, optimizer, LR schedule, trainer settings, ONNX modules |
| `focal2d_head.yaml`           | The auxiliary 2D head block (`model.img_roi_head`), composed by the leaves that train it                                      |
| `<dataset>/base_<range>.yaml` | Detection range, image size, camera order, metrics, partial annotations, transforms and the streaming datamodule              |
| `<dataset>/<leaf>.yaml`       | Database selection and taxonomy, batch size, experiment name, epochs, learning rates and the pretrained weights               |

## Three-Stage Training Flow

The production model is trained natively in three stages; each leaf pins the
exact recipe (GPU count, per-GPU batch, LRs) of the accepted run. There is no
`auto_scale_lr`: for a different total batch size rescale both LRs linearly
(each leaf's header states its base).

| Stage | Leaf                                                                | Database                                      | Epochs | Total batch / LR | Init from                      |
| ----- | ------------------------------------------------------------------- | --------------------------------------------- | ------ | ---------------- | ------------------------------ |
| 1     | `nuscenes/vov99_cpfpn_320x800_b8_30e_nuscenes_54m_pretrain`         | `nuscenes_base`                               | 30     | 32 / 8e-4        | RGB-stem DD3D/FCOS3D VoVNet-99 |
| 2     | `t4dataset/vov99_cpfpn_480x640_b8_35e_t4dataset_51m_gen1_gen2_base` | `t4dataset_gen1_base` + `t4dataset_gen2_base` | 35     | 32 / 2e-4        | stage-1 `best.ckpt`            |
| 3     | `t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base`   | `t4dataset_j6gen2_base`                       | 35     | 64 / 4e-4        | stage-2 `best.ckpt`            |

`--weights` accepts the previous stage's Lightning checkpoint directly, and
stage 2 -> 3 needs nothing else. Every stage trains the same 7 classes, but
stage 1 -> 2 crosses taxonomies whose last two classes are swapped: the
nuScenes taxonomy lists `car, truck, bus, bicycle, pedestrian, traffic_cone,
barrier`, the T4 stages select the `default_dynamic` taxonomy of the database
(the one the other T4 detectors train), which lists `barrier, traffic_cone`.
`--weights` loads by shape, so a stage-1 checkpoint loads into stage 2 without
a warning and the rows of class 5 and 6 of the `bbox_head.cls_branches.*`
final Linear layers (one per decoder layer, 6 in the production head) and of
the `img_roi_head.cls` conv (FocalHead2D) would be swapped silently: the
decoder would start stage 2 scoring barriers as traffic cones. Permute those
rows of the stage-1 checkpoint into the T4 order first (no other tensor
depends on the class order):

```python
import torch

NUSCENES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
T4 = ["car", "truck", "bus", "bicycle", "pedestrian", "barrier", "traffic_cone"]
rows = [NUSCENES.index(name) for name in T4]  # [0, 1, 2, 3, 4, 6, 5]

checkpoint = torch.load("stage1_best.ckpt", map_location="cpu", weights_only=False)
state = checkpoint["state_dict"]
for key, value in list(state.items()):
    is_decoder_cls = key.startswith("bbox_head.cls_branches.") and value.shape[0] == len(T4)
    if is_decoder_cls or key.startswith("img_roi_head.cls."):
        state[key] = value[rows].clone()
torch.save(checkpoint, "stage1_best_t4_order.ckpt")
```

`autoware_ml/models/detection3d/tests/test_streampetr_class_permutation.py`
checks this permutation: the per-class logits of the permuted model are those
of the source model with the columns reordered. Motorcycles train as bicycles,
debris and pushable/pullable objects as barriers, and animals are no targets.
Stage 2 reads the gen1 and the gen2 base corpora as two dataset sources of one
datamodule, both under the same taxonomy. The deployed node's class list
follows the export metainfo order: barrier before traffic_cone for a
T4-trained model.

The T4 stages train `traffic_cone`/`barrier` with partial annotations
(`model.partial_annotations`, as the other T4 detectors): a frame whose
scenario is flagged as annotated without them does not supervise those two
classes in either head, so the detector is not taught they are absent;
annotated frames train them normally. The flag comes from the scenario lists
through the record table (`traffic_cone_barrier_bbox_status`) and the batch
(`gt_traffic_cone_barrier_bbox_status`); the model refuses a batch without
it, so every scenario of the T4 stages has to carry the status. The nuScenes
stage supervises every class (`partial_annotations: null`).

The data is read from the record table of the database (built on first use
under `cache_root_path`) and split by the scenario lists under
`t4dataset_scenario_root_path`. Every dataloader streams the frames of its
scenes in the order of the sample time of the records: the range base composes
`/datamodule/streaming@datamodule: default_streaming_datamodule` after the
database datamodule, which makes it a `StreamingDataModule`. Its sampler joins
the scenes and cuts their frames into one lane per batch position of every
rank, and a lane cut inside a scene starts a new stream there. The lanes
differ by at most one frame, and a shorter lane ends with a padding slot that
repeats its last frame, so every rank runs the same number of batches; the
batch flags those samples (`padding_mask`) and the model gives them no loss,
no metric and no weight in the logged batch size, so every split serves every
frame exactly once. The training split reshuffles the order of the scenes
every epoch from the experiment `seed`.

The `img_backbone` LR is always `lr * 0.1`, set via
`model.optimizer_group_overrides.img_backbone.lr`. The LR schedule is bound
to `trainer.max_epochs`. Validation runs every 5 epochs and then every epoch
over the last 10 (`callbacks.dynamic_validation.epoch_intervals`);
`best.ckpt` is selected by `val/det3d/mAP`.

### Stage 1 — nuScenes pretrain

```bash
autoware-ml train \
    --config-name detection3d/streampetr/nuscenes/vov99_cpfpn_320x800_b8_30e_nuscenes_54m_pretrain \
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

### Stage 2 — T4 gen1 + gen2 base

```bash
autoware-ml train \
    --config-name detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b8_35e_t4dataset_51m_gen1_gen2_base \
    --weights stage1_best_t4_order.ckpt
```

`stage1_best_t4_order.ckpt` is the stage-1 `best.ckpt` with its
classification rows permuted into the T4 class order (see the snippet above);
the raw stage-1 checkpoint loads too, with the cone and barrier rows swapped.
Cross-stage checkpoint loads must report `Loaded matching weight tensors:
894/894` — full coverage, nothing dropped (the nuScenes and T4 stages share the
same 7-class head shapes).

Accepted run: val mAP **0.4420** at epoch 34. That run used the older
5-class base infos (no cone/barrier at all — a 35-epoch supervision gap for
those classes); the T4 base database carries both classes, so a re-run's
numbers are not directly comparable.

### Stage 3 — j6gen2 fine-tune

```bash
autoware-ml train \
    --config-name detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base \
    --weights mlruns/streampetr/T4Dataset-gen1_base-v1.0.0/gen1_gen2_base/vov99_cpfpn_480x640_b8_35e_t4dataset_51m/<run_id>/artifacts/checkpoints/best.ckpt
```

Checkpoints are written to
`mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/`
as `best.ckpt` and `last.ckpt`. For a pipeline validation run add
`+trainer.fast_dev_run=true`. For a short real run instead (e.g.
`trainer.max_epochs=1`) also set
`callbacks.dynamic_validation.epoch_intervals=[[0,1]]`: validation defaults
to every 5 epochs, and a run that finishes without ever validating fails at
teardown. Full-length training is unaffected.

Resuming an interrupted run: `--resume-checkpoint <last.ckpt>` (mutually
exclusive with `--weights`; repeat every other CLI override, they are not
restored from the checkpoint; add `--new-run` only to fork a new MLflow run).

### Evaluate

```bash
autoware-ml test \
    --config-name detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base \
    --weights mlruns/streampetr/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base/vov99_cpfpn_480x640_b16_35e_t4dataset_51m/<run_id>/artifacts/checkpoints/best.ckpt
```

Evaluation runs on a single device by default so every frame is scored.

The headline number is the `0-121m` bucket mAP, directly comparable to AWML
T4MetricV2's `mAP_center_distance_bev` in the `bev_center_0.0-121.0`
evaluator (see the alignment notes below).

### Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base \
    --weights mlruns/streampetr/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base/vov99_cpfpn_480x640_b16_35e_t4dataset_51m/<run_id>/artifacts/checkpoints/best.ckpt \
    --release v1.0.0
```

The export traces the three graphs the runtime node binds
(`extract_img_feat`, `position_embedding`, `pts_head_memory`) from the first
sample of the predict split. `extract_img_feat` takes images normalized with
the ImageNet statistics: the node normalizes the camera images itself, so the
`BatchNormalizeImages` layer of the training preprocessing is not part of the
graph. The current verification scope covers ONNX export. TensorRT engine
generation has not been validated yet.

`pts_head_memory` emits the predictions of the last decoder layer only, the
layer the offline metrics decode: `all_cls_scores` is `(1, 7, 900)` and
`all_bbox_preds` is `(1, 10, 900)`, one column per query (644 learned and 256
propagated). Earlier exports, the AWML one included, stacked all six decoder
layers into 5400 columns, and the node decoded every column, so it published
boxes of the intermediate layers as well. The model package that ships a
graph of this export to the universe node must therefore set
`num_proposals: 900` and retune its per-class score thresholds, which were
tuned on the decoded output of all six layers. The node reads `num_proposals`
from that parameter file; the binding validation of the node stack that follows
autoware.universe #13267 checks it against the engine at start-up, and without
it a mismatch goes unnoticed.

A stream starts from an all-zero memory bank: the graph recognizes a lane
whose bank poses are all zero and seeds it as the head does at the start of a
stream. Device allocations are not zeroed, so the node must clear the five
memory tensors when it allocates them, before the first frame, and on every
reset (autoware.universe #13267 clears them at initialization).

## Results

Final accepted stage-3 run (batch 64 / lr 4e-4, trained on the 2.8 info
generation; an earlier batch-32 / lr 2e-4 run matched within ±0.5 pp on
every headline number). The three stage `best.ckpt` files are archived on
the training host under `~/ckpt_backup/`. These numbers predate the
record-table data pipeline, the removal of the head's `use_bottom_center`
shift, the anti-aliased image resize of the typed transforms, the partial
annotations of the T4 stages, the VoVNet `norm_eval` applied from
construction and the elementwise pose products of the memory bank, so the
stages have to be retrained and re-measured on the current code.

The one-epoch fine-tunes of the converted AWML j6gen2 checkpoint run while
the stack was ported (j6gen2 val mAP 0.4458 at mid-epoch in the first run,
0.4784 and 0.4180 at the end of the epoch in the next two) are not comparable
with later runs either and have to be re-measured. They trained with the
BatchNorm statistics of the backbone updating: VoVNet applied `norm_eval`
only in an explicit `train()` call, which Lightning never makes during fit,
so all 99 BatchNorm layers of VoVNet-99 updated their statistics. Their
validation also aligned the memory bank with matrix products in TF32, the
float32 matrix precision the CLI sets, in which the pose products at batch
size 1 lose tens of meters 65 km from the map origin; the bank now composes
its poses elementwise, in IEEE float32 under any precision setting.

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
gap) and only part of the j6gen2 scenes carry them. The T4 base databases
include both classes, and the partial annotations keep the scenes without
them from punishing their background — the expected fix for these two
classes on the next training round.

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
   and predictions) cannot reproduce. The `default_dynamic` taxonomy therefore
   evaluates every class up to 121 m (a no-op inside the square), and the
   T4 distance buckets mirror T4MetricV2's (0-50 / 50-90 / 90-121 / 0-121 m).

Validation during training streams one lane per rank. The sampler cuts the
joined frames of the split into one lane per rank and ends the shorter lanes
with a padding slot, which the metrics leave out, so `val/det3d/mAP` (the
checkpoint monitor) scores every frame exactly once on any number of GPUs,
like `autoware-ml test`; a lane that starts inside a scene starts a new stream
there.

### Ground truth against the accepted run

The accepted run read AWML-generated info files; the stages now read the
record tables of the database. The `default_dynamic` taxonomy and the database
box pipelines differ from those info files in a few places:

| Ground truth                                                           | Accepted run (info files) | Now (database)                       |
| ---------------------------------------------------------------------- | ------------------------- | ------------------------------------ |
| forklift                                                               | car                       | truck                                |
| trailer without a truck to merge into                                  | not a target              | truck                                |
| construction sign                                                      | not a target              | barrier                              |
| truck–trailer merge distance                                           | 2.0 m                     | 1.0 m                                |
| ground-plane velocity                                                  | as annotated              | clipped at 25 m/s                    |
| two-wheeler with `two_wheel_vehicle_state.without_rider`               | kept                      | dropped                              |
| box recorded with fewer than 2 lidar points (training)                 | kept                      | dropped                              |
| box with a non-finite velocity, a zero extent or a speed above 150 m/s | as annotated              | velocity zeroed, the others dropped  |
| traffic cone / barrier of a scene annotated without them               | background                | not supervised (partial annotations) |

The first three follow the coarse taxonomy every other T4 detector of the
repository trains, the next three are the shared box pipelines and attribute
filters of the T4 dataset configs, the next is the point-count rule of the
evaluation (`min_num_points: 2`) and of the other T4 detectors' training
pipelines, read from the count the database records with every box since the
camera pipelines load no point cloud, and the last two are
`BBoxesPhysicalFilter`, which opens every pipeline, and the partial
annotations of the T4 stages.

## Implementation

| Path                                                                 | Description                                       |
| -------------------------------------------------------------------- | ------------------------------------------------- |
| `autoware_ml/models/detection3d/streampetr.py`                       | StreamPETR model wrapper                          |
| `autoware_ml/models/detection3d/heads/streampetr.py`                 | Query-based detection head                        |
| `autoware_ml/models/detection3d/heads/focal2d.py`                    | Auxiliary 2D head (FocalHead2D)                   |
| `autoware_ml/models/detection3d/task_modules/partial_annotations.py` | Annotated class masks of the partial annotations  |
| `autoware_ml/models/common/backbones/vovnet.py`                      | Multiview image backbone                          |
| `autoware_ml/models/common/multiview_image_encoder.py`               | Per-camera image encoder shared with BEVFusion    |
| `autoware_ml/models/common/necks/cp_fpn.py`                          | CPFPN neck (reference StreamPETR neck)            |
| `autoware_ml/models/detection3d/task_modules/`                       | Shared assigners, costs, coders, streaming memory |
| `autoware_ml/datamodule/streaming_data_module.py`                    | `StreamingDataModule`, the streaming datamodule   |
| `autoware_ml/datamodule/samplers.py`                                 | `GroupStreamingSampler`, the scene lanes          |
| `autoware_ml/datamodule/base_dataset.py`                             | Scene grouping of the record table                |
| `autoware_ml/dataclasses/batch/frame_meta.py`                        | Per-frame pose, scene and sample time             |
| `autoware_ml/dataclasses/batch/detection2d.py`                       | 2D boxes padded to `max_num_2d_gt_bboxes`         |
| `autoware_ml/transforms/camera/annotations2d.py`                     | 2D targets projected from the 3D boxes            |
| `autoware_ml/transforms/camera/order.py`                             | `ShuffleCameraOrder`, T4 training only            |
| `autoware_ml/transforms/camera/resize.py`                            | `ResizeCropFlipRotImage`, the image augmentation  |
| `autoware_ml/transforms/camera_lidar/geometry.py`                    | `GlobalRotScaleTrans`, camera-only here           |
| `autoware_ml/preprocessing/camera/normalize.py`                      | Batched image normalization                       |
| `autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py`           | Warmup + epoch-cosine LR schedule                 |
| `autoware_ml/configs/tasks/detection3d/streampetr/`                  | Task configurations                               |
| `autoware_ml/configs/database/t4dataset/taxonomy/`                   | `default_dynamic` taxonomy of the T4 stages       |

An AWML/mmdet3d checkpoint converter
(`autoware_ml/tools/convert_streampetr_checkpoint.py`) existed during the
migration and was removed once the native training flow replaced converted
checkpoints; recover it from git history if an mm-style checkpoint ever needs
importing again. Its output predates the image feature extractor, so prefix
its `img_backbone.*` and `img_neck.*` keys with `image_feature_extractor.`:
under the old names only the heads match, and `--weights` leaves the backbone
and the neck randomly initialized without an error.

Test a change with the whole suite and no `-k` filter:
`pixi run --environment dev test` runs `pytest autoware_ml/`, and its
BEVFusion and TransFusion tests need the `bev_pool` and `diff_iou_rotated`
extensions that `setup-project` builds into the tree. The StreamPETR tests
are spread over the model and head, the task modules, the datamodule and its
sampler, the camera and camera-lidar transforms, the image preprocessing and
the configs, and a filter drops whole directories of them without a trace:
`-k "not camera_lidar"` skips all of `transforms/camera_lidar/`, where the
camera-only augmentation is tested.

## Acknowledgment

<!-- cspell:ignore exiawsh -->
The Autoware-ML StreamPETR implementation was ported from the official streampetr
project by exiawsh.

<!-- cspell:ignore Shihao -->
- Repository: <https://github.com/exiawsh/streampetr>
- License: Apache License 2.0
- Paper: Wang, Shihao, et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection" ICCV, 2023.
