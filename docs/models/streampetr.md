---
icon: lucide/cctv
---

# StreamPETR

<!-- cspell:ignore CPFPN kokseang fcos3d imgbackbone md5sum -->

StreamPETR is a camera-based 3D object detection model integrated under the `detection3d` task namespace. It uses a multiview image backbone, a feature pyramid neck, and a native query-based detection head with a streaming memory queue for temporal modeling.

## Summary

| Property     | Value                                       |
|--------------|---------------------------------------------|
| Task         | 3D object detection                         |
| Modality     | Camera                                      |
| Input        | Synchronized multiview images               |
| Output       | 3D bounding boxes and class scores          |
| Architecture | Multiview VoVNet/CPFPN + query decoder head |
| Datasets     | NuScenes, T4Dataset                         |

## Available Configurations

| Config Name                                                                 | Dataset   | Purpose                                                            |
|-----------------------------------------------------------------------------|-----------|--------------------------------------------------------------------|
| `detection3d/streampetr/vov_320x800_nuscenes`                               | NuScenes  | NuScenes configuration                                             |
| `detection3d/streampetr/vov_320x800_nuscenes_pretrain`                      | NuScenes  | Three-stage flow, stage 1: AWML nuScenes pretrain recipe           |
| `detection3d/streampetr/vov_480x640_t4dataset_base`                         | T4Dataset | Three-stage flow, stage 2: full T4 base DB, 35 epochs              |
| `detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune`              | T4Dataset | Three-stage flow, stage 3: j6gen2 fine-tune from the base stage    |
| `detection3d/streampetr/vov_480x640_t4dataset_j6gen2`                       | T4Dataset | Default T4 recipe, reproduces the AWML j6gen2 production recipe    |
| `detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier` | T4Dataset | Fine-tune from an AWML checkpoint with cone/barrier partial-ignore |

The T4 default mirrors the AWML recipe
`t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_j6gen2_partialignore.py`
end to end: 2 GPUs x batch 8 (total 16), 10 epochs, AdamW lr 1.0e-4 with
`img_backbone` lr_mult 0.1, 500-iteration warmup into per-epoch cosine decay,
pc_range ±51.2 m, full train-time augmentation (resize/flip, global
rot/scale, per-frame camera shuffling, grid mask), an auxiliary 2D
`FocalHead2D`, `traffic_cone`/`barrier` partial-ignore, seed 0, and
mAP-based checkpoint selection. There is no `auto_scale_lr`: the config pins
`trainer.devices: 2`; for any other total batch size N rescale both LRs by
N/16 via the optimizer overrides, e.g. for 4 GPUs x batch 8 (total 32):

```bash
batch_size=8 trainer.devices=4 \
    model.optimizer.lr=2.0e-4 \
    model.optimizer_group_overrides.img_backbone.lr=2.0e-5
```

## Training Workflow (T4 / j6gen2)

### 1. Pretrained initialization

The recipe initializes from the nuScenes model-zoo pretrain. Download and
convert it once (the conversion maps AWML/mmdet3d weight names to the
autoware-ml module tree; the converted checkpoint is byte-identical on all
880 shared tensors):

```bash
wget https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/streampetr/streampetr-vov99/nuscenes/v1.0/nuscenes_vov99_baseline_320x800.pth
python -m autoware_ml.tools.convert_streampetr_checkpoint \
    --input nuscenes_vov99_baseline_320x800.pth \
    --output nuscenes_vov99_baseline_320x800_converted.pth \
    --bgr-to-rgb \
    --drop-pattern 'cls_branches\.\d+\.6\.' \
    --drop-pattern 'img_roi_head\.cls\.'
```

The `--drop-pattern` flags strip the 10-class nuScenes classification layers;
the 7-class T4 layers start from their focal-prior initialization, exactly
like AWML's `strict=False` load.

### 2. Train

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights nuscenes_vov99_baseline_320x800_converted.pth \
    datamodule.data_root=<data_root> \
    datamodule.train_ann_file=<infos_train.pkl> \
    datamodule.val_ann_file=<infos_val.pkl> \
    datamodule.test_ann_file=<infos_test.pkl>
```

For a pipeline validation run add `+trainer.fast_dev_run=true`. For a short
real run instead (e.g. `trainer.max_epochs=1`) also set
`trainer.check_val_every_n_epoch=1`: validation defaults to every 5 epochs,
and a run that finishes without ever validating fails at teardown when the
`val/loss` metric is missing. Full-length training is unaffected.

### 3. Evaluate

```bash
autoware-ml test \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt \
    trainer.devices=1
```

The headline number is the `0-121m` bucket mAP, directly comparable to AWML
T4MetricV2's `mAP_center_distance_bev` in the `bev_center_0.0-121.0`
evaluator (see the alignment notes below).

### 4. Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2 \
    --weights mlruns/.../checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

The current verification scope covers ONNX export. TensorRT engine
generation has not been validated yet.

## Three-Stage Training Flow (AWML production flow)

The workflow above is the single-stage recipe (nuScenes pretrain → j6gen2).
AWML's production flow inserts a full T4 base-DB stage in between:

| Stage | Config                                                         | Data                             | Epochs | Init from                              |
|-------|----------------------------------------------------------------|----------------------------------|--------|----------------------------------------|
| 1     | `detection3d/streampetr/vov_320x800_nuscenes_pretrain`         | nuScenes                         | 30     | DD3D/FCOS3D VoVNet-99 backbone         |
| 2     | `detection3d/streampetr/vov_480x640_t4dataset_base`            | `t4dataset_base_infos_*`         | 35     | stage-1 checkpoint                     |
| 3     | `detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune` | `t4dataset_j6gen2_base_infos_*`  | 35     | stage-2 checkpoint                     |

These mirror, in order, AWML's `nuscenes_vov_flash_320x800_baseline.py`,
`t4_base_vov_flash_480x640_bev_2_8_traffic_barrier_base_partialignore.py`, and
`t4_base_vov_flash_480x640_bev_2_7_traffic_barrier_j6gen2_partialignore.py`
(whose `load_from` is the stage-2 run). All three pin `trainer.devices: 2` so
the total batch stays 16 and the hard-coded LRs hold; for any other total
batch size N rescale both LRs by N/16 with the same
`model.optimizer.lr` / `model.optimizer_group_overrides.img_backbone.lr`
overrides shown above (stage 1's base LRs are 4.0e-4 / 4.0e-5, stages 2 and 3
use 1.0e-4 / 1.0e-5). Validation runs every 5 epochs
(AWML's `val_interval`); Lightning has no equivalent of AWML's
`dynamic_intervals`, so the last five epochs are not validated individually.

### Stage 1 — nuScenes pretrain

In practice this stage is *not* re-run: AWML's T4 configs start from the
published model-zoo artifact, so use the converted checkpoint from
[step 1 of the workflow above](#1-pretrained-initialization) and go straight
to stage 2. `vov_320x800_nuscenes_pretrain` exists for the case where the
pretrain itself must be reproduced inside autoware-ml — it swaps the neck to
CPFPN and adds the auxiliary `FocalHead2D` so its weights are compatible with
the T4 stages, and applies AWML's nuScenes recipe (lr 4e-4, grad-clip 35,
`eta_min = lr * 1e-3`, random flip + global rot/scale). It additionally needs
the nuScenes dataset mounted and AWML's DD3D/FCOS3D-pretrained VoVNet-99
backbone (`fcos3d_vovnet_imgbackbone-remapped.pth`) as `--weights`; without
that backbone init the recipe underperforms.

Download the backbone from the upstream StreamPETR release and flip its stem
from BGR to RGB once (expected output: `Converted 626 tensors; skipped 81`):

```bash
curl -L -o fcos3d_vovnet_imgbackbone-remapped.pth \
    https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth
md5sum fcos3d_vovnet_imgbackbone-remapped.pth
# ff1ac3040eabf0f0e54c3c594c26021e
python -m autoware_ml.tools.convert_streampetr_checkpoint \
    --input fcos3d_vovnet_imgbackbone-remapped.pth \
    --output fcos3d_vovnet_imgbackbone-remapped_converted.pth \
    --bgr-to-rgb
```

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_320x800_nuscenes_pretrain \
    --weights fcos3d_vovnet_imgbackbone-remapped_converted.pth \
    datamodule.data_root=<nuscenes_root>
```

The training log should report `Loaded matching weight tensors: 626/1526
(+626 shared-tensor aliases)` for this backbone init. Reference points for
the stage: AWML's published training logs
(<https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/streampetr/streampetr-vov99/nuscenes/v1.0/logs.zip>)
for the loss trend, and the published 10-class nuScenes result (mAP 0.4697)
as the acceptance bar.

### Stage 2 — T4 base DB

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_base \
    --weights nuscenes_vov99_baseline_320x800_converted.pth \
    datamodule.data_root=<t4_data_root>
```

The config defaults to `info/detection3d/t4dataset_base_infos_{train,val,test}.pkl`
under `data_root`; override `datamodule.{train,val,test}_ann_file` for a
different info directory. When loading the stage-1 checkpoint (and likewise
the stage-2 checkpoint in stage 3) the log should report
`Loaded matching weight tensors: 1526/1526 (+0 shared-tensor aliases)` —
full coverage, nothing dropped.

### Stage 3 — j6gen2 fine-tune

```bash
autoware-ml train \
    --config-name detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune \
    --weights mlruns/detection3d/streampetr/vov_480x640_t4dataset_base/<run_id>/artifacts/checkpoints/best.ckpt \
    datamodule.data_root=<t4_data_root>
```

`--weights` accepts the stage-2 Lightning checkpoint directly — no conversion
is needed between autoware-ml stages. Evaluate and deploy the stage-3
checkpoint exactly as in steps 3 and 4 above.

## Results (AWML parity, verified 2026-08-06)

Both frameworks trained the same recipe on the same j6gen2 data split
(kokseang_2_8 infos) from the same converted pretrain, and were scored with
the aligned evaluation (full ±51.2 m square GT, min-points filter engaged):

| Framework / checkpoint                      | Training               | val mAP     | test mAP    |
|---------------------------------------------|------------------------|-------------|-------------|
| **autoware-ml** (this recipe, run 92068f7b) | bf16, global loss norm | **0.39127** | **0.36609** |
| AWML (aligned_bf16, epoch 9)                | bf16                   | 0.37521     | 0.35515     |

Cross-checks that back these numbers:

- **Metric stacks are equivalent**: scoring an identical set of predictions
  and GT with AWML T4MetricV2 and with autoware-ml `MeanAP` agrees to
  ≤ 2.5e-8 per class (pure float32 round-trip noise).
- **Same-weights residual**: running one AWML checkpoint through both
  inference stacks leaves −0.80 mAP (test), consistent in sign across all 7
  classes — numerics of the camera pipeline (image decode/resize, attention
  kernels, fp16-vs-fp32 paths), not an evaluator or recipe difference.
- **Training-parity checklist**: identical pretrained init (880/880 shared
  tensors byte-equal), per-epoch LR schedule matches AWML's logged values
  (< 0.5 %), batch/optimizer/grad-clip/hooks equal.

## Evaluation Alignment Notes

Two evaluation-side fixes were required to make mAP comparable with AWML
T4MetricV2; both are part of the default config:

1. **`gt_num_points` collation** — without it the evaluation-time min-points
   GT filter silently never engages.
2. **No radial `eval_class_range` cap inside the pc_range square** — the GT
   is already limited to the ±51.2 m *square* by the pipeline
   `ObjectRangeFilter`. An additional radial cap at 51.2/54 m removed the
   square's corners from the GT while corner *predictions* stayed and became
   guaranteed false positives — a semantics T4MetricV2 (radial on both GT
   and predictions) cannot reproduce. The config therefore restates the
   121 m dataset default (a no-op inside the square) and mirrors
   T4MetricV2's distance buckets (0-50 / 50-90 / 90-121 / 0-121 m).

## Implementation

| Path                                                        | Description                                          |
|-------------------------------------------------------------|------------------------------------------------------|
| `autoware_ml/models/detection3d/streampetr.py`              | StreamPETR model wrapper                             |
| `autoware_ml/models/detection3d/heads/streampetr.py`        | Query-based detection head                           |
| `autoware_ml/models/detection3d/heads/focal2d.py`           | Auxiliary 2D head (FocalHead2D)                      |
| `autoware_ml/models/detection3d/partial_ignore.py`          | Partial-ignore label handling                        |
| `autoware_ml/models/common/backbones/vovnet.py`             | Multiview image backbone                             |
| `autoware_ml/models/common/necks/cp_fpn.py`                 | CPFPN neck (weight-compatible with AWML checkpoints) |
| `autoware_ml/models/detection3d/task_modules/`              | Shared assigners, costs, coders, streaming memory    |
| `autoware_ml/datamodule/common/multiview_detection3d.py`    | Shared multiview detection dataset                   |
| `autoware_ml/datamodule/nuscenes/multiview_detection3d.py`  | NuScenes multiview datamodule                        |
| `autoware_ml/datamodule/t4dataset/multiview_detection3d.py` | T4Dataset multiview datamodule                       |
| `autoware_ml/utils/schedulers/iter_warmup_epoch_cosine.py`  | Warmup + epoch-cosine LR schedule                    |
| `autoware_ml/tools/convert_streampetr_checkpoint.py`        | AWML/model-zoo checkpoint converter                  |
| `autoware_ml/configs/tasks/detection3d/streampetr/`         | Task configurations                                  |

## Acknowledgment

<!-- cspell:ignore exiawsh -->
The Autoware-ML StreamPETR implementation was ported from the official streampetr
project by exiawsh.

<!-- cspell:ignore Shihao -->
- Repository: <https://github.com/exiawsh/streampetr>
- License: Apache License 2.0
- Paper: Wang, Shihao, et al. "Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection" ICCV, 2023.
