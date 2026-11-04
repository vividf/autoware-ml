---
icon: lucide/scan-search
---

# CenterPoint

CenterPoint is a LiDAR-based 3D object detection model integrated under the `detection3d` task namespace. It uses a PointPillars-style frontend with a `SECOND` backbone, `SECONDFPN` neck, and CenterPoint detection head.

## Summary

| Property     | Value                                    |
| ------------ | ---------------------------------------- |
| Task         | 3D object detection                      |
| Modality     | LiDAR                                    |
| Input        | Point cloud                              |
| Output       | 3D bounding boxes and class scores       |
| Architecture | PointPillars + SECOND + SECONDFPN + head |
| Datasets     | T4Dataset, nuScenes                      |

## Available Configurations

Experiment configs live under `autoware_ml/configs/tasks/detection3d/centerpoint/<dataset>/` and are passed to the CLI as the path relative to `tasks/`.

| Leaf                                  | Database          | Weights                                             |
| ------------------------------------- | ----------------- | --------------------------------------------------- |
| `t4dataset/<variant>_gen1_base`       | T4Dataset gen1    | Trained from scratch                                |
| `t4dataset/<variant>_jpntaxi_base`    | T4Dataset jpntaxi | Fine-tuned from a checkpoint given with `--weights` |
| `t4dataset/<variant>_j6gen2_base`     | T4Dataset j6gen2  | Fine-tuned from a checkpoint given with `--weights` |
| `nuscenes/<variant>_nuscenes_<range>` | nuScenes          | Trained from scratch                                |

The leaf name spells out its variant: the voxel size, the backbone and neck, the batch size, the epoch count and the detection range, so the values live in one place, the config. The configs are layered so that each file owns one concern:

| File                                  | Owns                                                                                          |
| ------------------------------------- | --------------------------------------------------------------------------------------------- |
| `base.yaml`                           | Model architecture, pillar preprocessor, optimizer, trainer settings, and ONNX export modules |
| `<dataset>/base_<range>.yaml`         | Point cloud range, evaluation ranges, NMS radius, dataloaders, and the transform pipeline     |
| `<dataset>/base_<voxel>_<range>.yaml` | Voxel size, BEV canvas shape, backbone strides, and head output stride                        |
| `<dataset>/<voxel>_<variant>.yaml`    | Database selection, batch size, experiment name, epochs, and optional pretrained weights      |

## Training

Train the base model from scratch:

```bash
autoware-ml train --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_30e_t4dataset_120m_gen1_base
```

The `jpntaxi` and `j6gen2` configs declare `weights: ???`, so they must be started from a pretrained checkpoint. Pass it with `--weights`. The config already lowers the learning rate for fine-tuning:

```bash
autoware-ml train \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/centerpoint/T4Dataset-gen1_base-v1.0.0/gen1_base/voxel024_second_secfpn_b16_30e_t4dataset_120m/<run_id>/artifacts/checkpoints/best.ckpt
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_30e_t4dataset_120m_gen1_base \
    +trainer.fast_dev_run=true
```

Checkpoints are written to `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/` as `best.ckpt` and `last.ckpt`. Resume an interrupted run with `--resume-checkpoint` pointing at `last.ckpt`.

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/centerpoint/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base/voxel024_second_secfpn_b16_50e_t4dataset_120m/<run_id>/artifacts/checkpoints/best.ckpt
```

Evaluation runs on a single device by default so metrics are deterministic. Pass `--use-config-devices` to keep the `trainer.devices` from the config.

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/centerpoint/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base/voxel024_second_secfpn_b16_50e_t4dataset_120m/<run_id>/artifacts/checkpoints/best.ckpt \
    --release v1.0.0
```

The export produces the two ONNX modules consumed by `autoware_universe/perception/autoware_lidar_centerpoint`: `pts_voxel_encoder_centerpoint.onnx` encodes decorated pillar features into per-pillar descriptors, and `pts_backbone_neck_head_centerpoint.onnx` predicts the raw dense detection heads (`heatmap`, `reg`, `height`, `dim`, `rot`, `vel`) from the scattered BEV canvas. Voxelization, pillar decoration, BEV scatter, and box decoding all run in the runtime node.

Exports land in `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/exports/` of a dedicated deploy run linked to the training run. Every module is stamped with its provenance and the `--release` version. Omit `--release` only for throwaway exports, which are stamped `unversioned`. TensorRT engine generation is disabled in the base config (`deploy.tensorrt.enabled=false`) because the runtime builds engines itself.

### Verification and evaluation

CenterPoint declares its deployment as a stage graph (`CenterPointDetection3DModel.build_stages`),
so the export can be checked against the PyTorch model and scored per backend:

```bash
autoware-ml deploy \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights <best.ckpt> \
    deploy.onnx.precision=fp16 deploy.tensorrt.enabled=true \
    deploy.verification.enabled=true deploy.evaluation.enabled=true deploy.evaluation.num_samples=100
```

Verification compares the raw head outputs of `onnx` and `tensorrt` against `pytorch` on the
first predict batch (gates are measured values, see the base config); evaluation scores every
backend on the test split with the model's metric suites and logs a per-stage latency table
(`model_graphs` is the TensorRT time of the two engines). See
[Deployment](../user-guide/deployment.md#stage-graph-verification-and-evaluation).

### INT8

The `_int8` variant quantizes the SECOND backbone (except its first stage), the FPN neck and the
dense head with explicit Q/DQ; the pillar feature net stays fp16. Post-training quantization
calibrates on 400 validation samples and writes a self-describing checkpoint that `deploy` and
`test` take like any other `--weights`:

```bash
autoware-ml quantize \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base_int8 \
    --weights <best.ckpt>
autoware-ml deploy \
    --config-name detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base_int8 \
    --weights <checkpoints/ptq.ckpt of the quantize run>
```

`quantization.dry_run=true` prints the precision placement (which module gets which
transform and why) without weights or a GPU. Engines build strongly typed: INT8 comes from the
Q/DQ nodes in the ONNX, the un-quantized stages run fp16, and the graph I/O stays fp32.

## Implementation

| Path                                                                      | Description                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------ |
| `autoware_ml/models/detection3d/main_modules/centerpoint.py`              | CenterPoint model wrapper                        |
| `autoware_ml/models/detection3d/encoders/pillars/pillar_feature_net.py`   | Pillar feature encoder                           |
| `autoware_ml/models/detection3d/encoders/pillars/point_pillar_scatter.py` | Pillar scatter to BEV                            |
| `autoware_ml/models/detection3d/backbones/second.py`                      | SECOND backbone                                  |
| `autoware_ml/models/detection3d/necks/second_fpn.py`                      | SECONDFPN neck                                   |
| `autoware_ml/models/detection3d/heads/centerhead.py`                      | CenterPoint detection head                       |
| `autoware_ml/preprocessing/detection3d/point_pillar_preprocessor.py`      | Pillar preprocessing                             |
| `autoware_ml/datamodule/t4dataset/dataset.py`                             | T4Dataset records as a dataset                   |
| `autoware_ml/datamodule/nuscenes/dataset.py`                              | nuScenes dataset                                 |
| `autoware_ml/datamodule/common/detection3d.py`                            | 3D boxes of a record                             |
| `autoware_ml/configs/datamodule/`                                         | Datamodule, dataset and transform configurations |
| `autoware_ml/configs/tasks/detection3d/centerpoint/`                      | Experiment configurations                        |
| `autoware_ml/quantization/`                                               | PTQ engine and plan                              |

## Acknowledgment

The Autoware-ML CenterPoint implementation was ported from the official mmdetection3d
project by OpenMMLab.

<!-- cspell:ignore Zhijian -->
- Repository: <https://github.com/open-mmlab/mmdetection3d>
- License: Apache License 2.0
- Paper: Yin, Tianwei, et al. "Center-based 3D Object Detection and Tracking" CVPR, 2021.
