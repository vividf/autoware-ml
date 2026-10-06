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
| Datasets     | NuScenes, T4Dataset                      |

## Available Configurations

| Config Name                                                            | Dataset   | Purpose                                                  |
| ---------------------------------------------------------------------- | --------- | -------------------------------------------------------- |
| `detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes`          | NuScenes  | Standard NuScenes 51 m configuration                     |
| `detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2` | T4Dataset | 120 m T4Dataset configuration (aligned with TransFusion) |

## Training

```bash
autoware-ml train --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes
autoware-ml train --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes \
    +trainer.fast_dev_run=true
```

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes \
    --weights mlruns/detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2 \
    --weights mlruns/detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt
```

The export produces the two ONNX modules consumed by `autoware_universe/perception/autoware_lidar_centerpoint`: `pts_voxel_encoder_centerpoint.onnx` encodes decorated pillar features into per-pillar descriptors, and `pts_backbone_neck_head_centerpoint.onnx` predicts the raw dense detection heads (`heatmap`, `reg`, `height`, `dim`, `rot`, `vel`) from the scattered BEV canvas. Voxelization, pillar decoration, BEV scatter, and box decoding all run in the runtime node.

### Verification and evaluation

CenterPoint declares its deployment as a stage graph (`CenterPointDetectionModel.build_stages`),
so the export can be checked against the PyTorch model and scored per backend:

```bash
autoware-ml deploy \
    --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2 \
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
    --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2_int8 \
    --weights <best.ckpt>
autoware-ml deploy \
    --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2_int8 \
    --weights <quantized/ptq.ckpt>
```

`quantization.dry_run=true` prints the precision placement (which module gets which
transform and why) without weights or a GPU. Engines build strongly typed: INT8 comes from the
Q/DQ nodes in the ONNX, the un-quantized stages run fp16, and the graph I/O stays fp32.

## Implementation

| Path                                                    | Description                |
| ------------------------------------------------------- | -------------------------- |
| `autoware_ml/models/detection3d/centerpoint.py`         | CenterPoint model wrapper  |
| `autoware_ml/models/detection3d/encoders/pillar.py`     | Pillar encoder and scatter |
| `autoware_ml/models/detection3d/backbones/second.py`    | SECOND backbone            |
| `autoware_ml/models/detection3d/necks/second_fpn.py`    | SECONDFPN neck             |
| `autoware_ml/models/detection3d/heads/centerpoint.py`   | CenterPoint detection head |
| `autoware_ml/preprocessing/detection3d/point_pillar.py` | Pillar preprocessing       |
| `autoware_ml/datamodule/nuscenes/detection3d.py`        | NuScenes datamodule        |
| `autoware_ml/datamodule/t4dataset/detection3d.py`       | T4Dataset datamodule       |
| `autoware_ml/configs/tasks/detection3d/centerpoint/`    | Task configurations        |
| `autoware_ml/quantization/`                             | PTQ engine and plan        |

## Acknowledgment

The Autoware-ML CenterPoint implementation was ported from the official mmdetection3d
project by OpenMMLab.

<!-- cspell:ignore Zhijian -->
- Repository: <https://github.com/open-mmlab/mmdetection3d>
- License: Apache License 2.0
- Paper: Yin, Tianwei, et al. "Center-based 3D Object Detection and Tracking" CVPR, 2021.
