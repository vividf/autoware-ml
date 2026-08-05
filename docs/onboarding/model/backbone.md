# Backbones (and Encoders)

> **What this covers:** the "feature extractor" stages of a model — encoders that turn raw
> points into a grid, and backbones that extract multi-scale features from that grid.
> Prerequisite: [model_architecture.md](model_architecture.md).

---

## 1. Terminology in this repo

For LiDAR detection the front of the network has **three** distinct stages, and the repo names
them precisely (don't conflate them):

| Stage | Folder | Job | Example |
| ----- | ------ | --- | ------- |
| **Voxel encoder** | `models/detection3d/encoders/` | per-voxel/pillar point features → one vector per voxel | `PillarFeatureNet` |
| **Middle encoder** | `models/detection3d/encoders/` | scatter/convolve voxels into a dense BEV grid | `PointPillarsScatter`, `SparseEncoder` |
| **Backbone** | `models/detection3d/backbones/` | 2D CNN over the dense grid → multi-scale features | `SECONDBackbone` |

The "backbone" is the 2D CNN; the point→grid conversion is an "encoder". This matters when you
read `CenterPointDetectionModel.forward`: `pts_voxel_encoder` and `pts_middle_encoder` come
*before* `pts_backbone`.

---

## 2. `SECONDBackbone` — the canonical LiDAR backbone (`backbones/second.py:30`)

```python
class SECONDBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, layer_nums, layer_strides):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for stage_channels, num_layers, stride in zip(out_channels, layer_nums, layer_strides):
            layers = [ConvModule(current_channels, stage_channels, stride=stride)]     # downsample
            layers.extend(ConvModule(stage_channels, stage_channels) for _ in range(num_layers))
            blocks.append(nn.Sequential(*layers))
            current_channels = stage_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x) -> list[torch.Tensor]:
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)          # one feature map per stage → multi-scale
        return outputs
```

Configured (CenterPoint base):

```yaml
pts_backbone:
  _target_: autoware_ml.models.detection3d.backbones.second.SECONDBackbone
  in_channels: 32
  out_channels: [64, 128, 256]     # 3 stages
  layer_nums: [3, 5, 5]            # residual convs per stage
  layer_strides: [2, 2, 2]         # each stage halves resolution
```

So it takes a `(B, 32, H, W)` BEV grid and returns three maps at `1/2, 1/4, 1/8` resolution
with `64, 128, 256` channels. The neck ([neck.md](neck.md)) fuses them back to one map.

Every conv is a **`ConvModule`** (`models/common/layers/conv.py`) — the shared conv + norm +
activation block, reused across backbone, neck, and head. Learn `ConvModule` once and you can
read all of them.

---

## 3. The voxel/middle encoders (the point→grid front)

- **`PillarFeatureNet`** (`encoders/pillar.py`) — the "voxel encoder". Takes `voxels`,
  `num_points`, `voxel_coords` (produced by `PointPillarPreprocessor`), *decorates* each point
  with offsets to its pillar's center, runs a small PFN MLP, and pools to one feature vector
  per pillar. It exposes `decorate(...)` and `encode_decorated(...)` separately so deployment
  can export just the MLP (see [../deployment/export_pipeline.md](../deployment/export_pipeline.md)).
- **`PointPillarsScatter`** (`encoders/pillar.py`) — the "middle encoder" for pillars. Scatters
  the per-pillar vectors back onto a dense `(B, C, H, W)` BEV canvas using `voxel_coords`.
- **`SparseEncoder` / `SparseBasicBlock`** (`encoders/sparse.py`) — the voxel-based alternative
  (3D sparse convolutions via the external `spconv`), used by TransFusion/BEVFusion-style
  voxel detectors instead of pillar-scatter. Sparse ops relate to the custom ops in
  `autoware_ml/ops/spconv/` for deployment.

---

## 4. Camera backbones (shared, `models/common/backbones/`)

For camera and fusion models the backbone is an image CNN:

- **`ResNet18/50` and multi-scale variants** (`common/backbones/resnet.py`) — standard image
  feature extractors; multi-scale variants emit an FPN-style pyramid.
- **`VoVNet` / `VoVNet99` multi-scale** (`common/backbones/vovnet.py`) — the heavier image
  backbone used by camera 3D detectors (e.g. StreamPETR-style).

These feed camera necks (`CPFPN`, `GeneralizedLSSFPN`) and then a view transform
(`view_transforms/`) that lifts image features into BEV for fusion.

---

## 5. Point-transformer backbone (PTv3)

`segmentation3d/encoders/` contains `PointTransformerV3Encoder` — a serialized, attention-based
point backbone shared by the PTv3 segmentation, detection, and multi-task models via
`PTv3BaseModel`. Unlike the pillar/voxel path, it operates directly on concatenated points
(hence PTv3 datamodules use `concat`/`index_concat` collation and `batch["offset"]`).

---

## 6. How a backbone plugs in

A backbone is just a sub-module `_target_` in the model config; Hydra builds it and hands the
instance to the model constructor. To swap backbones, change the `_target_` and its args — no
Python change is needed as long as the input/output tensor contract matches (dense BEV grid in,
list of feature maps out).

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| Channel mismatch into the neck | `pts_backbone.out_channels` ≠ `pts_neck.in_channels` | keep them consistent in config |
| Wrong BEV grid size / `output_shape` error | `voxel_size`/`point_cloud_range` inconsistent with `pts_middle_encoder.output_shape` | recompute grid = range / voxel_size; fill `output_shape` |
| OOM in backbone | too-fine voxel grid or too many channels | coarsen `voxel_size`, reduce `out_channels`, or reduce batch size |
| `spconv` import error | sparse encoder without the external package | install `spconv-cu*` (already pinned) or use a pillar model |
| Camera backbone shape errors | image resize/crop transforms not matching backbone expectations | align `camera/` resize transforms with the backbone |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Deeper/wider backbone | tune `out_channels` / `layer_nums` in config |
| Switch pillar → voxel (sparse) | swap `pts_voxel_encoder`+`pts_middle_encoder` to sparse encoders; adjust preprocessing |
| Use a different image backbone | change `common/backbones/*` `_target_` (ResNet ↔ VoVNet) |
| Add a new backbone | add a `nn.Module` under `backbones/` returning a list of maps; reference by `_target_` |

---

**Next:** [neck.md](neck.md) — fusing the backbone's multi-scale outputs.
