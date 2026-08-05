# Necks

> **What this covers:** the neck stage — the small module between backbone and head that
> aggregates multi-scale features into the single feature map the head consumes.
> Prerequisite: [backbone.md](backbone.md).

---

## 1. Why a neck exists

A backbone emits features at several resolutions (e.g. `1/2, 1/4, 1/8`). The head wants **one**
feature map at a fixed resolution/channel count. The neck bridges that gap: it upsamples the
coarse maps, aligns resolutions, and merges them. It is deliberately small — a place to fuse,
not to add depth.

---

## 2. `SECONDFPN` — the canonical LiDAR neck (`necks/second_fpn.py:31`)

```python
class SECONDFPN(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_strides):
        super().__init__()
        blocks = []
        for input_channels, output_channels, stride in zip(in_channels, out_channels, upsample_strides):
            if stride >= 1:
                blocks.append(ConvModule(input_channels, output_channels, stride=int(stride), transpose=True))  # deconv upsample
            else:
                blocks.append(nn.Sequential(     # stride < 1 → downsample via strided conv
                    nn.Conv2d(input_channels, output_channels, kernel_size=int(round(1/stride)), stride=int(round(1/stride)), bias=False),
                    nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        upsampled = [block(feature) for block, feature in zip(self.blocks, x)]
        return torch.cat(upsampled, dim=1)     # bring all stages to one resolution, concat on channels
```

Configured (CenterPoint base):

```yaml
pts_neck:
  _target_: autoware_ml.models.detection3d.necks.second_fpn.SECONDFPN
  in_channels: [64, 128, 256]        # matches SECONDBackbone.out_channels
  out_channels: [128, 128, 128]      # each stage → 128 channels
  upsample_strides: [0.5, 1, 2]      # align the three stages to a common resolution
```

The output is a `(B, 384, H', W')` tensor (`128×3` concatenated) — exactly the `in_channels:
384` the `CenterHead` expects. **The neck's `in_channels` must equal the backbone's
`out_channels`, and the head's `in_channels` must equal the neck's total output channels.**
This three-way channel contract is the most common source of shape errors.

Each block is again a `ConvModule` (with `transpose=True` for deconvolution upsampling) — the
same shared block used everywhere.

---

## 3. Other necks (`models/common/necks/`)

Necks are shared across tasks:

| Neck | File | Use |
| ---- | ---- | --- |
| `SECONDFPN` | `detection3d/necks/second_fpn.py` | LiDAR BEV detection (CenterPoint, pillar/voxel detectors) |
| `CPFPN` | `common/necks/cp_fpn.py` | camera feature pyramid (StreamPETR-style camera detectors) |
| `GeneralizedLSSFPN` | `common/necks/lss_fpn.py` | camera FPN feeding an LSS view transform (BEVFusion camera branch) |
| `GlobalAveragePooling` | `common/necks/global_average_pooling.py` | classification necks (collapse spatial dims) |

The camera necks (`CPFPN`, `GeneralizedLSSFPN`) sit between an image backbone and a view
transform (`models/detection3d/view_transforms/`) that lifts 2D image features into the BEV
space where they can be fused with LiDAR.

---

## 4. How a neck plugs in

Like the backbone, the neck is a `_target_` sub-module. `CenterPointDetectionModel.forward`
calls it directly:

```python
bev_features = self.pts_backbone(bev_features)   # list of maps
bev_features = self.pts_neck(bev_features)        # one fused tensor
return self.bbox_head(bev_features)
```

To swap the neck, change its `_target_` and keep the channel contract intact.

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `RuntimeError: channels/size mismatch` at neck input | `pts_neck.in_channels` ≠ backbone `out_channels` | align them |
| Size mismatch at head input | head `in_channels` ≠ sum of neck `out_channels` | set head `in_channels` = `sum(out_channels)` |
| Feature maps not aligning for `torch.cat` | `upsample_strides` don't bring stages to one resolution | recompute strides from the backbone's downsampling |
| Camera BEV shapes disagree in fusion | camera neck / view-transform output ≠ lidar BEV shape | check view-transform grid vs lidar grid (BEVFusion validates this) |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Change neck output width | edit `out_channels` (and the head's `in_channels` to match) |
| Use fewer/more backbone stages | keep `in_channels`/`out_channels`/`upsample_strides` lists the same length as the backbone stages |
| Swap to a camera neck | change `_target_` to `CPFPN`/`GeneralizedLSSFPN` and wire the view transform |
| Add a new neck | add an `nn.Module` under `necks/` taking a list of maps → one tensor; reference by `_target_` |

---

**Next:** [head.md](head.md) — where predictions, targets, loss, and decoding live.
