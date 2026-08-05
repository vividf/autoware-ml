# Neck

> **本文涵蓋內容：** neck 階段 — 位於 backbone 和 head 之間的小型模組，負責將
> 多尺度特徵聚合（aggregate）成 head 所需的單一特徵圖。
> 先備知識：[backbone.md](backbone.md)。

---

## 1. 為什麼需要 neck

backbone 會輸出多種解析度的特徵（例如 `1/2, 1/4, 1/8`）。而 head 想要的是**單一**固定
解析度/channel 數的特徵圖。neck 就是用來銜接這個落差：它對粗解析度的特徵圖做上採樣
（upsample）、對齊解析度，並將它們合併。它被刻意設計得很小 — 是用來融合的地方，
而不是用來增加深度的。

---

## 2. `SECONDFPN` — 標準（canonical）LiDAR neck（`necks/second_fpn.py:31`）

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

設定方式（CenterPoint base）：

```yaml
pts_neck:
  _target_: autoware_ml.models.detection3d.necks.second_fpn.SECONDFPN
  in_channels: [64, 128, 256]        # matches SECONDBackbone.out_channels
  out_channels: [128, 128, 128]      # each stage → 128 channels
  upsample_strides: [0.5, 1, 2]      # align the three stages to a common resolution
```

輸出是一個 `(B, 384, H', W')` 的 tensor（`128×3` 串接而成）— 正好就是 `CenterHead`
所預期的 `in_channels: 384`。**neck 的 `in_channels` 必須等於 backbone 的
`out_channels`，而 head 的 `in_channels` 必須等於 neck 的總輸出 channel 數。**
這個三方 channel 約定（contract）是最常見的形狀（shape）錯誤來源。

每個區塊同樣是一個 `ConvModule`（其中 `transpose=True` 用於 deconvolution 上採樣）—
這是隨處可見的同一個共用區塊。

---

## 3. 其他 neck（`models/common/necks/`）

Neck 在各個任務之間是共用的：

| Neck | 檔案 | 用途 |
| ---- | ---- | --- |
| `SECONDFPN` | `detection3d/necks/second_fpn.py` | LiDAR BEV detection (CenterPoint, pillar/voxel detectors) |
| `CPFPN` | `common/necks/cp_fpn.py` | camera feature pyramid (StreamPETR-style camera detectors) |
| `GeneralizedLSSFPN` | `common/necks/lss_fpn.py` | camera FPN feeding an LSS view transform (BEVFusion camera branch) |
| `GlobalAveragePooling` | `common/necks/global_average_pooling.py` | classification necks (collapse spatial dims) |

camera neck（`CPFPN`、`GeneralizedLSSFPN`）位於影像 backbone 和 view transform
（`models/detection3d/view_transforms/`）之間，view transform 會將 2D 影像特徵
提升（lift）到 BEV 空間，以便和 LiDAR 進行 fusion。

---

## 4. Neck 如何接入（plug in）

和 backbone 一樣，neck 也是一個 `_target_` 子模組。`CenterPointDetectionModel.forward`
會直接呼叫它：

```python
bev_features = self.pts_backbone(bev_features)   # list of maps
bev_features = self.pts_neck(bev_features)        # one fused tensor
return self.bbox_head(bev_features)
```

若要替換 neck，只需更改其 `_target_`，並保持 channel 約定（contract）不變。

---

## Common debugging cases

| 症狀 | 原因 | 修正 |
| ------- | ----- | --- |
| neck 輸入處出現 `RuntimeError: channels/size mismatch` | `pts_neck.in_channels` ≠ backbone 的 `out_channels` | 使兩者對齊 |
| head 輸入處出現大小不匹配 | head 的 `in_channels` ≠ neck 各 `out_channels` 之總和 | 設定 head 的 `in_channels` = `sum(out_channels)` |
| 特徵圖在 `torch.cat` 時無法對齊 | `upsample_strides` 沒有將各階段對齊到同一解析度 | 根據 backbone 的降採樣重新計算 stride |
| fusion 時 camera BEV 形狀不一致 | camera neck / view-transform 的輸出 ≠ lidar 的 BEV 形狀 | 檢查 view-transform 的網格與 lidar 網格是否一致（BEVFusion 會驗證這點） |

---

## Common modification scenarios

| 我想要… | 這樣做 |
| ---------- | ------- |
| 改變 neck 的輸出寬度 | 編輯 `out_channels`（並同步調整 head 的 `in_channels` 使其匹配） |
| 使用更少/更多的 backbone 階段 | 讓 `in_channels`/`out_channels`/`upsample_strides` 清單長度與 backbone 階段數保持一致 |
| 換成 camera neck | 將 `_target_` 改為 `CPFPN`/`GeneralizedLSSFPN`，並接上 view transform |
| 新增一個 neck | 在 `necks/` 下新增一個 `nn.Module`，輸入特徵圖清單 → 輸出單一 tensor；透過 `_target_` 參照 |

---

**Next:** [head.md](head.md) — 預測（predictions）、目標（targets）、損失（loss）與解碼（decoding）所在之處。
