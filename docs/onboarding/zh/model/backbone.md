# Backbone（與 Encoder）

> **本文涵蓋內容：** 模型中「特徵提取器」（feature extractor）的各個階段 — 將原始
> 點（points）轉換成網格（grid）的 encoder，以及從該網格提取多尺度（multi-scale）特徵的 backbone。
> 先備知識：[model_architecture.md](model_architecture.md)。

---

## 1. 本 repo 中的術語

對於 LiDAR 偵測（detection）來說，網路的前端有**三個**不同的階段，而這個 repo 為它們取了
精確的名稱（請勿混淆）：

| 階段 | 資料夾 | 工作 | 範例 |
| ----- | ------ | --- | ------- |
| **Voxel encoder** | `models/detection3d/encoders/` | per-voxel/pillar point features → one vector per voxel | `PillarFeatureNet` |
| **Middle encoder** | `models/detection3d/encoders/` | scatter/convolve voxels into a dense BEV grid | `PointPillarsScatter`, `SparseEncoder` |
| **Backbone** | `models/detection3d/backbones/` | 2D CNN over the dense grid → multi-scale features | `SECONDBackbone` |

「backbone」指的是 2D CNN；而「點 → 網格」的轉換則是「encoder」。當你在閱讀
`CenterPointDetectionModel.forward` 時，這一點很重要：`pts_voxel_encoder` 和 `pts_middle_encoder`
會出現在 `pts_backbone` *之前*。

---

## 2. `SECONDBackbone` — 標準（canonical）LiDAR backbone（`backbones/second.py:30`）

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

設定方式（CenterPoint base）：

```yaml
pts_backbone:
  _target_: autoware_ml.models.detection3d.backbones.second.SECONDBackbone
  in_channels: 32
  out_channels: [64, 128, 256]     # 3 stages
  layer_nums: [3, 5, 5]            # residual convs per stage
  layer_strides: [2, 2, 2]         # each stage halves resolution
```

因此它接收一個 `(B, 32, H, W)` 的 BEV 網格，並回傳三張解析度分別為 `1/2, 1/4, 1/8`、
channel 數為 `64, 128, 256` 的特徵圖（feature map）。neck（[neck.md](neck.md)）會將它們融合回
一張特徵圖。

每一個卷積（conv）都是一個**`ConvModule`**（`models/common/layers/conv.py`）— 這是一個共用的
conv + norm + activation 區塊，在 backbone、neck 和 head 之間重複使用。只要學會 `ConvModule`
一次，你就能讀懂它們全部。

---

## 3. Voxel/Middle encoder（點 → 網格的前端）

- **`PillarFeatureNet`**（`encoders/pillar.py`）— 「voxel encoder」。它接收 `voxels`、
  `num_points`、`voxel_coords`（由 `PointPillarPreprocessor` 產生），為每個點*裝飾
  （decorate）*上相對於其 pillar 中心的偏移量，執行一個小型的 PFN MLP，然後將結果
  pool 成每個 pillar 一個特徵向量。它將 `decorate(...)` 和 `encode_decorated(...)`
  分開暴露，讓 deployment 可以只匯出（export）該 MLP
  （參見 [../deployment/export_pipeline.md](../deployment/export_pipeline.md)）。
- **`PointPillarsScatter`**（`encoders/pillar.py`）— pillar 的「middle encoder」。
  使用 `voxel_coords` 將每個 pillar 的向量 scatter 回一張密集的 `(B, C, H, W)` BEV 畫布上。
- **`SparseEncoder` / `SparseBasicBlock`**（`encoders/sparse.py`）— 基於 voxel 的替代方案
  （透過外部套件 `spconv` 實現的 3D sparse convolution），供 TransFusion/BEVFusion 風格的
  voxel 偵測器使用，取代 pillar-scatter。Sparse 運算（ops）與 deployment 用的
  `autoware_ml/ops/spconv/` 中的自訂 op 有關。

---

## 4. Camera backbone（共用，位於 `models/common/backbones/`）

對於 camera 和 fusion 模型來說，backbone 是一個影像 CNN：

- **`ResNet18/50` 及多尺度變體**（`common/backbones/resnet.py`）— 標準的影像
  特徵提取器；多尺度變體會輸出 FPN 風格的金字塔（pyramid）。
- **`VoVNet` / `VoVNet99` 多尺度**（`common/backbones/vovnet.py`）— 較重量級的影像
  backbone，供 camera 3D 偵測器使用（例如 StreamPETR 風格）。

這些會餵入（feed）camera neck（`CPFPN`、`GeneralizedLSSFPN`），接著透過 view transform
（`view_transforms/`）將影像特徵提升（lift）到 BEV 空間以進行 fusion。

---

## 5. Point-transformer backbone（PTv3）

`segmentation3d/encoders/` 內含 `PointTransformerV3Encoder` — 一個序列化（serialized）、
基於 attention 的 point backbone，透過 `PTv3BaseModel` 供 PTv3 的 segmentation、detection 和
multi-task 模型共用。與 pillar/voxel 路徑不同，它直接在串接（concatenated）後的點上運算
（因此 PTv3 的 DataModule 使用 `concat`/`index_concat` collation 以及 `batch["offset"]`）。

---

## 6. Backbone 如何接入（plug in）

backbone 只是模型 config 中的一個子模組 `_target_`；Hydra 會建構它，並將該 instance
交給模型的 constructor。若要替換 backbone，只需更改 `_target_` 及其參數（args）— 只要
輸入/輸出 tensor 的約定（contract）維持一致（輸入密集 BEV 網格、輸出特徵圖清單），就不需要
修改任何 Python 程式碼。

---

## Common debugging cases

| 症狀 | 原因 | 修正 |
| ------- | ----- | --- |
| neck 輸入的 channel 不匹配 | `pts_backbone.out_channels` ≠ `pts_neck.in_channels` | 在 config 中保持兩者一致 |
| BEV 網格大小錯誤 / `output_shape` 錯誤 | `voxel_size`/`point_cloud_range` 與 `pts_middle_encoder.output_shape` 不一致 | 重新計算 grid = range / voxel_size；填入正確的 `output_shape` |
| backbone 發生 OOM | voxel 網格過細或 channel 數過多 | 調粗 `voxel_size`、減少 `out_channels`，或減少 batch size |
| `spconv` import 錯誤 | 使用 sparse encoder 但未安裝外部套件 | 安裝 `spconv-cu*`（已在專案中釘選版本）或改用 pillar 模型 |
| Camera backbone 形狀（shape）錯誤 | 影像的 resize/crop transform 與 backbone 的預期不符 | 使 `camera/` 的 resize transform 與 backbone 對齊 |

---

## Common modification scenarios

| 我想要… | 這樣做 |
| ---------- | ------- |
| 讓 backbone 更深/更寬 | 在 config 中調整 `out_channels` / `layer_nums` |
| 從 pillar 切換到 voxel（sparse） | 將 `pts_voxel_encoder`+`pts_middle_encoder` 換成 sparse encoder；調整前處理（preprocessing） |
| 使用不同的影像 backbone | 更改 `common/backbones/*` 的 `_target_`（ResNet ↔ VoVNet） |
| 新增一個 backbone | 在 `backbones/` 下新增一個回傳特徵圖清單的 `nn.Module`；透過 `_target_` 參照 |

---

**Next:** [neck.md](neck.md) — 融合 backbone 的多尺度輸出。
