# 程式碼逐步解析 — Config 流程

> 說明驅動一次執行（run）的單一 `cfg` 物件，是如何從許多個小型 YAML 檔案組裝而成的。
> 這是整個框架中最不像單純 PyTorch 的部分，因此本文會提供一個完整的實作範例：
> **`detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes`**。
>
> 參考資料：`docs/user-guide/configuration.md`。本文件是「追蹤一份實際的 config，
> 看它如何建構出物件」的輔助說明。

---

## 心智模型：config = Hydra 會轉換成物件的一棵 tree

一次執行的 config 是一個巨大的巢狀字典（`DictConfig`）。有兩個機制負責建構它：

1. **Composition（組合）** — Hydra 依照 `defaults:` 清單的指引，把許多個 YAML 檔案合併成一棵
   tree。
2. **Instantiation（實例化）** — `hydra.utils.instantiate(cfg.<section>)` 會走訪整棵 tree，
   只要看到 `_target_`，就會匯入對應的 class／function 並呼叫它。巢狀的 `_target_` 會先被
   建構（由下往上），所以父層拿到的都是已經建構完成的子層。

所以 config 是一份*食譜*，而 `instantiate` 就是*烤箱*。

---

## 你必須知道的詞彙

| Token | 意義 | 範例 |
| ----- | ------- | ------- |
| `defaults:` | 依序合併進來的其他 config 清單 | `- /defaults/default_runtime` |
| `_self_` | *此檔案*自身的值在 defaults 順序中生效的位置 | 通常放在最後 = 此檔案優先 |
| `# @package _global_` | 把此檔案的鍵合併到 config 的根層級 | task config 的最上方 |
| `# @package nuscenes` | 把此檔案的鍵合併到 `cfg.nuscenes` 之下 | dataset group 檔案 |
| `_target_` | 要實例化的 Python import 路徑 | `autoware_ml.models.detection3d.centerpoint.CenterPointDetectionModel` |
| `_partial_: true` | 建構一個 `functools.partial`，先不呼叫 | optimizer／scheduler |
| `${a.b.c}` | 內插（interpolate）另一個 config 的值 | `${voxel_size}` |
| `${oc.env:VAR,default}` | 內插一個環境變數 | run 目錄 |
| `???` | **必填**值；必須被填上，否則 Hydra 會報錯 | `dataset: ???` |
| `${resolver:arg}` | 呼叫一個自訂的 OmegaConf resolver | `${user_config_name:...}` |

---

## 這個範例的 composition chain

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

由上往下讀，可以理解為「leaf 拉入 base，base 拉入 runtime scaffold，runtime scaffold 再拉入
各個 module 片段」。若從*優先權（precedence）*的角度，則要由下往上讀：後面的項目與 `_self_`
會覆寫前面的項目。

---

## Layer 1 — runtime scaffold（`defaults/default_runtime.yaml`）

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

每個 task config 都會繼承這份設定，因此每次執行*預設*都會有 trainer、logger、callbacks 與
deploy 區塊。task 只需要覆寫它真正需要的部分。

---

## Layer 2 — model／family 的 base（`tasks/detection3d/centerpoint/base.yaml`）

這裡就是**CenterPoint 架構**一次性定義的地方，供所有 CenterPoint 變體共用：

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

有兩件事值得注意：

- **`???`（必填但缺漏）。** base 無法在你選定 range／voxel size 之前得知 voxel grid 的大小。
  所以它把 `point_cloud_range`、`voxel_size`、`output_shape`、`layer_strides`、
  `out_size_factor`、`nms_min_radius`、`max_lr_factor` 都宣告為 `???`，強制要求 leaf 去填上
  這些值。漏填任何一個都會引發 `MissingMandatoryValue`。
- **`_partial_: true` 只用在 optimizer／scheduler 上。** 每一個*module*（`pts_voxel_encoder`、
  `pts_backbone`……）都會在模型建構子執行之前，先建構成一個 `nn.Module`。但
  optimizer／scheduler 還無法被建構 — 它們需要模型的參數（parameters），而這些參數要等模型
  建好之後才存在。因此它們會保持為*可呼叫物件*（`functools.partial`），並在稍後於
  `configure_optimizers()` 內部被呼叫。詳見
  [../training/optimizer_scheduler.md](../training/optimizer_scheduler.md)。

---

## Layer 3 — leaf（`voxel020_second_secfpn_51m_nuscenes.yaml`）

leaf 負責回答「用哪個 dataset、什麼樣的 range／voxel、怎麼載入資料」：

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

注意 `test_transforms: ${datamodule.val_transforms}` — 內插讓同一個 pipeline 可以被重複
使用，並保證 test 與 val 的前處理完全一致。

---

## Dataset group 與 `@package` 指令

為什麼是 `dataset: ${nuscenes}` 而不是把 dataset 直接寫在裡面？因為 dataset 存在於自己獨立的
檔案中，並帶有一個**package 指令**，用來決定它的鍵會落在*哪裡*：

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

所以在 composition 完成之後，`cfg.nuscenes.detection3d.class_names` 就會存在，而 leaf 的
`dataset: ${nuscenes}` 會讓 `cfg.dataset` 指向那個 group。這也是為什麼模型會讀取
`num_classes: ${dataset.detection3d.num_classes}` 與 `metrics: ${dataset.detection3d.metrics}`
— *同一份* dataset 定義同時餵給了 model、datamodule 與 metrics，因此它們對 class names 的
認知不可能不一致。

這種間接性（config → variable → group）是刻意設計的：只要換掉一個 `defaults:` 項目，就能讓
整個 run 改換到不同的 dataset。

---

## 從 `cfg` 到物件：`instantiate(cfg.model)` 做了什麼

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

子層會先被建構完成，再交給父層的建構子。這就是為什麼模型的 `__init__` 收到的是已經完整
建構好的 `nn.Module`，而不是 config。

---

## 從命令列覆寫

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

`--cfg job` 是 config 除錯時最有用的單一工具：它會顯示即將被實例化、且內插已全部解析完成的
精確 tree。

---

## 常見除錯情境

| 症狀 | 原因 | 解法 |
| ------- | ----- | --- |
| `MissingMandatoryValue` / `??? ` 錯誤 | base 的 `???` 欄位沒有被 leaf 填上 | 在 leaf 中把它填上（例如 `output_shape`、`voxel_size`） |
| `InterpolationKeyError: ${dataset...}` | dataset group 沒有被 compose 進來，或是 `dataset:` 沒有設定 | 確認 leaf 的 `defaults` 中有 `/datasets/...`，且有設定 `dataset: ${...}` |
| Override 出現「could not be added」 | 在一個不存在的鍵上使用了普通的 override | 加上 `+` 來建立新鍵 |
| 使用 `+` 時出現 Override「already exists」 | 在既有的鍵上使用了 `+` | 拿掉 `+` |
| 值被悄悄地沒有套用 | `_self_` 的順序：後面有東西覆寫了你的值 | 檢查 `defaults:` 的順序；`_self_` 放最後 = 此檔案優先 |
| 建構出錯誤的 class | `_target_` 打錯字／路徑過期 | 執行 `--cfg job --package <section>` 並檢查 `_target_` |
| 兩個區塊對 class names 的認知不一致 | 沒有從共用的 `dataset` group 讀取 | 把它們全部指向 `${dataset.detection3d....}` |

---

## 常見修改情境

| 我想要…… | 這樣做 |
| ---------- | ------- |
| 在新的 dataset 上訓練同樣的模型 | 建立新的 leaf：換掉 `/datasets/...` 的 `defaults` 項目與 `dataset: ${...}`，設定 range／voxel 與 annotation 檔案 |
| 新增一個變體（例如更長的 range） | 建立一個繼承 `.../base` 的新 leaf，填上該 range 所需的 `???` |
| 為單一實驗變更某個超參數 | 使用 CLI override（`model.optimizer.lr=...`），不需要修改檔案 |
| 為模型新增一個可設定的 module | 在 base 中加入一個 `_target_` 區塊；並在模型的 class 中加入對應的建構子參數 |
| 在不重新定義整個 suite 的情況下調整 metrics | 覆寫 `metric_ranges` / `metric_eval_class_range` 變數 — 請參閱 [../evaluation/metrics.md](../evaluation/metrics.md) |

---

**下一篇：** [important_classes.md](important_classes.md) — 這些 config 所實例化出的 class。
