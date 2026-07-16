# 01 · AWML ⇄ autoware-ml 對照

這份是給你「用 AWML / OpenMMLab 的既有知識，快速映射到 autoware-ml」的核心文件。

- [為什麼要重寫：OpenMMLab 依賴的深度](#為什麼要重寫openmmlab-依賴的深度)
- [技術棧對照](#技術棧對照)
- [概念對照表（最重要）](#概念對照表最重要)
- [實際 config side-by-side](#實際-config-side-by-side)
- [目錄結構對照](#目錄結構對照)
- [不見了／新增了／還沒搬](#不見了新增了還沒搬)

---

## 為什麼要重寫：OpenMMLab 依賴的深度

先量化一下 AWML 到底綁 mm 綁多深（以下都是 AWML repo 的實測）：

**7 個 mm 套件全是硬相依**，在 `Dockerfile` 用 OpenMMLab 的 `mim` 安裝並釘死版本：

```dockerfile
ARG MMCV="2.1.0"  MMENGINE="0.10.7"  MMDET="3.3.0"
ARG MMDEPLOY="1.3.1"  MMDET3D="1.4.0"  MMPRETRAIN="1.2.0"  MMSEGMENTATION="1.2.2"
RUN mim install mmcv==${MMCV} mmdeploy==${MMDEPLOY} mmdet==${MMDET} \
    mmdet3d==${MMDET3D} mmengine==${MMENGINE} mmpretrain[multimodal]==${MMPRETRAIN} \
    mmsegmentation==${MMSEGMENTATION}
```

**import 深度**（整個 repo 掃出來的次數，節錄）：

```
71 mmengine.registry   59 mmdet3d.registry   56 mmengine.config   45 mmengine.logging
41 mmdet3d.models      36 mmdet3d.structures 34 mmdet.models       24 mmcv.cnn
22 mmengine.runner     21 mmengine.model     16 mmdet3d.datasets   ...
```

**registry 裝飾器用量**：`@MODELS.register_module` ×64、`@TRANSFORMS` ×40、`@HOOKS` ×16、`@DATASETS` ×14；`MODELS.build` ×45、`Runner.from_cfg` ×14。

**最深的耦合點**是 `autoware_ml/registry.py`——它幾乎是 mmdet3d `registry.py` 的翻版，把約 20 個 registry 建成 mmengine 全域 registry 的**子節點**：

```python
from mmengine.registry import MODELS as MMENGINE_MODELS
MODELS = Registry("model", parent=MMENGINE_MODELS,
                  locations=["mmdet3d.models", "mmdet.models"])
DATASETS = Registry("dataset", parent=MMENGINE_DATASETS,
                    locations=["mmdet3d.datasets", "mmdet.datasets"])
```

其他徵狀：

- `default_scope = "mmdet3d"`（在 `autoware_ml/configs/detection3d/default_runtime.py`）。
- `T4Dataset(NuScenesDataset)`、`T4Metric(NuScenesMetric)`——直接繼承 mmdet3d 的類別。
- `tools/detection3d/train.py` / `test.py` / `deploy.py` 幾乎是 mmdet3d / mmdeploy 官方腳本的 copy。
- `setup.py` 還自稱 `"OpenMMLab's next-generation platform for general 3D object detection"`，作者 `"MMDetection3D Contributors"`。
- Docker build 時要 patch 三個 site-packages：`.patches/mmdet3d.patch`、`.patches/mmengine.patch`、`.patches/spconv.patch`（為了讓 mm 在新 numpy / PyTorch 2.6+ 上不炸）。

> 一句話總結 AWML：**它「就是」mmdetection3d，外面包一層 T4/Autoware。** 想動 PyTorch/CUDA/numpy 版本，就得跟這 7 個套件的版本網打架。這就是 autoware-ml 要根治的病。
>
> （補充：AWML 後期的 `deployment/` 資料夾其實已經是「去 mmdeploy」的 clean-room 重寫，只保留 mmengine config 解析 + mmdet3d 載入 checkpoint。可以把它看成 autoware-ml 精神的前哨站。）

---

## 技術棧對照

| 層面 | AWML（舊） | autoware-ml（新） |
| --- | --- | --- |
| 語言環境 | Python（mm 綁定），PyTorch **2.2 / 2.8**（被 mm 卡） | Python **3.11**，PyTorch **2.9.1 + cu128** |
| CUDA | 12.9 / cuDNN 9（Docker） | **12.8**（pixi system-requirement） |
| 環境管理 | `mim` + Docker + monkeypatch | **`pixi`**（`pixi.lock` 完整鎖定）+ Docker（Ubuntu 24.04 base） |
| 設定系統 | mmengine `Config`：`.py`、`dict(type=...)`、`_base_` | **Hydra 1.3 + OmegaConf**：`.yaml`、`_target_`、`defaults:` |
| 超參搜尋 | 無標準做法 | **Optuna**（`hydra-optuna-sweeper`，`--multirun`） |
| 訓練引擎 | mmengine `Runner` | **PyTorch Lightning 2.6 `Trainer`** |
| 模型基底 | mmdet3d detector / mmengine `BaseModule` | 自家 `BaseModel(MetricEvalMixin, LightningModule)` |
| 元件發現 | 全域 Registry + `custom_imports` import 副作用 | **無 registry**：`_target_` 直接指 Python 路徑 |
| 資料樣本結構 | `Det3DDataSample` / `InstanceData`（mmengine 結構） | **純 `dict[str, Any]`** + 自家 `geometry` 型別 |
| 低階 ops | `mmcv` ops | **`autoware_ml/ops/`（原生）** + 外部 `spconv-cu120` |
| 指標評估 | mmdet3d metric（`NuScenesMetric` 子類） | **torchmetrics**（`MetricSuite`/`Metric`，range-aware） |
| 部署 | `mmdeploy` | `torch.onnx.export`（含 dynamo）+ **TensorRT Python Builder API** |
| 實驗追蹤 | text log、`work_dirs/` | **MLflow**（sqlite + UI）、`mlruns/` |
| 型別安全 | 少 | **jaxtyping**（tensor shape 標註）、pydantic、大量 type hints |
| CLI | 每個任務一支腳本 `tools/<task>/train.py` | 統一 `autoware-ml` 指令（**Typer**） |

---

## 概念對照表（最重要）

> 讀法：左邊是「你在 mmdet3d / AWML 已經會的」，右邊是「autoware-ml 裡的對應物」。

### 設定與組裝

| mmdet3d / AWML | autoware-ml | 說明 |
| --- | --- | --- |
| `dict(type="SECOND", ...)` 由 Registry 解析 | `{_target_: autoware_ml.models.detection3d.backbones.second.SECONDBackbone, ...}` | 從「字串 → registry 查表」變成「**完整 Python 路徑 → 直接 import 並呼叫**」。由 `hydra.utils.instantiate()` 遞迴建構。 |
| `@MODELS.register_module()` / `@TRANSFORMS.register_module()` | **（不需要）** | 沒有 registry。要用哪個類別，config 直接寫它的 import 路徑即可。少了一整層「註冊 → scope → 查表」的間接。 |
| `build_from_cfg(cfg, REGISTRY)` / `MODELS.build(cfg)` | `hydra.utils.instantiate(cfg)` | 遞迴實例化；預設會把巢狀 `_target_` 也建好（`_recursive_: true`）。 |
| `default_scope = "mmdet3d"`、`type="mmdet.L1Loss"` 這種 scope 前綴 | **（不存在）** | 沒有 scope 概念，路徑本身就是唯一解。 |
| mmengine `Config.fromfile(...)`、`.py` config、`_base_ = [...]` | Hydra YAML + `defaults:` 清單 + `# @package _global_` | 繼承從「Python `_base_` list」變成「Hydra defaults 組合」。 |
| config 內用 `{{_base_.class_names}}` 插值 | `${dataset.detection3d.class_names}`（OmegaConf 插值） | 還支援 `${oc.env:...}`、`${point_cloud_range.0}`（索引）、自訂 resolver。 |
| `_partial_` 概念無（都是直接 build） | `_partial_: true` → 回傳 `functools.partial` | optimizer / scheduler 必須這樣寫，因為要延後到拿到 `params` 才呼叫。 |

### 訓練與模型

| mmdet3d / AWML | autoware-ml | 說明 |
| --- | --- | --- |
| `Runner.from_cfg(cfg).train()` | `trainer = instantiate(cfg.trainer); trainer.fit(model, datamodule)` | 訓練迴圈整個換成 Lightning。 |
| `MVXTwoStageDetector` / `Base3DDetector` / `BaseModule` | `BaseModel(MetricEvalMixin, L.LightningModule, ABC)` | 新基底只要求兩個抽象方法：`forward(**kwargs)` 與 `compute_metrics(batch, outputs) → dict`（必含 `"loss"`）。 |
| detector 的 `loss()` / `predict()` / `_forward()` 三態 | 單一 `forward()` + `compute_metrics()`；`training/validation/test/predict_step` 已在基底寫死（`@final`） | 所有模型共用同一條 step 路徑；差異靠 override hook。 |
| model `forward(batch_inputs_dict, batch_data_samples)` 固定簽名 | `forward(self, points, gt_boxes, ...)` **任意簽名** | 基底用 `inspect.signature(self.forward)` **只挑名字對得上的 batch key** 餵進去。所以參數名要跟 batch dict 的 key 一致。 |
| `data_preprocessor=dict(type="Det3DDataPreprocessor", voxel=True,...)` | `data_preprocessing`（`DataPreprocessing` pipeline），在 `model.on_after_batch_transfer` 執行 | voxelization 等 GPU 前處理，從「mmdet3d 的 data_preprocessor」變成「model 自帶、在 batch 搬到 GPU 後跑」的 pipeline。 |
| `default_hooks=dict(checkpoint=..., logger=...)` + custom `@HOOKS` | Lightning **Callbacks**（`ModelCheckpoint`、`LearningRateMonitor`、自家 `EarlyStopping`） | Hook 機制換成 Lightning callback。 |
| `optim_wrapper=dict(type="AmpOptimWrapper", optimizer=dict(type="AdamW",...))` | `model.optimizer` / `model.scheduler`（`_partial_` 的 factory），由 `configure_optimizers()` 組 | AMP 改成 `trainer.precision=16-mixed`。 |
| `param_scheduler=[dict(type="CosineAnnealingLR"),...]` | `model.scheduler`（可用自家 `CyclicCosineAnnealingLR` 等） | LR 排程 |
| `EpochBasedTrainLoop(max_epochs=...)` | `trainer.max_epochs` | |
| 分散式：`Runner` 自動 launch | `trainer.devices=auto` + Lightning DDP（自動偵測多卡） | |

### 資料

| mmdet3d / AWML | autoware-ml | 說明 |
| --- | --- | --- |
| `T4Dataset(NuScenesDataset)`、mmdet3d `Det3DDataset` | `Dataset(TorchDataset, ABC)`（實作 `get_data_info(index)→dict`） | 不再繼承 mmdet3d。`get_data_info` **只回 metadata**，真正讀檔在 transforms。 |
| Dataset 的 `pipeline=[dict(type="LoadPointsFromFile"),...]` | `datamodule` 的 `train_transforms`/`val_transforms`（`TransformsCompose`） | pipeline 概念保留，但改成 autoware-ml 自己的 transforms。 |
| mmcv/mmdet3d transforms（`LoadPointsFromFile`、`GlobalRotScaleTrans`、`ObjectRangeFilter`…） | `autoware_ml.transforms.*` 同名/類似的原生實作 | 全部重寫成 `BaseTransform`（dict-in/dict-out）。 |
| DataLoader 由 `Runner` 依 `train_dataloader` cfg 建 | `LightningDataModule` + `DataLoaderConfig`；batch 疊法由 **`collation_map`** 白名單決定 | `collation_map` 用 `stack`/`concat`/`index_concat`/`list` 策略處理不定長點雲；**沒列到的 key 會被丟掉**。 |
| info `.pkl`（mmdet3d nuScenes 格式：`metainfo` + `data_list`，含 `instances`/`lidar_points`/`images`…） | **兩套**：① info `.pkl`（datamodule 用，`create-dataset` 產生）② `databases/` 的 **Polars `.parquet`** scenario-record（pydantic `DatasetRecord`） | 見下方「兩套資料系統」。 |
| `create_data_t4dataset.py`（走 mmdet3d converter） | `autoware-ml create-dataset` → `NuScenesDatasetGenerator`（目前只註冊 nuScenes） | |
| `Det3DDataSample` / `InstanceData` / `LiDARInstance3DBoxes` | 純 `dict` batch + `geometry`（`LidarBBoxes3D`、`LiDARPoints`）+ `types`（`Box3DFieldIndex` 固定 10 維框佈局） | 資料結構去 mmengine 化。 |

**兩套資料系統**（新框架同時存在，別搞混）：

1. **`datamodule/`（info-file / pkl，Lightning-facing，主力）**——`Dataset`/`DataModule` 讀 `*_infos_train.pkl`，`get_data_info` 回 metadata，transforms 再 lazy 讀點雲。detection3d / segmentation3d 現在都走這套。
2. **`databases/`（Polars + pydantic scenario-record，較新、transitional）**——`DatabaseInterface`(Protocol) / `BaseDatabase`，把 T4 標註整理成 frozen pydantic `DatasetRecord`、cache 成 `.parquet`（用 schema hash 當檔名），餵給 `MultiTaskDataModule` + `ScenarioSplitter`。由獨立的 `scripts/generate_dataset.py`（`@hydra.main`）驅動。程式碼裡明說這是過渡、未來會併入正式框架。

### ops 與部署

| mmdet3d / AWML | autoware-ml | 說明 |
| --- | --- | --- |
| `mmcv` 的 `Voxelization` | `autoware_ml/ops/voxelization/`（**純向量化 PyTorch GPU**，無 C++；刻意對齊 mmcv 的 ZYX + `round()` 慣例） | |
| `mmcv` / spconv 的 sparse conv | `autoware_ml/ops/spconv/`（包在外部 `spconv-cu120` 上的 deploy-aware wrapper） | |
| BEV pooling（BEVFusion，原本各 project 自帶 CUDA） | `autoware_ml/ops/bev_pool/`（**唯一需要編譯的 CUDA 擴充** `bev_pool_ext`） | |
| — | `autoware_ml/ops/segment/`（`segment_csr`）、`ops/indexing/`（`unique`/`argsort`，PTv3 export 用） | 新增的 graph-level ops |
| `mmdeploy` deploy_cfg + `torch2onnx` + `to_backend` | `autoware-ml deploy`：`model.build_export_spec()` → `torch.onnx.export`（dynamo）→ TensorRT Builder | 完全去 mmdeploy。ONNX 自訂算子改用 `autoware::*` symbolic（搭 Autoware 端 TensorRT plugin）。 |
| deploy config 是 mmdeploy schema（`codebase_config`/`backend_config`/`onnx_config`） | config 的 `deploy:` 區塊（`deploy.onnx.*`、`deploy.tensorrt.*`） | |

### 專案組織與周邊

| mmdet3d / AWML | autoware-ml | 說明 |
| --- | --- | --- |
| `projects/<Model>/`（configs + 自訂模組 + 註冊 + README + Dockerfile） | `autoware_ml/models/<task>/<model>.py`（原生類別）+ `autoware_ml/configs/tasks/<task>/<model>/*.yaml` | 「project = 一包要註冊的東西」→「model 就是一個一般 Python 類別 + 幾份 YAML」。沒有 per-project Dockerfile / setup.py。 |
| `tools/<task>/{train,test}.py`（每任務一支，抄 mmdet3d） | 單一 `autoware-ml {train,test,deploy,create-dataset}` | |
| `work_dirs/<config>/` | `mlruns/<task>/<model>/<config>/<run_id>/`（artifacts + hydra + checkpoints） | 產物目錄換成 MLflow 結構。 |
| `pipelines/webauto/`、active-learning tools | **（尚未移植）** | |
| `mim install ...` | `pixi install --locked` | |

---

## 實際 config side-by-side

同一個 TransFusion，兩邊 config 的寫法差異一眼就懂。

**AWML（mmengine `.py`，registry 字串）** — `projects/TransFusion/configs/.../*.py`：

```python
model = dict(
    type="TransFusion",                       # ← registry 字串，靠 mmdet3d MODELS registry 查表
    data_preprocessor=dict(
        type="Det3DDataPreprocessor",         # ← mmdet3d 元件
        voxel=True, voxel_layer=dict(...),
    ),
    pts_voxel_encoder=dict(type="PillarFeatureNet", ...),
    pts_middle_encoder=dict(type="PointPillarsScatter", ...),
    pts_backbone=dict(type="SECOND", ...),
    pts_neck=dict(type="SECONDFPN", ...),
    pts_bbox_head=dict(type="TransFusionHead", ...,
        loss_cls=dict(type="mmdet.AmpGaussianFocalLoss", ...)),  # ← 還跨 scope 到 mmdet
)
```

對應的模型類別 `transfusion.py`：

```python
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

@MODELS.register_module()                     # ← 要註冊
class TransFusion(MVXTwoStageDetector):       # ← 繼承 mmdet3d
    ...
```

**autoware-ml（Hydra `.yaml`，`_target_` 路徑）** — `autoware_ml/configs/tasks/detection3d/transfusion/base.yaml`：

```yaml
# @package _global_
defaults:
  - /defaults/default_runtime
  - _self_

model:
  _target_: autoware_ml.models.detection3d.transfusion.TransFusionDetectionModel  # ← 完整路徑，直接 import
  pts_voxel_encoder:
    _target_: autoware_ml.models.detection3d.encoders.voxel.HardSimpleVoxelSinCosEncoder
    in_channels: 4
  pts_middle_encoder:
    _target_: autoware_ml.models.detection3d.encoders.sparse.SparseEncoder
  pts_backbone:
    _target_: autoware_ml.models.detection3d.backbones.second.SECONDBackbone
  pts_neck:
    _target_: autoware_ml.models.detection3d.necks.second_fpn.SECONDFPN
  bbox_head:
    _target_: autoware_ml.models.detection3d.heads.transfusion.TransFusionHead
    assigner:
      _target_: autoware_ml.models.detection3d.task_modules.assigners.HungarianAssigner3D
  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true          # ← optimizer 一定要 partial
    lr: 0.0001
```

對應的模型類別 `models/detection3d/transfusion.py`：

```python
from autoware_ml.models.base import BaseModel

class TransFusionDetectionModel(BaseModel):   # ← 繼承自家 BaseModel，不用註冊
    def __init__(self, pts_voxel_encoder, pts_middle_encoder,
                 pts_backbone, pts_neck, bbox_head, **kwargs):
        super().__init__(**kwargs)
        ...
    def forward(self, points, ...): ...
    def compute_metrics(self, batch_inputs_dict, outputs):
        return self.bbox_head.loss(...)       # 回傳含 "loss" 的 dict
```

差異重點：`type="X"` → `_target_: a.b.C`；`@register_module` → 不用；`MVXTwoStageDetector` → `BaseModel`；`Det3DDataPreprocessor` → `data_preprocessing`；`.py` dict → `.yaml` 組合；`???` 是 Hydra 的「必填但這層還沒給值」佔位符（等 leaf config 填）。

---

## 目錄結構對照

**AWML（舊）** — 「library + projects + tools + pipelines」的 mm 風格：

```
AWML/
├── autoware_ml/        # 被當 library 的共用層：registry.py(!)、detection3d/、configs/(.py 片段)
├── projects/           # 每個模型一包（CenterPoint/ BEVFusion/ TransFusion/ YOLOX/ ...）
│   └── <Model>/{configs(.py), models/, deploy/, Dockerfile, setup.py, README}
├── tools/              # 每任務的 train.py/test.py/deploy.py（抄 mmdet3d/mmdeploy）
├── deployment/         # 較新的去-mmdeploy 部署重寫
├── pipelines/webauto/  # MLOps glue
├── third_party/        # vendored onnxruntime headers
├── data/  work_dirs/   # 資料 / 產物（mm 慣例）
└── Dockerfile          # mim install 7 個 mm 套件 + patch
```

**autoware-ml（新）** — 「單一 package + Hydra configs」的 Lightning 風格：

```
autoware-ml/
├── autoware_ml/            # 唯一的 package
│   ├── cli/                # Typer CLI（cli.py 入口、runtime.py 接 Hydra/MLflow）
│   ├── scripts/            # @hydra.main 進入點：train.py/test.py/deploy.py/create_dataset.py/generate_dataset.py
│   ├── configs/            # Hydra YAML：tasks/ datasets/ database/ datamodule/ defaults/ generators/
│   ├── datamodule/         # Dataset/DataModule（info-pkl 系統）+ collation
│   ├── databases/          # Polars+pydantic scenario-record 系統（+ schemas/）
│   ├── transforms/         # BaseTransform：point_cloud/ boxes3d/ camera/ camera_lidar/ segmentation3d/ ...
│   ├── preprocessing/      # DataPreprocessing（on_after_batch_transfer 的 GPU 前處理）
│   ├── models/             # BaseModel 及子模組：detection3d/ segmentation3d/ multi/ calibration_status/ common/
│   ├── ops/                # 原生 ops：bev_pool/(CUDA) voxelization/ spconv/ segment/ indexing/
│   ├── geometry/           # LidarBBoxes3D / LiDARPoints
│   ├── losses/ metrics/    # 原生 loss；torchmetrics 的 MetricSuite/Metric
│   ├── callbacks/          # Lightning callbacks（EarlyStopping…）
│   ├── types/ utils/       # enum 詞彙表；runtime/checkpoints/deploy/mlflow/schedulers/...
│   └── tests/
├── docs/                   # 官方英文文件（zensical）
├── docker/  ansible/       # 環境
├── pyproject.toml  pixi.lock  # pixi 環境（取代 mim）
└── set_data_path.sh        # 設 AUTOWARE_ML_DATA_PATH
```

---

## 不見了／新增了／還沒搬

**新框架新增（AWML 沒有或很弱）**：

- 統一 CLI（`autoware-ml`）、Hydra 組態 + `--multirun` 掃參、Optuna、MLflow（UI/lineage/param-drift）。
- `session` 子指令：tmux-backed 背景訓練（取代 `nohup`）。
- 型別化資料 schema（pydantic `DatasetRecord` + Polars parquet）。
- torchmetrics 的 range-aware 指標（每個距離帶各算一次 mAP / IoU）。
- 多任務（PTv3 同時做 seg + det）與 multi-head 部署（`--weights` 可合併多個 checkpoint，且**強制全參數覆蓋檢查**）。

**還沒從 AWML 搬過來**（要用就先回 AWML）：

- 2D：YOLOX / YOLOX_opt、GLIP、MobileNetv2、BLIP-2、traffic light recognition。
- Active learning 全套：auto-labeling（`pseudo_label`）、`scene_selector`、data mining、calibration_classification 工具。
- WebAuto pipeline、大部分分析／視覺化 tools。
- CenterPoint 目前只有 head，還沒有完整的 top-level 模型與 train/deploy。

**心態建議**：把 autoware-ml 當成「3D 感知 train→deploy 主幹的乾淨重寫」。它的 API 還會變（Early Alpha），但骨架（Hydra + Lightning + 原生 ops + BaseModel/DataModule/transforms/metrics 契約）已經定型，值得先把這套契約讀熟。

→ 內部每個子系統怎麼運作，看 [`02-architecture.md`](02-architecture.md)。
