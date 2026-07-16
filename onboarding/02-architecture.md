# 02 · 架構深入

這份把 autoware-ml 內部每一層攤開講。看完你應該能回答：一次 `autoware-ml train` 從按下 Enter 到反向傳播，中間到底發生什麼事、由哪個檔案負責。

- [整條資料流](#整條資料流)
- [1. CLI 層（Typer + 惰性載入）](#1-cli-層typer--惰性載入)
- [2. Config 層（Hydra / OmegaConf）](#2-config-層hydra--omegaconf)
- [3. 資料層（datamodule 與 databases）](#3-資料層datamodule-與-databases)
- [4. Transforms 層](#4-transforms-層)
- [5. Preprocessing 層（GPU 前處理）](#5-preprocessing-層gpu-前處理)
- [6. Model 層（BaseModel 契約）](#6-model-層basemodel-契約)
- [7. Ops 層（原生算子）](#7-ops-層原生算子)
- [8. Metrics 層](#8-metrics-層)
- [9. Callbacks / Logger](#9-callbacks--logger)
- [10. Deployment 層](#10-deployment-層)
- [檔案位置速查](#檔案位置速查)

---

## 整條資料流

```
autoware-ml train --config-name detection3d/transfusion/voxel0075_..._nuscenes  trainer.max_epochs=50
        │
        ▼  (1) CLI：Typer 解析、惰性載入
autoware_ml/cli/cli.py::main → cli/runtime.py::run_hydra_entrypoint
        │      · 先 hydra.compose() 預建 MLflow run + hydra run-dir
        │      · 匯出 AUTOWARE_ML_RUN_ID / AUTOWARE_ML_HYDRA_RUN_DIR
        │      · 改寫 sys.argv 成 Hydra 形式，跑 scripts/train.py::main
        ▼  (2) Hydra：讀 YAML、組 defaults、解析 ${...}、instantiate 每個 _target_
autoware_ml/scripts/train.py  (@hydra.main)
        │      datamodule = instantiate(cfg.datamodule)
        │      model      = instantiate(cfg.model)
        │      model.set_data_preprocessing(instantiate(cfg.data_preprocessing))
        │      trainer    = instantiate(cfg.trainer) + callbacks + MLflowLogger
        ▼  (3) Lightning：trainer.fit(model, datamodule)
   ┌────────────────────────── 每個 batch ───────────────────────────┐
   │ Dataset.__getitem__(i)                                            │
   │   → get_data_info(i)      只回 metadata dict                       │
   │   → transforms (CPU)      讀點雲/影像、擴增（dict-in/dict-out）     │
   │ collate_fn                依 collation_map 疊 batch（offset/…）    │
   │ ── batch 搬到 GPU ──                                              │
   │ model.on_after_batch_transfer                                     │
   │   → DataPreprocessing     GPU 前處理（voxelization…）             │
   │ model._shared_step                                                │
   │   → 用 inspect.signature 過濾 batch → forward(**對得上的 key)      │
   │   → compute_metrics(batch, outputs) → {"loss": ...}               │
   │   → loss.backward()（Lightning 管）                               │
   │ (val/test) MetricEvalMixin：suite.update(build_eval_output(...))  │
   └───────────────────────────────────────────────────────────────┘
        ▼  epoch 結束
   MetricSuite.compute()（跨 GPU sync）→ log val/<prefix>/<key> → MLflow
```

---

## 1. CLI 層（Typer + 惰性載入）

入口在 `pyproject.toml`：

```toml
[project.scripts]
autoware-ml = "autoware_ml.cli.cli:main"
```

`cli/cli.py` 用 **Typer**（不是 argparse）建 app，分三個群組：主 `app`、`mlflow_app`、`session_app`。子指令：

- `train` / `test` / `deploy` / `create-dataset`
- `mlflow ui` / `mlflow export`
- `session start|attach|detach|ls|stop`（tmux-backed 背景任務）
- **沒有 `predict` 子指令**——預測只存在於 Lightning 的 `predict_step` / `predict_transforms`；要輸出成品是走 `deploy`。

**惰性載入是刻意的**：`cli.py` 只 import Typer，讓 shell 補全很快；真正重的東西（Hydra、MLflow、Lightning）都在 `cli/runtime.py`，每個指令透過 `run_lazy_script(...)` 才載入。

**Hydra 怎麼被接上**：`cli/runtime.py::run_hydra_entrypoint()` 做三件事——
1. `prepare_runtime_environment()` 先 `hydra.compose()` 一次，**預先建好 MLflow run 與 Hydra run-dir**，把 `AUTOWARE_ML_RUN_ID` / `AUTOWARE_ML_HYDRA_RUN_DIR` 塞進環境變數（這是 resume / deploy 能接回同一個 run 的關鍵）。
2. 把 `sys.argv` 改寫成 `[module, "--config-name", <name>, *overrides]`。
3. import 並跑進入點模組的 `main`（`autoware_ml.scripts.{train,test,deploy}`）。

進入點本身是標準 `@hydra.main`（`scripts/train.py`）：

```python
@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model      = hydra.utils.instantiate(cfg.model)
    model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))
    callbacks  = instantiate_callbacks(cfg, ...)
    trainer    = instantiate_trainer(cfg, callbacks, trainer_logger, ...)
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_checkpoint_path)
```

> 對照 AWML：這一整段就是取代 `tools/detection3d/train.py` 裡的 `Runner.from_cfg(cfg).train()`。

---

## 2. Config 層（Hydra / OmegaConf）

所有設定在 `autoware_ml/configs/`（是個 Python module，用 `initialize_config_module` 載入；`configs/paths.py` 提供 `CONFIGS_ROOT`）。

**子目錄**：

| 目錄 | 內容 |
| --- | --- |
| `tasks/` | 使用者實驗 config：`detection3d/{transfusion,bevfusion,ptv3}`、`segmentation3d/{frnet,ptv3}`、`multi/ptv3`、`calibration_status/` |
| `datasets/{nuscenes,t4dataset}/` | `base` / `camera` / `lidar` / `detection3d` / `segmentation3d`（含 class_names、metrics 定義） |
| `database/t4dataset/` | scenario DB、label remapper、box3d pipeline（給 Polars database 系統用） |
| `datamodule/` | dataloader / splitter / transforms 的預設片段 |
| `defaults/` | `default_runtime.yaml` + `modules/*.yaml`（stock Lightning 物件） |
| `generators/` | 給 `generate_dataset.py` 用的 DB 生成 config |
| `resolvers.py` / `paths.py` | Python 端的自訂 resolver 與路徑 |

**組裝機制**（Hydra 慣用語，對照 mmengine `_base_`）：

- `# @package _global_`：把此檔內容併到 root（task config 幾乎都要寫）。
- `defaults:` 清單：組合繼承，例如一個 leaf config：
  ```yaml
  defaults:
    - /tasks/detection3d/transfusion/base    # 模型骨架
    - /datasets/nuscenes/detection3d          # 類別、metrics
    - /datasets/nuscenes/lidar                # 感測器
    - _self_                                  # 本檔覆寫最後套用
  dataset: ${nuscenes}
  point_cloud_range: [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
  ```
- `_target_`：要 instantiate 的 Python 類別/函式；巢狀 `_target_` 預設遞迴建構（`_recursive_: true`）。
- `_partial_: true`：回傳 `functools.partial`（optimizer/scheduler 必用）。
- **插值**：`${data_root}`、索引 `${point_cloud_range.0}`、跨樹 `${dataset.detection3d.class_names}`、環境變數 `${oc.env:AUTOWARE_ML_DATA_PATH}`、`${oc.select:...}`。
- **`# @package <ns>` 命名空間**：dataset config 用 `# @package nuscenes.detection3d` 把自己掛到某個 namespace 下（所以才有 `dataset: ${nuscenes}` 這種寫法）。

**自訂 resolver**（`configs/resolvers.py::register_config_resolvers()`，在 `utils/runtime.py` import 時註冊）：

- `user_config_name`：把 `tasks/` 前綴剝掉，用來組 run-dir 路徑。
- `seg_class_names`：把 segmentation 的 `class_mapping` 反轉成有序類別名 list。
- `merge_lists`：串接多個 metric suite（多任務時 `[${a}, ${b}]` 這種寫法會壞掉，要用這個）。

**run-dir 也是 resolver 驅動的**（`configs/defaults/modules/run.yaml`）：
`${oc.env:AUTOWARE_ML_HYDRA_RUN_DIR, mlruns/${user_config_name:...}/_hydra/...}`。

`configs/defaults/modules/*.yaml` 把 stock 物件以 `_target_` 提供：`trainer.yaml`→`lightning.Trainer`、`logger.yaml`→`MLFlowLogger`、`callbacks.yaml`→`ModelCheckpoint`/`LearningRateMonitor`/自家 `EarlyStopping`、`data_preprocessing.yaml`→`DataPreprocessing`。

> 除錯技巧：`autoware-ml train --config-name ... --cfg job` 只印組好的 config 不執行；加 `--package model` 只印某段。

---

## 3. 資料層（datamodule 與 databases）

新框架**同時存在兩套**資料系統，用途不同：

### (a) `datamodule/`：info-file（pkl）系統，Lightning-facing，目前主力

`datamodule/base.py`：

```python
class Dataset(TorchDataset, ABC):
    def __getitem__(self, index):
        input_dict = self.get_data_info(index)                 # 只回 metadata
        ctx = PipelineContext(dataset=self, index=index)
        return self.apply_transforms(input_dict, self.dataset_transforms, ctx)
    @abstractmethod
    def get_data_info(self, index) -> dict[str, Any]: ...

class DataModule(L.LightningDataModule, ABC):
    @abstractmethod
    def _create_dataset(self, split, transforms) -> Dataset: ...
    def collate_fn(self, batch): ...                            # 由 collation_map 驅動
```

- `setup(stage)` 把 Lightning 的 stage 映射到 split（fit→train/val、test→test、predict→predict），呼叫 `_create_dataset`。
- **`collation_map` 是白名單**（`datamodule/collation.py::CollationStrategy`）：
  - `stack`：固定形狀 tensor 疊新 batch 維（形狀要一致）。
  - `concat`：不定長 tensor 沿 dim0 串接，**額外產生 `offset`** 記錄每個 sample 的累積長度。
  - `index_concat`：像 concat，但值是「索引」，會依前面 sample 的元素數平移，讓不定長點雲的 index 串接後仍有效。
  - `list`：保留成 Python list（不轉 tensor）。
  - **沒列在 `collation_map` 的 key，會在進 model 前被丟掉**（見 gotchas）。
- `DataLoaderConfig` 是 dataclass（batch_size / num_workers / shuffle / pin_memory / persistent_workers…）。

具體 detection datamodule（`datamodule/nuscenes/detection3d.py`、`datamodule/t4dataset/detection3d.py`）用 `pickle.load` 讀 info 檔，`get_data_info` 回 `{"instances", "lidar_path", "sweeps", "class_names"}` 這類 metadata；真正讀點雲是 transforms 的事。T4 版還多了 repeat-factor frame sampling（`FrameSamplingConfig`）。

info 檔由 `autoware-ml create-dataset` 產生 → `scripts/create_dataset.py` → `tools/dataset/runner.py::generate_dataset` → `NuScenesDatasetGenerator`（目前 generator registry 只註冊 `nuscenes`）。

變體齊全：`datamodule/{common,nuscenes,t4dataset,multi_task}/` 各有 detection3d / segmentation3d / multiview / segdet / calibration_status 模組。

### (b) `databases/`：Polars + pydantic scenario-record 系統（較新、transitional）

- `database_interface.py::DatabaseInterface`（`Protocol`）+ `base_database.py::BaseDatabase`。
- `BaseDatabase` 把記錄 cache 成 **Polars `.parquet`**，檔名帶 schema hash 保證可重現：
  ```python
  df_cache_path = self._cache_path / f"{prefix}_{self.database_hash}.parquet"
  records = [DatasetRecord.load_from_dictionary(r) for r in pl.read_parquet(df_cache_path).to_dicts()]
  ```
- Schema 在 `databases/schemas/`：`DatasetRecord` 是 **frozen pydantic `BaseModel`**，欄位含 `scenario_id / sample_id / timestamp_seconds / lidar_frames / lidar_sources / category_mapping / boxes_3d`；子 schema 有 `Box3DDataModel`、`lidar_frames`、`lidar_sources`、`category_mapping`。
- `databases/t4dataset/t4dataset.py::T4Dataset(BaseDatabase)` 實作 `process_scenario_records()`（平行處理、寫 parquet），拆成 `t4records_generator` / `t4sample_records` / `t4scenarios`。
- `databases/scenarios.py`：`Scenarios` / `ScenarioData` / `DatasetParams`（immutable pydantic，全可 hash 以利 cache）。
- `databases/box3d_pipelines/`：`Box3DPipeline` + label remapper / merger / velocity clip。
- 這套餵給 `datamodule/multi_task/multi_task_data_module.py::MultiTaskDataModule(database, splitter, ...)`：
  ```python
  df = self.database.load_polars_scenario_dataframe()
  splits = self.splitter.split_by_polars_dataframe(df, scenarios=self.database.scenarios)
  ```
- Splitter 在 `datamodule/splitters/`（`ScenarioSplitter` / `SplitterInterface`）。
- 由獨立進入點 `scripts/generate_dataset.py`（另一個 `@hydra.main`，config 在 `configs/generators/`）驅動——程式碼註解明講這是過渡、未來會併回正式框架。

> 對照 AWML：兩套都在取代「mmdet3d 的 info `.pkl` + `T4Dataset(NuScenesDataset)`」。(a) 保留 pkl 介面最接近舊習慣；(b) 是想長期取代的「型別化 + 可重現」資料層。

---

## 4. Transforms 層

`transforms/base.py`：

```python
class BaseTransform(ABC):
    p: float | None = None
    _required_keys: Sequence[str] = ()
    _optional_keys: Sequence[str] = ()
    def __call__(self, input_dict, context=None):
        self._validate_required_keys(input_dict)     # 缺 key 直接報錯
        self._handle_optional_keys(input_dict)        # 補預設
        if not self._should_apply():                  # 機率 p 閘門
            return self.on_skip(input_dict)
        return self.transform(input_dict)
    @abstractmethod
    def transform(self, input_dict) -> dict: ...

class TransformsCompose:
    def __call__(self, input_dict, context=None):
        for t in self.pipeline:
            input_dict |= t(input_dict, context=context)   # dict merge
        return input_dict
```

重點：dict-in/dict-out、`|=` 合併、`_required_keys` 缺就報錯、`p` 機率閘門、`context`（`PipelineContext`）讓 transform 能回頭拿 dataset。config 裡以 `_target_` 指到**實作模組**（不要靠 `__init__` re-export）。

**分類**（子目錄，括號為 `BaseTransform` 子類數量）：

- `point_cloud/`（17）：`loading.LoadPointsFromFile`、`sweeps.LoadPointsFromMultiSweeps`、`geometry.{GlobalRotScaleTrans,RandomFlip3D}`、`crop.{PointsRangeFilter,SphereCrop,CenterShift,CropBoxInner/Outer}`、`sampling.PointShuffle`、`perturbation.{RandomJitter,RandomShift,RandomDropout,ElasticDistortion,GridSample}`、`formatting`
- `boxes3d/`（6）：`loading.LoadAnnotations3D`、`filters.{ObjectRangeFilter,ObjectNameFilter,ObjectMinPointsFilter,ObjectRangeMinPointsFilter}`、`merge.MergeObjects3D`
- `camera/`（11）：`resize.ResizeCropFlipRotImage`、`distortion.UndistortImage`、`masking.GridMask`、`normalize.NormalizeMultiviewImage`…
- `camera_lidar/`（8）：`camera_lidar.{LidarCameraFusion,CalibrationMisalignment,Affine,SaveFusionPreview}`、`geometry.{ImageAug3D,BEVLoadMultiViewImageFromFiles}`
- `segmentation3d/`（5）：`loading.LoadSegAnnotations3D`、`range_view.RangeInterpolation`、`mixing.{FrustumMix,InstanceCopy}`
- `common/`（3）、`image/`（1）、`multi_task/`（compose 包裝）

> 對照 AWML：這就是 mmcv/mmdet3d 的 dataset pipeline transforms（`LoadPointsFromFile`、`GlobalRotScaleTrans`、`ObjectRangeFilter`…）的原生重寫版，名字大多刻意沿用。

---

## 5. Preprocessing 層（GPU 前處理）

`preprocessing/base.py::DataPreprocessing` 是 dict-in/dict-out 的 pipeline，但**跑在 GPU 上**：

```python
class DataPreprocessing:
    def __call__(self, batch_inputs_dict):
        for layer in self.pipeline:
            batch_inputs_dict |= layer(batch_inputs_dict)
        return batch_inputs_dict
```

它由 `BaseModel.on_after_batch_transfer()` 在「batch 已搬到目標 device、forward 之前」呼叫。具體層：`detection3d/point_pillar.py::PointPillarPreprocessor`（呼叫 voxelization op）、`segmentation3d/frustum_range.py`。

**設計原則（重要）**：只有「輸入側」前處理放這裡（voxelization 等）。**「輸出側」整形（logits→機率、voxel-to-point scatter、框解碼）一律放在 model 內**（`forward` / `compute_metrics` / `predict_outputs`），不是可設定的 framework pipeline。理由是避免「設定組合」與「指標正確性」之間出現看不見的耦合。

> 對照 AWML：取代 `data_preprocessor=dict(type="Det3DDataPreprocessor", voxel=True, ...)`。差別是它現在明確屬於 model、在 device 上跑，而且輸出側後處理被刻意排除。

---

## 6. Model 層（BaseModel 契約）

`models/base.py`：

```python
class BaseModel(MetricEvalMixin, L.LightningModule, ABC):
    @abstractmethod
    def forward(self, **kwargs): ...
    @abstractmethod
    def compute_metrics(self, batch_inputs_dict, outputs) -> dict[str, torch.Tensor]: ...  # 必含 "loss"
```

關鍵設計：

- `training_step` / `validation_step` / `test_step` / `predict_step` 都是 **`@final`**（不給覆寫）。它們走同一條 `_shared_step`。
- **簽名過濾**：`_shared_step` 用 `inspect.signature(self.forward)` 把 batch dict **只挑名字對得上的 key** 餵進 `forward`。所以你 `forward(self, points, gt_boxes)` 就會自動從 batch 拿 `points`、`gt_boxes`。
- **runtime preprocessing**：`on_after_batch_transfer` 呼叫注入的 `DataPreprocessing`（用 `set_data_preprocessing` 設，由進入點在 instantiate 後掛上）。
- **optimizer/scheduler**：`configure_optimizers()` 用 `_partial_` 的 factory，加上 `optimizer_group_overrides`（每個 parameter group 不同 LR，如 `block` / `encoder_block`）與 `scheduler_config`。
- **deploy hooks**：`build_export_spec(s)`、`get_export_output_names`、`prepare_export_outputs`，還有通用 `_PredictionExportWrapper`。
- **metrics**：由 `MetricEvalMixin` 管；model 只要實作 `build_eval_output(batch, outputs)` 把 forward 輸出攤平成 suite 讀的 flat dict。

**頂層模型**（`BaseModel` 子類）：

| 模型 | 檔案 |
| --- | --- |
| `TransFusionDetectionModel` | `models/detection3d/transfusion.py` |
| `BEVFusionDetectionModel` | `models/detection3d/bevfusion.py` |
| `PTv3BaseModel` → `PTv3SegmentationModel` / `PTv3DetectionModel` / `PTv3SegDetModel` | `models/segmentation3d/ptv3_base.py`、`segmentation3d/ptv3.py`、`detection3d/ptv3.py`、`multi/ptv3_segdet.py` |
| `FRNet` | `models/segmentation3d/frnet.py` |
| `CalibrationStatusClassifier` | `models/calibration_status/calibration_status.py` |

> **CenterPoint 目前只是一個 head**（`models/detection3d/heads/centerpoint.py`），還沒有頂層模型包裝。

**「Blocks」**：可重用的 `nn.Module` building block，散在各 model 檔內（沒有專門的 `blocks/` 目錄），例如 `SparseBasicBlock`（`encoders/sparse.py`）、`Block`/`PointModule`（`encoders/ptv3.py`）、`PTv3BEVResidualBlock`（`detection3d/ptv3.py`）。

**支援子套件**：`detection3d/task_modules/`（`assigners.HungarianAssigner3D`、`bbox_coders.TransFusionBBoxCoder`、`match_costs.{ClassificationCost,BBoxBEVL1Cost,IoU3DCost}`）、`detection3d/{encoders,backbones,necks,heads,view_transforms,fusion}`、`common/`（`backbones.resnet.ResNet18`、`necks.lss_fpn`、`heads.linear_cls_head`、`layers.conv`）。

一個模型 = 一堆 Hydra instantiate 出來的子模組組合。以 TransFusion 為例：`__init__(pts_voxel_encoder, pts_middle_encoder, pts_backbone, pts_neck, bbox_head)`，`forward` 串起來，`compute_metrics` 呼叫 `self.bbox_head.loss(...)`，`build_export_spec` 回傳只吃 tensor 的 `_TransFusionExportWrapper`。

> 對照 AWML：取代 `MVXTwoStageDetector` 家族的 `loss/predict/_forward` 三態。新契約更薄——只有 `forward` + `compute_metrics`，其餘靠 hook。

---

## 7. Ops 層（原生算子）

`autoware_ml/ops/` 是「執行圖或匯出圖的一部分」的低階算子——取代 mmcv ops。編譯只透過 `ops/build.py`，**目前只編一個 CUDA 擴充**：

| Op（路徑） | 是什麼 |
| --- | --- |
| `ops/bev_pool/`（`bev_pool.py` + `src/*.cpp,*.cu`） | **唯一需要編譯的 CUDA 擴充 `bev_pool_ext`**。`QuickCumsumTrainingCuda`（有 backward，訓練用）、`QuickCumsumCuda`（推論用，ONNX symbolic 發 `autoware::QuickCumsumCuda`）。camera-lidar BEV 模型用。 |
| `ops/voxelization/voxelization.py` | `hard_voxelize`——**純向量化 PyTorch GPU scatter，沒有 C++**，取代 mmcv `Voxelization`，刻意對齊 mmcv 的 ZYX 座標與 `round()` grid 慣例。 |
| `ops/spconv/`（`sparse_conv.py`…） | 包在外部 `spconv-cu120` 上的 deploy-aware wrapper：`SparseConv3d` / `SubMConv3d`，Native 與 implicit-GEMM 兩種執行計畫，ONNX 走 `Fsp_custom` bridge。 |
| `ops/segment/`（`segment_csr.py`） | `segment_csr`：eager 用 `torch.segment_reduce`，ONNX 發 `autoware::SegmentCSR`。 |
| `ops/indexing/operators.py` | `unique`（ONNX `autoware::CustomUnique`）、`argsort`（ONNX `autoware::Argsort`）——PTv3 export 用。 |

**共同模式**：eager 路徑用 stock torch/CUDA；在 `torch.onnx.is_in_onnx_export()` 時改發 `autoware::*` 自訂 symbolic，搭配 Autoware 端的 TensorRT plugin。這就是「訓練用 PyTorch、部署用自訂算子」兩條路對齊的做法。

> 需要改 ops 或第一次 build，一定要用 dev 環境：`pixi run --environment dev setup-project`（default 環境編不起來）。

---

## 8. Metrics 層

torchmetrics-based，兩層設計（`metrics/base.py`）：

- **`MetricSuite(torchmetrics.Metric)`**：任務的「狀態引擎」。擁有累積 state、跨 GPU reduction、per-range dispatch。它不決定「跑哪些指標」。
- **`Metric`**：小而自足、可注入的物件。從 suite 建好的 state 算出自己的數字，並宣告自己在哪些 `stages`（val/test）跑。

「跑哪些指標、在哪個 stage 跑」**純由 config 決定**。

- `metrics/eval_mixin.py::MetricEvalMixin`（混進 `BaseModel`）：epoch 開始 reset；每個 stage clone 一份 suite 當 submodule；每 batch `update`，epoch 末 `result`，log 成 `{stage}/{prefix}/{key}`。
- Detection：`detection3d/suite.py::Detection3DMetricSuite`（prefix `det3d`），components 有 `MeanAP` / `HeadingAP` / `Nds` / `TpErrors`，state 是 per-frame list（`dist_reduce_fx=None`，因為 mAP 配對要在 frame 內、score 排序）。
- Segmentation：`segmentation3d/suite.py::Segmentation3DMetricSuite`（prefix `seg3d`），components 有 `IoU` / `Accuracy` / `PrecisionRecallF1`，state 是一個堆疊的 confusion matrix（`dist_reduce_fx=sum`）。
- **range-aware**：設定 `ranges`（`MetricRange` 半徑帶），每個 key 會額外輸出各距離帶版本，如 `test/det3d/mAP_car_50m_90m`、`test/seg3d/iou_car_0m_50m`。
- model 只提供 `build_eval_output(batch, outputs)` 把 forward 輸出攤平成 `{"predictions", "gt_boxes", "gt_labels"}` 這種 flat dict；`update/compute/result` 都不由 model 呼叫。

**分散式**：loss 由 Lightning `sync_dist=True` 平均；metric 由 torchmetrics 依各 state 的 `dist_reduce_fx` 合併後才 `compute`。`autoware-ml test` **預設單卡**（無 padding、指標精確），要多卡評估才加 `--use-config-devices`。

> 對照 AWML：取代 `T4Metric(NuScenesMetric)`。差別是它拆成「引擎 suite + 可注入 metric」，且原生 range-aware。

---

## 9. Callbacks / Logger

- Callbacks 在 `configs/defaults/modules/callbacks.yaml` 以 `_target_` 掛：stock 的 `ModelCheckpoint`、`LearningRateMonitor`，加自家 `callbacks/early_stopping.py::EarlyStopping`。
- **`EarlyStopping(ConfigAuthoritativeStateMixin, LightningEarlyStopping)`**：resume 時**以目前 config 的值為準**（覆蓋 checkpoint 還原的 callback state），每個被覆蓋的欄位都會 log。這解決「改了 patience 卻被舊 checkpoint 蓋回去」的問題。
- Logger：`MLFlowLogger`（sqlite backend）。`experiment_name` / `run_name` / `run_id` / 預設 tag 都由框架在 runtime 自動填。產物結構：
  - `mlruns/<task>/<model>/<config>/<run_id>/hydra/`（Hydra scratch）
  - `mlruns/<task>/<model>/<config>/<run_id>/artifacts/`（checkpoints、resolved config 快照、run metadata）
- `utils/mlflow_helpers.py` + `mlflow_store.py` 管 run context / lineage / param-drift（resume 時記錄哪些 param 跟原 run 不同）。

---

## 10. Deployment 層

流程：`Checkpoint(.ckpt) → ONNX(.onnx) → TensorRT(.engine)`。指令 `autoware-ml deploy`，`utils/deploy.py`（`ExportSpec` / `infer_export_spec` / ONNX / TensorRT build）+ `utils/onnx_modifiers.py`。

1. **載入 checkpoint**：從 config instantiate model，`utils/checkpoints.py::apply_matching_weights` 把一個或多個 `--weights` 合併進去。
2. **取樣本**：用 predict dataloader 拿一筆前處理過的樣本當 example input。
3. **解析 export spec**：呼叫 model 的 `build_export_spec()` 取得「要匯出的 module + example 輸入 + 動態維度」。首選是**把 export 邏輯放在 model 內**（回傳只吃 tensor 的 wrapper），而非事後改 ONNX 圖。
4. **ONNX**：`torch.onnx.export`（`deploy.onnx.dynamo=true` 走 `torch.export`；`false` 走 legacy symbolic）。動態形狀由 `deploy.onnx.dynamic_shapes` 給。
5. **TensorRT**：用 TensorRT Python `Builder` API（不是 mmdeploy）依 `deploy.tensorrt.input_shapes`（min/opt/max）建 engine。

**multi-head 匯出**：PTv3 detection 能把 backbone 與 detection head 匯成不同 module；`--weights` 可依序合併「seg3d 預訓練 backbone checkpoint」+「det3d checkpoint」。**強制全覆蓋檢查**：所有 `--weights` 載完後，若 export model 有任何參數沒被覆蓋到，指令直接失敗並列出缺哪些 key（避免匯出含未訓練層的 engine）。

> 對照 AWML：取代 `mmdeploy` 的 `deploy_cfg`+`torch2onnx`+`to_backend`。ONNX 自訂算子改用 `autoware::*` symbolic。

---

## 檔案位置速查

| 你想找 | 檔案 |
| --- | --- |
| CLI 入口 / 子指令 | `autoware_ml/cli/cli.py` |
| CLI → Hydra/MLflow 接線 | `autoware_ml/cli/runtime.py` |
| train / test / deploy 進入點 | `autoware_ml/scripts/{train,test,deploy}.py` |
| dataset 生成進入點 | `autoware_ml/scripts/{create_dataset,generate_dataset}.py` |
| 全域執行時期預設 | `autoware_ml/configs/defaults/default_runtime.yaml` + `modules/*.yaml` |
| 任務 config | `autoware_ml/configs/tasks/<task>/<model>/*.yaml` |
| 自訂 Hydra resolver | `autoware_ml/configs/resolvers.py` |
| Dataset/DataModule 基底 | `autoware_ml/datamodule/base.py` |
| collation 策略 | `autoware_ml/datamodule/collation.py` |
| Polars/pydantic 資料層 | `autoware_ml/databases/`（`base_database.py`、`schemas/`、`t4dataset/`） |
| Transform 基底 | `autoware_ml/transforms/base.py` |
| GPU 前處理 | `autoware_ml/preprocessing/base.py` |
| 模型基底契約 | `autoware_ml/models/base.py` |
| 原生 ops | `autoware_ml/ops/`（`bev_pool` / `voxelization` / `spconv` / `segment` / `indexing`） |
| 指標引擎/契約 | `autoware_ml/metrics/base.py`、`metrics/eval_mixin.py` |
| checkpoint 合併（--weights） | `autoware_ml/utils/checkpoints.py` |
| ONNX/TensorRT export | `autoware_ml/utils/deploy.py`、`utils/onnx_modifiers.py` |
| runtime 工具（instantiate/seed/TF32/log） | `autoware_ml/utils/runtime.py` |

→ 實際動手做的流程與所有坑，看 [`03-workflow-and-gotchas.md`](03-workflow-and-gotchas.md)。
