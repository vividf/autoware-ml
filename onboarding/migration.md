# Migration Plan：AWML `deployment/` → autoware-ml

> **狀態**：Draft v1（2026-07-16）· 僅計畫、未動任何程式碼
> **範圍**：把 AWML 新版部署框架（`AWML/deployment/`，非舊 mmdeploy 路徑）的能力移植到 autoware-ml
> **來源盤點**：`AWML/deployment/`（CLI/config/export/execution/evaluation/verification/metrics/runtime/quantization/projects）
> **目標盤點**：`autoware-ml/autoware_ml/`（`scripts/deploy.py`、`utils/deploy.py`、`utils/onnx_modifiers.py`、`utils/checkpoints.py`、`ops/`、`metrics/`）

---

## 0. TL;DR

autoware-ml 目前的 deploy 是「**純匯出**」：checkpoint → per-module ONNX → TensorRT engine，掛在 MLflow deploy run 下。
AWML `deployment/` 比它多出四大能力，這就是 migration 的主體：

| # | 能力 | AWML deployment/ | autoware-ml 現況 |
| --- | --- | --- | --- |
| 1 | **Verification**（跨 backend 數值比對：torch vs ONNX vs TRT，tolerance 判定） | ✅ `BackendVerifier` + `OutputComparator` | ❌ 無 |
| 2 | **Backend Evaluation**（用 ONNX/TRT 實跑資料集算 mAP + latency，跨 backend 對照表） | ✅ `BackendExecutor` + `Detection3DEvaluator` | ❌ 無 |
| 3 | **Quantization**（INT8 PTQ/QAT：QDQ 插入、calibration、BN fusion、部分量化） | ✅ `quantization/` 引擎 + per-project plan | ❌ 無 |
| 4 | **Export 工程品質**（atomic publish、onnxsim、precision policy、TRT plugin 載入、graph 手術） | ✅ | 部分（有 modify_graph hook；無 atomic/simplify/plugin/precision policy） |

同時，AWML deployment/ 殘留的 mm 耦合（mmengine `Config`、`MODELS.build`、`load_checkpoint`、`T4MetricV2` 抽取、MMEngine `QATHook`）在移植時要全部換成 Hydra / Lightning 原生機制——這些耦合**只在邊緣**（config 解析與模型載入），核心引擎（exporter、orchestrator、verification、quantization engine）**幾乎是 mm-agnostic、可近乎原樣搬**。

建議分 **7 個 Phase**（0–6），每個 Phase 有獨立驗收條件，可獨立合併。

---

## 1. 目標與非目標

### 目標

1. autoware-ml 的 `deploy` 從「匯出工具」升級為「**匯出 → 驗證 → 評估** 的一條龍」，與 AWML deployment/ 能力對齊。
2. 引入 INT8 PTQ（優先）與 QAT（次要）量化能力，沿用 AWML 已驗證的 QDQ 設計。
3. 全程 **零 mm 依賴**：mmengine Config → Hydra/OmegaConf；`MODELS.build`+`load_checkpoint` → `hydra.utils.instantiate` + `apply_matching_weights`；MMEngine Hook → Lightning Callback。
4. 保留 autoware-ml 既有的 deploy 優點：MLflow lineage（deploy run 連回 training run）、multi-checkpoint `--weights` 合併 + 全覆蓋檢查、Hydra 統一 config。

### 非目標（本計畫不做）

- 不搬 AWML 舊 mmdeploy 路徑（`tools/detection3d/deploy.py`）——它已被 deployment/ 取代。
- 不搬 spconv INT8（AWML `spec.md` Goal 1 已裁定 sparse encoder 維持 FP16，只做 SparseConv+BN fold）。
- 不搬 AWML 的 13 個 ad-hoc quantization boolean flag config 介面——直接採 `spec.md` Goal 2 的宣告式介面（見 §5 Phase 4）。
- 不在本計畫內補齊 CenterPoint 的訓練側（autoware-ml 目前只有 head）；CenterPoint 的部署整合列為**相依項**（§7）。
- C++ TensorRT plugin 的建置流程（沿用 Autoware 端現有 `.so`，只做「載入」）。

---

## 2. 現況：兩邊架構速覽

### 2.1 Source：AWML `deployment/`

```
CLI（python -m deployment.cli.main <project> <deploy_cfg.py> <model_cfg.py>）
  └─ ProjectRegistry/ProjectAdapter（import 副作用註冊）
       └─ run_detection3d_deployment（共用 wiring）
            └─ BaseDeploymentRunner.run()
                 ├─ ExportOrchestrator      → OnnxExportPipeline（per-component ONNX）
                 │                            → TensorRTExportPipeline（per-component engine）
                 ├─ VerificationOrchestrator → BackendVerifier + OutputComparator
                 └─ EvaluationOrchestrator   → BackendExecutor → {PyTorch,ONNX,TRT} InferencePipeline
                                              → Detection3DEvaluator（perception_eval 核心）+ latency
quantization/（模型無關引擎）
  ├─ core/（QuantConv2d…、CalibrationManager(max/mse/entropy/percentile)、replace、fuse_bn）
  ├─ recipes/（依 class name 比對的 Q/DQ 擺放：BasicBlock/OSA/eSE/MaxPool…）
  ├─ schemes/（QuantizationScheme + QuantizationPlan；「同一個 build_<model>_plan」不變量）
  └─ projects/<model>/quantization/（plan.py、quantize.py=PTQ/QAT producer、qat_hook.py）
```

關鍵設計（值得原樣保留的）：

- **BackendExecutor seam**：evaluation 與 verification 共用同一個「backend 抽象」，per-project 只實作 executor + inference pipelines。
- **Per-component 匯出**：CenterPoint 拆 `pts_voxel_encoder` + `pts_backbone_neck_head` 兩個 ONNX（middle encoder 留在 PyTorch）；BEVFusion 拆 sparse（FP16）+ dense（INT8），可再 merge 成單一 ONNX。
- **Atomic artifact publish**：ONNX 先寫 `.staging`、engine 先寫 `.tmp`+`fsync`，`os.replace` 收尾——永不留半成品。
- **INT8 = QDQ-in-ONNX**：不是 TRT builder flag，是把 Q/DQ 節點烙進 ONNX（`use_fb_fake_quant=True`），TRT 端 `precision_policy` 維持 `fp16`。
- **量化「神聖不變量」**：PTQ producer、deploy loader、QAT hook 三方都用**同一個** `build_<model>_plan(config).prepare(model)` 建樹，讓 calibrated state_dict 與部署載入天然對齊。
- **Verification 語意**：ref-vs-test 依 scenario（如 `pytorch(cpu) vs onnx(cpu)`、`onnx(cuda) vs trt(cuda)`），判定 `max_diff <= tolerance`（絕對值，預設 0.1）。
- **Evaluation 與訓練同一把尺**：`Detection3DEvaluator` 用 `autoware_perception_evaluation`（= T4MetricV2 核心），GT 過濾（points-in-box ≥ min_num_points）刻意跟 `test.py` 一致，mAP 可直接對照。

殘留 mm 耦合（移植時要替換的邊緣）：`Config.fromfile`（config 全部）、`build_mmdet3d_model`（`MODELS.build`+`init_default_scope`+`load_checkpoint`）、ONNX-variant 類別註冊進 mmdet3d registry、`extract_t4metric_v2_config`（讀 model_cfg 的 `val_evaluator`）、`PointCloudDataLoader`（mmdet3d pcd pipeline）、QAT 的 MMEngine `Runner`/`Hook`。

### 2.2 Target：autoware-ml deploy

- `autoware-ml deploy --config-name <cfg> --weights <ckpt>...` → `scripts/deploy.py`（`@hydra.main`）。
- 流程：weights 檢查 → MLflow lineage（deploy run 連回 training run）→ `instantiate(model/datamodule)` → `apply_matching_weights`（**強制全參數覆蓋**）→ `resolve_export_specs()`（model 的 `build_export_specs(batch)` 回傳 `{module_name → ExportSpec}`）→ 每個 module：ONNX（dynamo/legacy 雙路徑 + 可選 `modify_graph`）→ TRT engine。
- TRT builder 用 **`STRONGLY_TYPED`** network flag（精度由 ONNX dtype 決定）+ optimization profile + workspace。
- 已具備且要保留：per-module ONNX config 合併（`merge_module_onnx_cfg`）、`dynamic_shapes`（dynamo）/`dynamic_axes`（legacy）雙格式、外部資料 shard 合併、輸出強制在 MLflow artifact 目錄內。
- 相依套件已就位：`onnx`、`onnx_graphsurgeon`、`onnxruntime-gpu`、`onnxscript`、`tensorrt-cu12`。**缺**：`pytorch_quantization`（NVIDIA）、`onnxsim`（若要 simplify）、`autoware_perception_evaluation`（若選用，見 §6 D2）。

---

## 3. 概念對應（Source → Target）

| AWML deployment/ | autoware-ml 對應 | 策略 |
| --- | --- | --- |
| `cli/main.py` + `ProjectRegistry`/`ProjectAdapter` | `autoware-ml deploy`（Typer）+ Hydra config；**不需要 registry**——per-model 行為由 `BaseModel` hook（`build_export_specs` 等）承載 | 以既有 CLI 為準，捨棄 project registry |
| mmengine `deploy_cfg.py` + `model_cfg.py` 兩份 config | **單一 Hydra task config** 的 `deploy:` 區段（訓練/部署同一份 config，本來就是 autoware-ml 的設計） | 重寫成 Hydra schema |
| `BaseDeploymentConfig` + frozen dataclasses（`ExportConfig`/`ComponentsConfig`/…） | Hydra `deploy:` 區段 + pydantic/dataclass 驗證層（沿用 frozen dataclass 型別，改由 OmegaConf 餵） | 型別搬過來、載入層重寫 |
| `ComponentsConfig`（per-component io/onnx_file/engine_file/trt_profile） | 既有 `deploy.onnx.modules.<name>` 的擴充（補 `tensorrt.modules.<name>` per-module profile） | 合併進既有 per-module 機制 |
| `ExportOrchestrator` → `OnnxExportPipeline`/`TensorRTExportPipeline` | 既有 `scripts/deploy.py` 主迴圈 + `utils/deploy.py`；把 AWML 的工程品質（atomic、simplify、plugin、precision policy）補進來 | 增強既有實作，不另起爐灶 |
| `ModelComponentBuilder`/`SampleExtractor` | `BaseModel.build_export_specs(batch)`（**已存在且等價**：回傳 module+args per component） | 已對齊，不搬 |
| `BaseModelWrapper`/ONNX-variant 類別（registry 註冊） | model 內的 export wrapper（如 `_TransFusionExportWrapper`），純 Python 類別 | 已對齊，不搬 |
| `BackendExecutor` + per-backend `InferencePipeline` | **新增** `autoware_ml/deployment/execution/`（見 §4） | 移植（去 mm 化） |
| `BackendVerifier` + `OutputComparator` | **新增** `autoware_ml/deployment/verification/` | 幾乎原樣移植（mm-agnostic） |
| `BaseEvaluator` + `Detection3DEvaluator` + `Detection3DMetricsInterface` | **新增** `autoware_ml/deployment/evaluation/`；指標接 autoware-ml 既有 `Detection3DMetricSuite`（torchmetrics），`perception_eval` 為選配 | 移植 + 換指標後端（§6 D2） |
| `ArtifactManager` | 移植（解析優先序：registered > eval-backend config > export 輸出） | 移植 |
| `inference/tensorrt_runner.py` + `GPUResourceMixin` | **新增** `autoware_ml/deployment/runtime/`（TRT engine 載入/執行/CUDA-event 計時） | 幾乎原樣移植 |
| `primitives/`（`DeviceSpec`/`Artifact`/`LatencyStats`…） | **新增** `autoware_ml/deployment/primitives/` | 原樣移植 |
| `quantization/`（core/recipes/schemes/sparse） | **新增** `autoware_ml/quantization/`（或 `deployment/quantization/`） | 引擎原樣移植；config 介面改宣告式（spec.md Goal 2） |
| `projects/<model>/quantization/plan.py` | `autoware_ml/models/<task>/<model>` 對應的 `build_<model>_plan` | 每模型重接 |
| `qat_hook.py`（MMEngine Hook） | Lightning **Callback**（`on_fit_start` = plan.prepare；首個 epoch 前 calibrate/load cache） | 重寫（薄層） |
| `quantize.py`（PTQ/QAT producer CLI） | 新子指令 `autoware-ml quantize`（Typer + Hydra，同 train/deploy 模式） | 重寫（薄層） |
| `io/mmdet3d_model.py::build_mmdet3d_model` | `hydra.utils.instantiate(cfg.model)` + `apply_matching_weights` | **刪除**（被既有機制取代） |
| `PointCloudDataLoader`（mmdet3d pipeline） | 既有 datamodule 的 `predict_dataloader()`（deploy 已在用） | **刪除** |
| `extract_t4metric_v2_config`（讀 mm `val_evaluator`） | Hydra config 的 `model.metrics` / `deploy.evaluation.metrics` | **刪除**（config 原生化） |
| `cli/args.py` 的 absl logging hijack workaround | 移植時重新驗證 `pytorch_quantization` import 副作用，必要時保留對策 | 視情況 |

---

## 4. 目標套件佈局（提案）

把 deploy 從 `utils/` 升格為一級子套件（維持 `ops/README.md` 同款「設計規則」文件）：

```
autoware_ml/
├── deployment/
│   ├── primitives/        # DeviceSpec, Artifact, LatencyStats, ModelSpec, InferenceInput
│   ├── export/            # 現 utils/deploy.py 遷入：ExportSpec、onnx/trt export
│   │                      #   + atomic publish、simplify、precision policy、plugin 載入
│   ├── execution/         # BackendExecutor（ABC）、PointCloudBackendExecutor
│   ├── runtime/           # tensorrt_runner（load/run/CUDA-event 計時）、GPUResourceMixin、ArtifactManager
│   ├── verification/      # BackendVerifier、OutputComparator、reporting
│   └── evaluation/        # BaseEvaluator、Detection3DEvaluator（接 torchmetrics suite）
├── quantization/
│   ├── core/              # modules(QuantConv2d…)、descriptors、replace、calibration、fusion
│   ├── recipes/           # attach.py、forward_hooks.py（class-name 比對、idempotent）
│   ├── schemes/           # QuantizationScheme、QuantizationPlan、DenseQDQScheme
│   └── sparse/            # fuse_spconv_bn_in_encoder
└── scripts/
    ├── deploy.py          # 擴充：export 後接 verification → evaluation（各自可關）
    └── quantize.py        # 新：PTQ producer（QAT 走 train + callback）
```

Per-model 的接點維持在 model 檔案旁（不是獨立 project bundle）：

- export 拆件：`BaseModel.build_export_specs()`（已有）
- 量化：`autoware_ml/models/<task>/<model>_quant.py` 內 `build_<model>_plan(config)`
- backend 推論：`autoware_ml/deployment/execution/` 下 per-model executor（或 model hook `build_backend_pipeline(backend, artifacts)`——Phase 3 定案，見 §6 D3）

Config（Hydra `deploy:` 區段擴充，示意）：

```yaml
deploy:
  onnx:      { enabled: true, dynamo: true, opset_version: 21, simplify: false, modules: {...} }
  tensorrt:
    enabled: true
    precision_policy: strongly_typed   # auto | fp16 | fp32_tf32 | strongly_typed（新增）
    workspace_size: 4294967296
    plugin_libraries: []               # 新增：dlopen 自訂 plugin .so
    modules: {...}                     # 新增：per-module min/opt/max profile
  verification:                        # 新增
    enabled: true
    num_samples: 5
    tolerance: 0.1
    scenarios:
      - { ref: {backend: pytorch, device: cuda}, test: {backend: tensorrt, device: cuda} }
  evaluation:                          # 新增
    enabled: true
    num_samples: -1
    num_warmup: 5
    backends: { pytorch: {enabled: true}, onnxruntime: {enabled: true}, tensorrt: {enabled: true} }
  quantization:                        # 新增（宣告式，spec.md Goal 2）
    enabled: false
    mode: ptq                          # ptq | qat
    default_precision: int8
    keep_fp16: []                      # subtree globs，取代 13 個 boolean flag
    disable_recipes: []
    calibration: { method: mse, num_batches: 64, cache: null }
    fuse_bn: true
```

---

## 5. 分階段計畫

> 每個 Phase 可獨立 PR/合併；標注規模（S/M/L/XL）與驗收條件。Phase 1–2 無新依賴，可先行。

### Phase 0 — 基準與腳手架（S）

1. 在 AWML 端**凍結基準數字**：CenterPoint / BEVFusion-L 的 torch/ONNX/TRT mAP（含 INT8 基準 0.3228 / 0.3931，出處 `spec.md`）、latency 分布、verification tolerance 實測值。存成表格（本檔附錄或獨立 baseline.md）。
2. autoware-ml 端建立 `autoware_ml/deployment/` 空骨架 + README（設計規則、依賴方向表——照抄 AWML `docs/architecture.md` 的 allowed-dependency 表精神：Runner→Evaluator/Verifier；Evaluator/Verifier→Executor；Executor→Pipelines；Metrics 不依賴 Runner/Pipelines）。
3. 把 `utils/deploy.py` 遷至 `deployment/export/`（純搬家 + re-export shim，不改行為）。

**驗收**：現有 `autoware-ml deploy` 行為不變（TransFusion/FRNet/PTv3 匯出結果 byte-level 或 hash 一致）。

### Phase 1 — Export 工程品質對齊（M）

從 AWML `export/exporters/` 移植（全部 mm-agnostic）：

1. **Atomic publish**：ONNX 經 `.staging` 目錄（先搬 external-data sidecar、最後 `os.replace` 主檔）；engine 經 `.tmp`+`fsync`+`os.replace`。
2. **`onnxsim.simplify`** 選配（`deploy.onnx.simplify`，同樣 staged）。新增 `onnxsim` 依賴（optional extra）。
3. **PrecisionPolicy enum**（`auto|fp16|fp32_tf32|strongly_typed`）：現行為（STRONGLY_TYPED）設為預設值，維持向後相容；`fp16`/`tf32` 走 `BuilderFlag`。注意：`STRONGLY_TYPED` 是 network-creation flag、與 builder flag 互斥，照 AWML `_apply_precision_policy` 的處理。
4. **TRT plugin 載入**：`deploy.tensorrt.plugin_libraries` → dlopen + `init_libnvinfer_plugins`（BEVFusion spconv plugin 前置需求）。
5. **Per-module TRT profile**：`deploy.tensorrt.modules.<name>.input_shapes`（現在 profile 是全域的）。
6. （選配）`make_qdq_readable`（Q/DQ scale 可視化）——為 Phase 4 鋪路。

**驗收**：kill -9 匯出中途不留半成品；simplify 開關可用；`precision_policy=fp16` 能建出 fp16 engine；單元測試移植 `test_onnx_exporter.py`（atomic 三案例）。

### Phase 2 — Verification（跨 backend 數值比對）（M）

移植 `verification/`（`BackendVerifier`、`OutputComparator`、`OutputDiffSummary`/`TensorDiffDetail`）與 `runtime/tensorrt_runner.py`、`primitives/`：

1. `OutputComparator`：遞迴 diff 結構化輸出、`max_diff <= tolerance` 絕對判定、named-output 標籤——**原樣移植**（有現成測試 `test_output_comparator.py` 一起搬）。
2. 最小 backend 執行能力：pytorch（直接呼叫 export module）、onnxruntime（`onnxruntime-gpu` 已在依賴）、tensorrt（`tensorrt_runner`）。此時不需要完整 `BackendExecutor`——verification 只要能餵**同一個 export sample** 給兩邊。
3. `scripts/deploy.py` 在匯出後跑 `deploy.verification.scenarios`；失敗預設 fail run（可設 warn-only）。
4. 結果寫進 MLflow deploy run（per-scenario max/mean diff、pass/fail tag）。

**驗收**：TransFusion / FRNet 匯出後自動得到 torch-vs-ORT、ORT-vs-TRT 的 diff 報告；故意打壞 tolerance 會 fail。

### Phase 3 — Backend Evaluation（mAP + latency）（L）

移植 `execution/` + `evaluation/`，去 mm 化：

1. `BackendExecutor`（ABC）+ `PointCloudBackendExecutor`：`prepare_input` 改吃 autoware-ml datamodule 的 predict batch（取代 `PointCloudDataLoader`）。
2. Per-backend `InferencePipeline`（preprocess→run→postprocess）：pytorch pipeline 直接用 `BaseModel.predict_outputs()`；TRT pipeline 用 `tensorrt_runner` + CUDA-event 分段計時；ORT pipeline 比照。
3. `BaseEvaluator`：warmup → 逐 sample 推論 + latency → 收集 pred/GT；`compute_latency_stats`（mean/std/min/max/median）與 per-stage breakdown 原樣搬。
4. **指標後端**：接 autoware-ml 既有 `Detection3DMetricSuite` / `Segmentation3DMetricSuite`（torchmetrics）——backend eval 與 `autoware-ml test` 同一把尺（§6 D2）。GT 過濾規則（min points-in-box）需在新指標側對齊。
5. `EvaluationOrchestrator` 的跨 backend 對照表（mAP per backend side-by-side + latency）進 log 與 MLflow。
6. CLI：`deploy.evaluation.enabled`（預設 off 以免 deploy 變慢），或獨立 `autoware-ml eval-backends --config-name ... --artifacts-dir ...`（Phase 內定案）。

**驗收**：TransFusion 在 nuScenes val 子集上，pytorch/ORT/TRT 三 backend 的 mAP 差 < 0.5pt（fp16 預期範圍），latency 表可重現；`test` 指令的 mAP 與 pytorch-backend eval 的 mAP 一致（同一 suite、同一過濾）。

### Phase 4 — Quantization 引擎（XL，核心）

移植 `quantization/`（core/recipes/schemes/sparse），**引擎原樣、介面重造**：

1. **依賴**：新增 `pytorch_quantization`（NVIDIA）。先驗證與 torch 2.9.1+cu128 相容性（AWML 在 torch 2.8 環境使用）；不相容則評估 `nvidia-modelopt`（設計本就 modelopt-style，遷移成本低）——**這是本計畫最大的技術風險**（§8 R1）。
2. **core/ 原樣移植**：`QuantConv2d`/`QuantConvTranspose2d`/`QuantLinear`、descriptors（per-channel Conv2d / per-tensor ConvTranspose2d / per-row Linear / histogram activations）、`replace.py`、`CalibrationManager`（max/mse/entropy/percentile + `.calib` cache）、`fuse_model_bn`。
3. **recipes/ 原樣移植**且**改為 always-on**（採 `spec.md` Goal 3/R2 的裁定：class-gated recipe 是「架構的正典描述」，比對不到就 no-op，不需要 config 開關）。
4. **schemes/ 原樣移植**，並守住**神聖不變量**：PTQ producer、deploy loader、QAT callback 三方都經同一個 `build_<model>_plan(config).prepare(model)`。
5. **Config 介面採宣告式**（spec.md Goal 2，hard-cut）：`default_precision` + `keep_fp16`（subtree glob）+ `disable_recipes`，**不移植** `quant_backbone/neck/head/...`、`skip_*`、`sensitive_layers` 等 13 個 flag。注意 AWML `replace.py` 的 skip 是「整名精確比對→跳過整棵子樹」，glob 化時要明確定義語意並補測試（spec.md R3 footgun）。
6. **PTQ producer**：新 `autoware-ml quantize --config-name <cfg> --weights <ckpt>`（scripts/quantize.py）：instantiate model → `apply_matching_weights` → plan.prepare → 用 train/val dataloader calibrate → 存 quantized state_dict + `.calib` cache（MLflow artifacts）。
7. **Deploy 端載入**：`scripts/deploy.py` 在 `deploy.quantization.enabled` 時，於 `apply_matching_weights` **之前** plan.prepare（同一棵樹），export 前設 `TensorQuantizer.use_fb_fake_quant=True` 讓 Q/DQ 進 ONNX。TRT 端 precision_policy 維持 `fp16`（INT8 由 QDQ 驅動，不是 builder flag）——照 AWML 語意。
8. **QAT**（次要、可延後）：Lightning Callback 版 `QATHook`（`on_fit_start` prepare；首 epoch 前 calibrate/load cache），配 `autoware-ml train --config-name ... +quantization.mode=qat`。

**驗收**：單元測試覆蓋 replace/skip 語意、plan idempotency、calib cache round-trip；e2e：任一已支援模型（建議 TransFusion 或 BEVFusion dense 塔）PTQ INT8 的 mAP 相對 fp16 降幅在容許內（以 Phase 0 基準的降幅為對照），且 deploy-load 的 state_dict 與 PTQ 產物 key 完全對齊。

### Phase 5 — Per-model 整合（L，依模型逐一）

優先序依「autoware-ml 訓練側成熟度 × 產品需求」：

1. **BEVFusion-L**（訓練側已有 `BEVFusionDetectionModel`）：
   - `build_export_specs` 擴成 sparse/dense 兩件（對應 `BEVFusionSparseWrapper`/`BEVFusionDenseWrapper`）；dense 的 trace input 由 sparse 前段產生（AWML `BEVFusionComponentBuilder` 的做法）。
   - 移植 graph 手術（`onnx_graphsurgeon` 已在依賴）：`fix_topk_constant_k`、`fuse_autoware_implicit_gemm_trailing_relu`、`fuse_spconv_bn_in_encoder`、split→merge（`onnx.compose.merge_models`）——掛成該 model 的 `modify_graph`/finalize。
   - TRT eval 需 `plugin_libraries`（Phase 1 已備）。ONNX-runtime backend 對 sparse 圖**不支援**（AWML 同）；executor 宣告 supported backends。
   - `build_bevfusion_plan`（DenseQDQ + SpconvBnFuse）接上。
2. **TransFusion / PTv3 / FRNet**（AWML deployment 沒有、但 autoware-ml 已能匯出）：直接受益於 Phase 2/3（verification + backend eval），量化視需求接 plan。
3. **CenterPoint**：**相依於訓練側補齊 top-level model**（目前只有 head）。部署拆件（voxel encoder ONNX + backbone/neck/head ONNX、middle encoder 留 PyTorch）與 `build_centerpoint_plan` 在 model 就緒後照搬。若產品時程要求先行，備選：提供「AWML checkpoint → autoware-ml state_dict」的一次性轉換器（風險與成本另評）。

**驗收（每模型）**：torch/TRT mAP 差在 Phase 0 基準的同等範圍；INT8（若接）相對 fp16 降幅 ≤ AWML 基準降幅；verification 全 scenario pass。

### Phase 6 — 收斂與退役（M）

1. AWML `deployment/` 標記 maintenance-only（README banner）；新模型部署一律走 autoware-ml。
2. 文件：`docs/user-guide/deployment.md` 擴寫 verification/evaluation/quantization 三章；把 AWML `deployment/docs/architecture.md` 的依賴規則表、`quantization/README.md` 的不變量說明改寫收錄。
3. 測試遷移完成度盤點（AWML `deployment/tests/` 的 7 個檔案應全數有對應）。
4. 基準對照報告：新舊框架各 backend mAP/latency 對照表，作為退役依據。

---

## 6. 關鍵設計決策（含建議）

| # | 決策 | 選項 | 建議 |
| --- | --- | --- | --- |
| D1 | **CLI 形態** | (a) 全塞進 `autoware-ml deploy`（config 開關）；(b) 拆 `deploy` / `eval-backends` / `quantize` 三指令 | **(b) 偏向拆**：deploy 保持快（預設只 export+verify），backend eval 跑資料集很慢、quantize 是 producer——職責不同。三者共用同一份 task config。 |
| D2 | **backend eval 的指標後端** | (a) autoware-ml torchmetrics suite；(b) 移植 `perception_eval`（T4MetricV2 核心）；(c) 兩者並存 | **(a) 為主**：train/test/deploy-eval 同一把尺、零新依賴。遷移驗證期若需與 AWML 舊發布模型直接對數，再以 optional extra 掛 (b)。 |
| D3 | **per-model backend 推論的接點** | (a) `deployment/execution/` 下獨立 executor 類別（AWML 式）；(b) `BaseModel` 新增 hook（如 `build_backend_pipeline`） | 傾向 **(b)**：與 `build_export_specs` 對稱、符合「export 邏輯住在 model 內」的既有原則；Phase 3 開工時以 TransFusion 打樣後定案。 |
| D4 | **量化 config 介面** | (a) 照搬 13 flags；(b) spec.md Goal 2 宣告式 | **(b)**，hard-cut。AWML 自己的 spec 已裁定方向，migration 是唯一一次免費改介面的機會。 |
| D5 | **quantization 套件位置** | (a) `autoware_ml/quantization/`（一級）；(b) `deployment/quantization/` | **(a)**：QAT 會被 train 側 callback 使用，不只 deploy。 |
| D6 | **pytorch_quantization vs modelopt** | 見 §8 R1 | Phase 4 第 1 步做 spike 後定案；引擎抽象（scheme/plan）兩者皆可承載。 |

---

## 7. 相依與前置

- **CenterPoint top-level model**（autoware-ml 訓練側）：Phase 5.3 的硬前置。
- **`pytorch_quantization` × torch 2.9.1/cu128 相容性 spike**：Phase 4 的硬前置（1–2 天）。
- **Autoware 端 TensorRT plugin `.so`**（spconv ImplicitGemm 等）：BEVFusion TRT eval 需要；只載入、不負責建置。
- 新增依賴（提案為 optional extras）：`pytorch_quantization`（或 `nvidia-modelopt`）、`onnxsim`；`perception_eval` 僅在 D2(b) 啟用時。

---

## 8. 風險與緩解

| # | 風險 | 影響 | 緩解 |
| --- | --- | --- | --- |
| R1 | `pytorch_quantization` 與 torch 2.9/cu128 不相容（AWML 在 2.8 驗證） | Phase 4 卡死 | 先 spike；備案 `nvidia-modelopt`（介面本就對齊）；最壞自帶 fake-quant modules（core/modules 已是自寫子類，僅 `TensorQuantizer` 需替代） |
| R2 | torch 2.9 的 dynamo export 與 QDQ 圖互動未知（AWML 走 legacy exporter 為主） | INT8 ONNX 產出失敗 | 量化匯出路徑先固定 `dynamo=false`（既有 legacy 路徑健在）；dynamo+QDQ 另開 spike |
| R3 | `STRONGLY_TYPED` 與 `FP16` builder flag 互斥的語意混淆 | engine 精度不符預期 | PrecisionPolicy enum 明文互斥 + config 驗證（照 AWML `_apply_precision_policy`）；文件寫清楚 INT8=QDQ 不是 policy |
| R4 | 指標換後端（perception_eval → torchmetrics）造成 mAP 定義差異，parity 對不上 | 驗收爭議 | Phase 0 就在**同一組預測**上跑兩種指標，量化定義差；驗收以「同框架內 torch-vs-TRT 差」為主，不跨框架比絕對值 |
| R5 | skip/keep_fp16 語意從「精確比對整子樹」變 glob 時行為漂移（spec.md R3） | 量化到不該量化的層、mAP 掉 | 語意明文化 + 針對性單元測試 + `ptq_accuracy_vov99.md` 的案例當回歸 |
| R6 | AWML checkpoint 與 autoware-ml 模型 state_dict key 不相容（模型是重寫的） | 不能直接拿舊模型驗 parity | 驗收一律用 autoware-ml 自己訓的 checkpoint；舊模型對數僅供參考（或做一次性 key-map 轉換器） |
| R7 | BEVFusion sparse ONNX 依賴自訂 plugin，ORT backend 無法驗證 sparse 段 | verification 覆蓋缺口 | 沿用 AWML 策略：sparse 段以 torch-vs-TRT 驗證；executor 明確宣告 supported backends |
| R8 | deploy 指令變慢（verify+eval 全開） | 開發體驗 | 預設 verify=on（樣本數小）、eval=off；eval 拆獨立指令（D1） |

---

## 9. 驗收總表

| 項目 | 標準 |
| --- | --- |
| Export 回歸 | Phase 0 錄的既有模型匯出產物 hash/行為不變 |
| Atomicity | 中斷測試不留半成品（移植 `test_onnx_exporter.py` 全綠） |
| Verification | 每模型全 scenario `max_diff <= tolerance`；故障注入會 fail |
| Backend eval | pytorch backend mAP ≡ `autoware-ml test` mAP（同 suite）；TRT fp16 相對 torch 差 < 既定門檻 |
| Quantization | plan 三方同樹（producer/loader/QAT）state_dict key 全對齊；INT8 mAP 降幅 ≤ AWML 基準降幅（0.3228/0.3931 對照組） |
| Latency | 每 backend latency 統計可重現、進 MLflow |
| 零 mm | `grep -rE "mmengine|mmdet|mmcv"` 在新增碼中為 0 |
| 測試遷移 | AWML `deployment/tests/` 7 檔全數有對應或明確豁免 |

---

## 附錄 A：AWML deployment/ 關鍵類別索引（移植時查表用）

| 關注點 | 類別/函式 | AWML 檔案 |
| --- | --- | --- |
| 共用 wiring | `run_detection3d_deployment` | `runtime/detection3d_entrypoint.py` |
| Orchestration | `BaseDeploymentRunner`、`ExportOrchestrator`、`EvaluationOrchestrator`、`VerificationOrchestrator`、`ArtifactManager` | `runtime/*.py` |
| 型別 config | `BaseDeploymentConfig` + `ExportConfig`/`ComponentsConfig`/`TensorRTConfig`/`QuantizationConfig`/`EvaluationConfig`/`VerificationConfig`；enums `Backend`/`ExportMode`/`PrecisionPolicy` | `config/base.py`、`config/schema.py`、`config/enums.py` |
| Exporters | `ONNXExporter`（atomic+simplify）、`TensorRTExporter`（precision policy+plugins+profiles） | `export/exporters/*.py` |
| Export pipelines | `OnnxExportPipeline`、`TensorRTExportPipeline`、`ModelComponentBuilder`、`SampleExtractor` | `export/pipelines/*.py` |
| Execution seam | `BackendExecutor`、`PointCloudBackendExecutor` | `execution/*.py` |
| Eval | `BaseEvaluator`、`Detection3DEvaluator`、`Detection3DMetricsInterface`、`extract_t4metric_v2_config` | `evaluation/*.py`、`metrics/detection_3d_metrics.py` |
| Verify | `BackendVerifier`、`OutputComparator` | `verification/*.py` |
| TRT runtime | `load_trt_engine`、`run_trt_engine`、`GPUResourceMixin` | `inference/tensorrt_runner.py`、`inference/gpu_resource_mixin.py` |
| mm 耦合核心（要刪） | `build_mmdet3d_model` | `io/mmdet3d_model.py` |
| Quant 引擎 | `QuantizationScheme`/`QuantizationPlan`、`CalibrationManager`、`quant_conv_module`、`attach_quant_add`、`fuse_model_bn` | `quantization/**` |
| CenterPoint | `CenterPointDeploymentRunner`、`CenterPointComponentBuilder`、`build_centerpoint_model`、`CenterPointExecutor`、`build_centerpoint_plan` | `projects/centerpoint/**` |
| BEVFusion | `BEVFusionDeploymentRunner`、`BEVFusionComponentBuilder`、`build_bevfusion_model`、`BEVFusionExecutor`、`bevfusion_merge_finalize`/`fix_topk_constant_k`、`build_bevfusion_plan` | `projects/bevfusion_l/**` |

## 附錄 B：參考文件

- AWML：`deployment/docs/architecture.md`（依賴規則、project layout contract）、`deployment/docs/MIGRATION_FROM_OLD_FRAMEWORK.md`（Executor seam 的由來）、`deployment/quantization/README.md`（引擎設計與不變量）、`spec.md`（量化重構 spec：Goal 1 已做 / Goal 2–3 設計，本計畫直接採納）
- autoware-ml：`docs/user-guide/deployment.md`、`autoware_ml/utils/deploy.py`、`autoware_ml/scripts/deploy.py`、`autoware_ml/ops/README.md`（`autoware::*` ONNX symbolic 慣例）
