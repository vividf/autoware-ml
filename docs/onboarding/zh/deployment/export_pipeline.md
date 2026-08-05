# 匯出流程 (Export Pipeline)

> **本文涵蓋內容：** 訓練好的 checkpoint 如何變成可部署的產出物（artifact） — `autoware-ml
> deploy` 流程、`ExportSpec` / `build_export_specs` 合約、多模組（multi-module）與多 head 匯出，
> 以及權重合併（weight merging）。ONNX/TensorRT 的內部機制請見
> [onnx_tensorRT.md](onnx_tensorRT.md)。
>
> 先備知識：[../model/model_architecture.md](../model/model_architecture.md)、
> [../code_walkthrough/entry_point.md](../code_walkthrough/entry_point.md)。

---

## 1. 為何部署是一個一等公民（first-class）、由模型擁有的步驟

Autoware 在車輛上是透過 **TensorRT** 來執行感知（perception）的。如果匯出（export）邏輯獨立
存在於另一個腳本中，就有可能與訓練好的模型逐漸產生落差（drift）。因此 autoware-ml 將匯出納入
模型合約（contract）的一部分：模型透過 `build_export_specs()` 宣告*要匯出什麼*，而框架則負責
執行 ONNX → TensorRT 的轉換。**同一份 config** 同時驅動 train/test/deploy，因此可以保證匯出的
圖（graph）與訓練時的架構完全一致。

```text
checkpoint(.ckpt) ──load──▶ model (eval) ──predict batch──▶ build_export_specs()
                                                                 │  {module_name: ExportSpec}
                                                                 ▼
                                        per module:  torch.onnx.export ──▶ .onnx
                                                     (optional graph modify)
                                                     TensorRT build ──▶ .engine
```

---

## 2. 指令

```bash
autoware-ml deploy \
    --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes \
    --weights mlruns/.../checkpoints/best.ckpt
```

- `--config-name` — 與訓練時使用的**同一份**任務 config。
- `--weights` — 一個或多個 `.ckpt` 路徑，其參數會被合併進匯出用的模型中（可重複指定；若 key
  重疊，後面指定的會覆蓋前面的）。這是提供參數的*唯一*方式。
- 選項：`output_name=<name>`、`output_dir=<path>`，以及像 `deploy.tensorrt.enabled=false`
  這類階段開關（stage toggle）。

分派（dispatch）方式與 train/test 完全相同：`cli.py deploy` → `run_hydra_entrypoint`（stage
為 `deploy`）→ `scripts/deploy.py:main`（`@hydra.main`）。詳見
[../code_walkthrough/entry_point.md](../code_walkthrough/entry_point.md)。

---

## 3. `scripts/deploy.py:main` 逐步解析

```python
weight_paths = [...]                                   # from cfg.weights, all must exist
checkpoint_path = weight_paths[-1]
# ... MLflow deploy-run + lineage (resolve_deploy_lineage) linking to the source training run ...

validate_cuda_available(); configure_torch_runtime()  # TensorRT needs CUDA → device = cuda   :149-152
output_dir, _, _ = resolve_output_paths(checkpoint_path, cfg.get("output_name"), configured_output_dir)   # :159
# when MLflow logging is on, output_dir MUST stay inside the run artifact dir                 :164

datamodule = hydra.utils.instantiate(cfg.datamodule)  # :183
model      = hydra.utils.instantiate(cfg.model)        # :186
model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))

apply_matching_weights(model, weight_paths, map_location=device, device=device,
                       set_eval=True, enforce_full_coverage=True, ...)   # :192  load + eval + full-coverage check

export_specs = resolve_export_specs(datamodule, model, device)          # :203  {module_name: ExportSpec}

for module_name, export_spec in export_specs.items():                   # :207
    module_onnx_cfg = merge_module_onnx_cfg(deploy_cfg.onnx, module_name)   # shared + per-module overrides
    if should_export_stage(deploy_cfg.onnx):
        if not supports_export_stage(export_spec, "onnx"): raise ...     # stage vs model support
        export_to_onnx(export_spec.module, export_spec.args, module_onnx_cfg,
                       export_spec.input_param_names, export_spec.output_names,
                       export_spec.dynamic_axes, output_dir / f"{module_name}.onnx")   # :219
        if should_modify_graph(module_onnx_cfg.get("modify_graph")):
            module_onnx_path = modify_onnx_graph(module_onnx_path, ...)   # optional graph edit
    if should_export_stage(deploy_cfg.tensorrt):
        if not supports_export_stage(export_spec, "tensorrt"): raise ...
        build_tensorrt_engine(module_onnx_path, deploy_cfg, output_dir / f"{module_name}.engine")   # :246
```

整個流程都被包裝起來，因此只要發生任何例外（exception），MLflow 的 deploy run 就會被標記為
FAILED；成功時則標記為 FINISHED。輸出結果會存放在 `{output_dir}/{module_name}.onnx` 與
`.engine` — 當啟用記錄（logging）功能時，會存放在 MLflow run 的 `exports/` 目錄中。

---

## 4. 取得範例輸入（`resolve_export_specs`）

匯出需要具體的範例 tensor。框架是從 **predict dataloader** 取得這些範例，並經過與訓練時*相同*
的裝置轉移（device transfer）與前處理（preprocessing）：

```python
def get_predict_batch(datamodule, model, device):        # utils/deploy.py:127
    datamodule.setup("predict")
    batch = next(iter(datamodule.predict_dataloader()))
    batch = move_to_device(batch, device)
    return model.on_after_batch_transfer(batch, dataloader_idx=0)   # runs DataPreprocessing (voxelize, etc.)

def resolve_export_specs(datamodule, model, device):     # :157
    batch = get_predict_batch(datamodule, model, device)
    return model.build_export_specs(batch)               # ← the model decides what to export
```

因此，這個範例 batch 是*真實、經過前處理*的資料 — 這也是為何在 `build_export_specs` 執行時，
像 `voxels`、`voxel_coords` 這類的 voxel key 已經存在。

---

## 5. 匯出合約：`ExportSpec` 與 `build_export_specs`

```python
@dataclass(frozen=True)                                  # utils/deploy.py:40
class ExportSpec:
    module: torch.nn.Module                 # the exact submodule/wrapper to trace
    args: tuple[Any, ...]                   # example positional inputs (from the predict batch)
    input_param_names: list[str]
    output_names: list[str] | None = None
    dynamic_axes: dict[str, dict[int, str]] | None = None   # legacy path only (dynamo=False)
    supported_stages: frozenset[str] = frozenset({"onnx", "tensorrt"})
```

`BaseModel` 提供了預設實作（`models/base.py:358`）：

```python
def build_export_spec(self, batch):                      # default: one end-to-end module
    raw = infer_export_spec(self, batch)                 # derive args from forward signature
    return ExportSpec(module=_PredictionExportWrapper(self), args=raw.args,
                      input_param_names=raw.input_param_names,
                      output_names=self.get_export_output_names(), supported_stages=raw.supported_stages)

def build_export_specs(self, batch):                     # what deploy actually calls
    return {"end_to_end": self.build_export_spec(batch)}
```

`_PredictionExportWrapper`（`base.py:420`）讓 ONNX 圖輸出**任務層級的預測結果**（它會依序執行
`forward` → `predict_outputs` → `prepare_export_outputs`），因此一個簡單的模型完全不需要撰寫
任何匯出程式碼：預設行為會將整個模型追蹤（trace）為單一個 `end_to_end` 模組。

### 拆分模組匯出：CenterPoint (`models/detection3d/centerpoint.py:163`)

CenterPoint 無法作為單一個 ONNX 圖存在 — 因為 pillar encoder 與 BEV backbone 之間的 scatter
步驟是一個執行期（runtime）操作（無法作為單一個 op 追蹤）。因此它覆寫了 `build_export_specs`，
輸出**兩個**模組，以符合 Autoware CenterPoint 既有（historical）的 ABI：

```python
def build_export_spec(self, batch):                      # single-module export is rejected
    raise RuntimeError("CenterPoint deployment uses split modules; call build_export_specs().")

def build_export_specs(self, batch):                     # :163
    # run the front of the net once to get realistic example inputs for each module
    input_features   = self.pts_voxel_encoder.decorate(batch["voxels"], batch["num_points"], batch["voxel_coords"])
    pillar_features  = self.pts_voxel_encoder.encode_decorated(input_features).squeeze(1)
    spatial_features = self.pts_middle_encoder(pillar_features, batch["voxel_coords"], batch_size=...)
    return {
        "pts_voxel_encoder_centerpoint": ExportSpec(          # PFN MLP:  input_features → pillar_features
            module=_CenterPointVoxelEncoderExportWrapper(self.pts_voxel_encoder),
            args=(input_features,), input_param_names=["input_features"], output_names=["pillar_features"]),
        "pts_backbone_neck_head_centerpoint": ExportSpec(     # backbone+neck+head: spatial_features → dense maps
            module=head_wrapper, args=(spatial_features,),
            input_param_names=["spatial_features"], output_names=head_wrapper.output_names),
    }
```

**scatter**（`PointPillarsScatter`）是在車輛推論（inference）時，作為兩個匯出引擎（engine）
*之間*的執行期前處理來執行的 — 它並不屬於任何一個 ONNX 圖的一部分。當某個模型中間存在無法匯出
的步驟時，這就是可以套用的模式。

### `supported_stages` — 模型可以選擇不支援（opt out）TensorRT

`ExportSpec.supported_stages`（預設為 `{"onnx", "tensorrt"}`）讓模型可以宣告自己無法支援某個
階段（stage）。**PTv3** 將 `EXPORT_SUPPORTED_STAGES` 設為 `frozenset({"onnx"})`，因為它需要
目標執行環境（runtime）可能沒有的 sparse conv 外掛（plugin） — 因此
`autoware-ml deploy ... ptv3` 只會產生 ONNX，若要求產生 TensorRT，則會拋出明確的錯誤。
`should_export_stage`（config 中的 `enabled`）× `supports_export_stage`（模型的能力）共同決定
每個階段是否放行。

---

## 6. 權重合併與完整覆蓋率（`apply_matching_weights`）

`--weights` 可以被多次傳入。`apply_matching_weights`（`utils/checkpoints.py`）會以
`strict=False` 的方式逐一載入，只套用那些存在於匯出模型中且形狀（shape）相符的 key；後面的
checkpoint 會覆蓋前面的。當 `enforce_full_coverage=True`（deploy 時）啟用時，在載入完所有
checkpoint 之後，它會驗證**每一個**模型參數都已被覆蓋 — 否則會立即失敗並列出缺少的 key，因此
你絕對不會出貨一個帶有未訓練層（untrained layers）的 engine。

這使得**多 head 匯出**成為可能：例如 PTv3 detection 會將一個預訓練的 segmentation backbone
checkpoint 與一個 detection-head checkpoint 合併：

```bash
autoware-ml deploy --config-name detection3d/ptv3/voxel012_122m_t4dataset_j6gen2 \
    --weights .../segmentation3d/ptv3/.../best.ckpt \      # backbone
    --weights .../detection3d/ptv3/.../best.ckpt           # detection head
```

---

## 7. Config：`deploy` 區段

預設值（`configs/defaults/modules/deploy.yaml`）：

```yaml
deploy:
  onnx:      { enabled: true, dynamo: true, opset_version: 21, modify_graph: null }
  tensorrt:  { enabled: true, workspace_size: 4294967296 }   # 4 GiB
```

每個模組的 ONNX 覆寫設定存放於 `deploy.onnx.modules.<module_name>` 之下，並由
`merge_module_onnx_cfg`（`utils/deploy.py:176`）與共用設定合併（模組層級優先，`modules` 這個
key 會被移除）。舉例來說，CenterPoint 的基礎 config 設定了 `dynamo: false`、
`opset_version: 17`、`tensorrt.enabled: false`，以及各模組的 `dynamic_axes`。詳細內容請見
[onnx_tensorRT.md](onnx_tensorRT.md)。

> 注意：`trainer.precision`（例如 `bf16-mixed`）是**訓練（training）**設定 — 它*不會*控制
> TensorRT engine 的精度（precision）。詳見 [onnx_tensorRT.md](onnx_tensorRT.md)。

---

## 常見除錯情境

| 症狀 | 原因 | 修正方式 |
| ------- | ----- | --- |
| `--weights must be specified` | deploy 需要明確指定的權重 | 傳入 `--weights <ckpt>` |
| "missing keys" / 未完全覆蓋 | `enforce_full_coverage` 檢查失敗 | 新增/替換 `--weights`，直到每個參數都被覆蓋為止 |
| `Config must define a 'deploy' section` | 任務 config 缺少 `deploy` | 這個區段是從 `default_runtime` 繼承而來的；請確認沒有不小心移除它 |
| `Module 'X' does not support ONNX/TensorRT` | 模型設定了 `supported_stages`（例如 PTv3 僅支援 ONNX） | 停用該階段（`deploy.tensorrt.enabled=false`） |
| 輸出必須保留在 artifact 目錄內 | 自訂的 `output_dir` 位在 MLflow run 之外 | 不設定 `output_dir`，或停用 logger |
| `CUDA is not available` | deploy 需要 GPU | 在具備 CUDA 的機器上執行 |
| CenterPoint 單一模組匯出錯誤 | 呼叫了 `build_export_spec` | CenterPoint 使用（拆分後的）`build_export_specs` |

---

## 常見修改情境

| 我想要… | 這麼做 |
| ---------- | ------- |
| 只匯出 ONNX（跳過 TRT） | `deploy.tensorrt.enabled=false` |
| 命名/重新導向輸出 | `output_name=...`、`output_dir=...`（若有記錄則需位於 run 的 artifact 目錄內） |
| 為新模型新增匯出功能 | 依賴預設的 `end_to_end` 路徑；只有在需要拆分模組或有無法追蹤的中間步驟時，才覆寫 `build_export_specs` |
| 將一個模型拆分為多個 engine | 覆寫 `build_export_specs`，為每個子圖（sub-graph）回傳一個 `ExportSpec`（參見 CenterPoint） |
| 將模型標記為僅支援 ONNX | 在其 spec 中將 `supported_stages` 設為 `frozenset({"onnx"})` |
| 合併多個 checkpoint | 重複傳入 `--weights`（多 head） |

---

**下一步：** [onnx_tensorRT.md](onnx_tensorRT.md) — ONNX 匯出與 TensorRT 建構（build）的內部機制。
