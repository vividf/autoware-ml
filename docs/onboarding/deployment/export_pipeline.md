# Export Pipeline

> **What this covers:** how a trained checkpoint becomes deployable artifacts — the
> `autoware-ml deploy` flow, the `ExportSpec` / `build_export_specs` contract, multi-module and
> multi-head exports, and weight merging. The ONNX/TensorRT mechanics are in
> [onnx_tensorRT.md](onnx_tensorRT.md).
>
> Prerequisites: [../model/model_architecture.md](../model/model_architecture.md),
> [../code_walkthrough/entry_point.md](../code_walkthrough/entry_point.md).

---

## 1. Why deployment is a first-class, model-owned step

Autoware runs perception on-vehicle via **TensorRT**. If export lived in a separate script, it
would drift from the trained model. So autoware-ml makes export part of the model's contract:
the model declares *what to export* via `build_export_specs()`, and the framework performs the
ONNX → TensorRT conversion. The **same config** drives train/test/deploy, so the exported
graph is guaranteed to match the trained architecture.

```text
checkpoint(.ckpt) ──load──▶ model (eval) ──predict batch──▶ build_export_specs()
                                                                 │  {module_name: ExportSpec}
                                                                 ▼
                                        per module:  torch.onnx.export ──▶ .onnx
                                                     (optional graph modify)
                                                     TensorRT build ──▶ .engine
```

---

## 2. The command

```bash
autoware-ml deploy \
    --config-name detection3d/centerpoint/voxel020_second_secfpn_51m_nuscenes \
    --weights mlruns/.../checkpoints/best.ckpt
```

- `--config-name` — the **same** task config used for training.
- `--weights` — one or more `.ckpt` paths whose parameters are merged into the export model
  (repeatable; later ones overwrite earlier on overlapping keys). This is the *only* way to
  supply parameters.
- Options: `output_name=<name>`, `output_dir=<path>`, and stage toggles like
  `deploy.tensorrt.enabled=false`.

Dispatch is identical to train/test: `cli.py deploy` → `run_hydra_entrypoint` (stage
`deploy`) → `scripts/deploy.py:main` (`@hydra.main`). See
[../code_walkthrough/entry_point.md](../code_walkthrough/entry_point.md).

---

## 3. `scripts/deploy.py:main` step by step

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

The whole thing is wrapped so the MLflow deploy run is marked FAILED on any exception and
FINISHED on success. Outputs land at `{output_dir}/{module_name}.onnx` and `.engine` — inside
the MLflow run's `exports/` dir when logging is enabled.

---

## 4. Getting the example inputs (`resolve_export_specs`)

Export needs concrete example tensors. The framework gets them from the **predict
dataloader**, run through the *same* device transfer + preprocessing as training:

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

So the example batch is *real, preprocessed* data — which is why voxel keys like `voxels`,
`voxel_coords` exist by the time `build_export_specs` runs.

---

## 5. The export contract: `ExportSpec` and `build_export_specs`

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

`BaseModel` provides the default (`models/base.py:358`):

```python
def build_export_spec(self, batch):                      # default: one end-to-end module
    raw = infer_export_spec(self, batch)                 # derive args from forward signature
    return ExportSpec(module=_PredictionExportWrapper(self), args=raw.args,
                      input_param_names=raw.input_param_names,
                      output_names=self.get_export_output_names(), supported_stages=raw.supported_stages)

def build_export_specs(self, batch):                     # what deploy actually calls
    return {"end_to_end": self.build_export_spec(batch)}
```

`_PredictionExportWrapper` (`base.py:420`) makes the ONNX graph emit **task-level predictions**
(it runs `forward` → `predict_outputs` → `prepare_export_outputs`), so a simple model needs no
export code at all: the default traces the whole model as one `end_to_end` module.

### Split-module export: CenterPoint (`models/detection3d/centerpoint.py:163`)

CenterPoint can't be one ONNX graph — the scatter step between the pillar encoder and the BEV
backbone is a runtime (non-traceable-as-one-op) operation. So it overrides `build_export_specs`
to emit **two** modules, matching the historical Autoware CenterPoint ABI:

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

The **scatter** (`PointPillarsScatter`) runs as runtime preprocessing *between* the two
exported engines at inference on the vehicle — it is not part of either ONNX graph. This is the
pattern to copy when a model has a non-exportable step in the middle.

### `supported_stages` — models can opt out of TensorRT

`ExportSpec.supported_stages` (default `{"onnx", "tensorrt"}`) lets a model declare it can't do
a stage. **PTv3** sets `EXPORT_SUPPORTED_STAGES = frozenset({"onnx"})` because it needs sparse
conv plugins the target runtime may not have — so `autoware-ml deploy ... ptv3` produces ONNX
only, and asking for TensorRT raises a clear error. `should_export_stage` (config
`enabled`) × `supports_export_stage` (model capability) gate each stage.

---

## 6. Weight merging and full coverage (`apply_matching_weights`)

`--weights` can be passed multiple times. `apply_matching_weights`
(`utils/checkpoints.py`) loads each with `strict=False`, applying only keys that exist on the
export model and match shapes; later checkpoints overwrite earlier ones. With
`enforce_full_coverage=True` (deploy), after all checkpoints it verifies **every** model
parameter was covered — otherwise it fails up front listing the missing keys, so you never ship
an engine with untrained layers.

This enables **multi-head exports**: e.g. PTv3 detection merges a pretrained segmentation
backbone checkpoint with a detection-head checkpoint:

```bash
autoware-ml deploy --config-name detection3d/ptv3/voxel012_122m_t4dataset_j6gen2 \
    --weights .../segmentation3d/ptv3/.../best.ckpt \      # backbone
    --weights .../detection3d/ptv3/.../best.ckpt           # detection head
```

---

## 7. Config: the `deploy` section

Default (`configs/defaults/modules/deploy.yaml`):

```yaml
deploy:
  onnx:      { enabled: true, dynamo: true, opset_version: 21, modify_graph: null }
  tensorrt:  { enabled: true, workspace_size: 4294967296 }   # 4 GiB
```

Per-module ONNX overrides live under `deploy.onnx.modules.<module_name>` and are merged over
the shared settings by `merge_module_onnx_cfg` (`utils/deploy.py:176`; module wins, the
`modules` key is dropped). CenterPoint's base config, for instance, sets `dynamo: false`,
`opset_version: 17`, `tensorrt.enabled: false`, and per-module `dynamic_axes`. Details in
[onnx_tensorRT.md](onnx_tensorRT.md).

> Note: `trainer.precision` (e.g. `bf16-mixed`) is a **training** setting — it does *not*
> control the TensorRT engine precision. See [onnx_tensorRT.md](onnx_tensorRT.md#4-precision).

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `--weights must be specified` | deploy needs explicit weights | pass `--weights <ckpt>` |
| "missing keys" / not fully covered | `enforce_full_coverage` failed | add/replace `--weights` until every param is covered |
| `Config must define a 'deploy' section` | task config lacks `deploy` | it's inherited from `default_runtime`; check you didn't drop it |
| `Module 'X' does not support ONNX/TensorRT` | model set `supported_stages` (e.g. PTv3 = ONNX only) | disable that stage (`deploy.tensorrt.enabled=false`) |
| outputs must stay inside artifact dir | custom `output_dir` outside MLflow run | leave `output_dir` unset, or disable the logger |
| `CUDA is not available` | deploy needs a GPU | run on a CUDA machine |
| CenterPoint single-module export error | called `build_export_spec` | CenterPoint uses `build_export_specs` (split) |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Export ONNX only (skip TRT) | `deploy.tensorrt.enabled=false` |
| Name/redirect outputs | `output_name=...`, `output_dir=...` (inside the run artifact dir if logging) |
| Add export to a new model | rely on the default `end_to_end` path; override `build_export_specs` only if you need split modules or non-traceable middle steps |
| Split a model into multiple engines | override `build_export_specs` returning one `ExportSpec` per sub-graph (see CenterPoint) |
| Mark a model ONNX-only | set `supported_stages = frozenset({"onnx"})` on its specs |
| Merge multiple checkpoints | pass repeated `--weights` (multi-head) |

---

**Next:** [onnx_tensorRT.md](onnx_tensorRT.md) — the ONNX export and TensorRT build internals.
