# ONNX & TensorRT

> **What this covers:** the export internals — `torch.onnx.export` (dynamo vs legacy), dynamic
> shapes, ONNX graph modifiers, the TensorRT engine build, and the custom ops that need
> matching TensorRT plugins. This is the deepest, most vehicle-facing layer.
> Prerequisite: [export_pipeline.md](export_pipeline.md).

All code here lives in `autoware_ml/utils/deploy.py` (export + TRT), `autoware_ml/utils/onnx_modifiers.py`
(graph edits), and `autoware_ml/ops/` (custom ops).

---

## 1. ONNX export (`export_to_onnx`, `utils/deploy.py:328`)

```python
def export_to_onnx(model, input_sample, onnx_cfg, input_param_names,
                   output_names_override, dynamic_axes_override, output_path):
    dynamo = onnx_cfg.get("dynamo", True)
    dynamic_shapes = build_dynamic_shapes(onnx_cfg, input_param_names) if dynamo else None
    dynamic_shapes = normalize_dynamic_shapes_for_model(model, dynamic_shapes) if dynamo else None
    dynamic_axes = (dynamic_axes_override or build_dynamic_axes(onnx_cfg)) if not dynamo else None
    input_names  = list(onnx_cfg.get("input_names", input_param_names))
    output_names = list(output_names_override or onnx_cfg.get("output_names", ["output"]))

    register_scatter_reduce_onnx_symbolic(opset_version=int(onnx_cfg.opset_version))   # :360

    export_kwargs = {"model": model, "args": input_sample, "f": str(output_path),
                     "input_names": input_names, "output_names": output_names,
                     "opset_version": onnx_cfg.opset_version, "dynamo": dynamo,
                     "do_constant_folding": onnx_cfg.get("do_constant_folding", True)}
    export_kwargs["dynamic_shapes" if dynamo else "dynamic_axes"] = dynamic_shapes if dynamo else dynamic_axes
    torch.onnx.export(**export_kwargs)                                                  # :377
    # if a `.onnx.data` shard was written, merge it back into a single file
```

It's a single `torch.onnx.export` call, but with **two modes** selected by `deploy.onnx.dynamo`:

| Mode | `dynamo` | Dynamic dims via | When |
| ---- | -------- | ---------------- | ---- |
| **Dynamo** (default) | `true` | `dynamic_shapes` (`torch.export.Dim`) | modern `torch.export`-based export |
| **Legacy** | `false` | `dynamic_axes` (name→dim map) | models relying on legacy ONNX symbolic functions (e.g. CenterPoint, TransFusion, FRNet) |

Before exporting it registers a shared symbolic (`register_scatter_reduce_onnx_symbolic`) so
`aten::scatter_reduce` maps to standard ONNX `ScatterElements`. After exporting, if
`torch.onnx` wrote external-data shards (`.onnx.data`), `merge_onnx_external_data` folds them
back into one self-contained `.onnx`.

### Dynamic shapes (dynamo, `build_dynamic_shapes:218`)

Config maps input name → {dim index → symbolic}. Two forms:

```yaml
deploy:
  onnx:
    dynamic_shapes:
      input_tensor: { 2: height, 3: width }        # shorthand: dim 2 = "height", dim 3 = "width"
      points:
        0: { name: num_points, min: 2 }             # explicit, with bounds
```

Each becomes a `torch.export.Dim(name, min=?, max=?)`. `normalize_dynamic_shapes_for_model`
wraps the structure one level deeper for `forward(*args)` wrappers, because `torch.export`
requires `dynamic_shapes` to mirror the positional-arg pytree. Unknown parameter names raise.

### Dynamic axes (legacy, `build_dynamic_axes:285`)

Same config, but produces the legacy `{tensor_name: {dim: name}}` map. CenterPoint's config:

```yaml
deploy:
  onnx:
    dynamo: false
    opset_version: 17
    modules:
      pts_voxel_encoder_centerpoint:
        input_names: [input_features]
        output_names: [pillar_features]
        dynamic_axes:
          input_features: { 0: num_voxels, 1: num_max_points }
```

---

## 2. Optional ONNX graph modifiers (`utils/onnx_modifiers.py`)

Some TensorRT limitations are easier to fix by rewriting the ONNX graph than by changing the
model. After export, if `deploy.onnx.modify_graph` is set, `modify_onnx_graph` instantiates the
modifier (via Hydra `_target_`) and applies it:

| Modifier | What it rewrites | Why |
| -------- | ---------------- | --- |
| `TopKConstantKModifier` | a TopK node's `K` input → a compile-time constant | TensorRT rejects argsort-derived dynamic `K` |
| `AttentionScaleToDivModifier` | `Mul(q, scale)` → `Div(q, 1/scale)` | TRT-friendlier attention scaling |
| `TransHeadTensorRTModifier` | composes both above | TransFusion-style decoder heads (used by PTv3 detection) |

Config example (PTv3 detection `det3d_head` module):

```yaml
deploy:
  onnx:
    modules:
      det3d_head:
        modify_graph:
          _target_: autoware_ml.utils.onnx_modifiers.TransHeadTensorRTModifier
          k: ${num_proposals}
          topk_node_name_substring: /bbox_head/TopK
          attention_node_name_substring: /bbox_head/decoder
```

These use raw `onnx` protobuf + `numpy_helper` (no `onnx_graphsurgeon`/`onnxscript` dependency).
The preferred path is still to keep export logic in the model (`build_export_specs`); graph
modifiers are the escape hatch for TRT-only quirks.

---

## 3. TensorRT engine build (`build_tensorrt_engine`, `utils/deploy.py:484`)

```python
def build_tensorrt_engine(onnx_path, deploy_cfg, output_path):
    tensorrt_cfg = deploy_cfg.tensorrt
    builder, network, parser, config = create_tensorrt_builder_config(tensorrt_cfg)   # :492
    parse_onnx_file(parser, onnx_path)                                                 # :493
    profile = create_optimization_profile(builder, tensorrt_cfg)                       # :495
    if profile is not None:
        config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, config)              # :500
    output_path.write_bytes(serialized_engine)
```

Builder setup (`create_tensorrt_builder_config:427`):

```python
import tensorrt as trt
trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, "")                        # loads TRT plugins, incl. custom autoware:: ops
builder = trt.Builder(trt_logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))   # :434
parser  = trt.OnnxParser(network, trt_logger)
config  = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, tensorrt_cfg.get("workspace_size", 1 << 30))
```

Two design choices to note:

- **`init_libnvinfer_plugins`** loads TensorRT's plugin registry — this is how the vehicle-side
  custom `autoware::` operator plugins (for the ops in §5) become available to the parser.
- **`STRONGLY_TYPED` network** — precision is taken from the **ONNX tensor dtypes**, not set via
  builder flags. See §4.

### Optimization profile (`create_optimization_profile:457`)

For dynamic inputs you give TensorRT min/opt/max shapes so it can pre-plan kernels:

```yaml
deploy:
  tensorrt:
    workspace_size: 8589934592     # 8 GiB
    input_shapes:
      input:
        min_shape: [1, 3, 224, 224]
        opt_shape: [1, 3, 256, 256]   # TRT optimizes most aggressively for opt_shape
        max_shape: [1, 3, 512, 512]
```

All three (`min`/`opt`/`max`) are required per input or it raises. Set `opt_shape` to your
typical on-vehicle resolution.

---

## 4. Precision

Because the network is **`STRONGLY_TYPED`**, FP16/FP32 is baked in from the ONNX graph's tensor
dtypes — there are **no** `config.set_flag(FP16/INT8)` calls and no INT8 calibrator in this
builder. Consequences:

- To export a half-precision engine, the ONNX must carry fp16 tensors (produced from a half /
  autocast export), not a builder flag.
- `trainer.precision` (e.g. `bf16-mixed`) is **training** precision and has no effect here.
- There is no INT8 PTQ path in this TRT builder.

---

## 5. Custom ops and the ONNX ↔ TensorRT bridge (`autoware_ml/ops/`)

Some operations have no standard ONNX/TensorRT equivalent, so the repo ships custom ops. Each
defines a `torch.autograd.Function` whose `forward` runs the eager kernel and whose `symbolic`
emits either a **standard** ONNX op or an **`autoware::`-namespaced** custom op that a matching
TensorRT plugin implements on the vehicle.

| Package | Native? | Export bridge |
| ------- | ------- | ------------- |
| `ops/bev_pool/` | **Yes — CUDA** (`bev_pool_ext`, built from `src/*.cu,*.cpp`) | `QuickCumsumCuda.symbolic` → `g.op("autoware::QuickCumsumCuda", ...)` (BEVFusion camera→BEV pooling) |
| `ops/indexing/` | No (wraps `torch.unique`/`torch.sort`) | `_Unique.symbolic` → `autoware::CustomUnique`; `_Argsort.symbolic` → `autoware::Argsort` (only under `torch.onnx.is_in_onnx_export()`) |
| `ops/segment/` | No | `_SegmentCSR.symbolic` → `autoware::SegmentCSR`; and `scatter_reduce` → standard ONNX `ScatterElements` (registered in `export_to_onnx`) |
| `ops/spconv/` | No (depends on external `spconv`) | autograd `Function`s emitting `autoware::` sparse-conv ops |
| `ops/voxelization/` | No custom ONNX op | pure-PyTorch voxelization (used by preprocessing, not exported as a custom op) |

The only compiled native op is `bev_pool` (`ops/build.py` → `bev_pool_ext`, arch gencodes
sm_80/86/89/90/120). The rest are Python wrappers with ONNX symbolics.

**The chain that makes this work end to end:** model uses the op → the op's `symbolic` writes an
`autoware::` node into the ONNX graph → at engine build, `init_libnvinfer_plugins` loads the
matching plugin → the vehicle's TensorRT runtime executes it. This is why a model like PTv3
(sparse conv) is marked ONNX-only when the target runtime lacks those plugins — see
[export_pipeline.md](export_pipeline.md#5-the-export-contract-exportspec-and-build_export_specs).

---

## 6. End-to-end: checkpoint → ONNX → engine

```text
scripts/deploy.py
  load ckpt → model.eval()
  resolve_export_specs → {module: ExportSpec(module, args, names, ...)}
  per module:
    merge_module_onnx_cfg          shared deploy.onnx.* + deploy.onnx.modules.<name>.*
    export_to_onnx                 register scatter symbolic → torch.onnx.export(dynamo|legacy) → {module}.onnx
    (modify_onnx_graph)            optional TRT-oriented graph rewrite
    build_tensorrt_engine          init plugins → STRONGLY_TYPED network → parse ONNX
                                   → optimization profile (min/opt/max) → build_serialized_network → {module}.engine
```

---

## Common debugging cases

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| ONNX export fails on a custom op | op lacks a symbolic, or exported without `is_in_onnx_export` guard | use an op from `ops/` with a symbolic, or add one |
| `torch.export` dynamic_shapes structure error | wrapper `forward(*args)` needs deeper nesting | handled by `normalize_dynamic_shapes_for_model`; check the wrapper's signature |
| Unknown dynamic-shape param | name ∉ export input names | match `input_param_names` |
| TRT parse error on TopK / argsort | dynamic `K`, unsupported pattern | add `TopKConstantKModifier` / `TransHeadTensorRTModifier` via `modify_graph` |
| TRT "no plugin for autoware::X" | plugin not loaded / not built for the target | ensure `init_libnvinfer_plugins` + the vehicle runtime ships the plugin; else export ONNX-only |
| Engine bigger/slower than expected | workspace too small, opt_shape wrong | raise `workspace_size`; set `opt_shape` to real resolution |
| fp16 not taking effect | STRONGLY_TYPED reads dtypes from ONNX | export the ONNX in fp16; don't rely on a builder flag |
| `CUDA not available` | TRT needs a GPU | run on CUDA hardware |

---

## Common modification scenarios

| I want to… | Do this |
| ---------- | ------- |
| Switch dynamo ↔ legacy export | `deploy.onnx.dynamo=true|false` |
| Add/adjust dynamic dims | `deploy.onnx.dynamic_shapes` (dynamo) or `dynamic_axes` (legacy) |
| Change opset | `deploy.onnx.opset_version` |
| Add a TRT-oriented graph fix | set `deploy.onnx.modules.<m>.modify_graph._target_` to a modifier |
| Tune the engine profile | `deploy.tensorrt.input_shapes.<in>.{min,opt,max}_shape`, `workspace_size` |
| Add a custom op | add a `torch.autograd.Function` under `ops/` with a `symbolic` emitting `autoware::<Op>` (+ a matching TRT plugin) |

---

**End of the onboarding guide.** You've now traced the framework from the CLI to a TensorRT
engine. Return to the [README](../README.md) for the reading map, and remember the guiding
rule: **the source is the source of truth — verify against the code, and fix these docs when
they drift.**
