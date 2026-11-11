---
icon: lucide/package
---

# Deployment

Autoware-ML exports trained models for production use. The pipeline converts
PyTorch checkpoints to optimized inference formats, one deployable module at a
time, and stamps every exported module with its identity and provenance.

## Deployment Pipeline

```text
Checkpoint (.ckpt) -> ONNX (.onnx) -> graph modifiers -> precision -> metadata stamp -> TensorRT (.engine)
```

Each stage runs per module. The stamp is applied to the final ONNX, so the
shipped file always carries its own metadata.

## Basic Usage

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt
```

`--weights` accepts one or more checkpoint paths and is the only way to supply
parameters to the export model. For a single-task export, pass one
checkpoint. For a multi-head export, pass one `--weights` per source
checkpoint (see [Multi-head exports](#multi-head-exports)).

!!! warning
    Pass `--release vMAJOR.MINOR.PATCH` if the model is version controlled.
    Without it the artifacts are stamped `unversioned`, which is fine for
    quick iteration but never for a release build.

This generates one ONNX (`<module>.onnx`) and one TensorRT engine
(`<module>.engine`) per deployable module, when both stages are enabled and
supported by the model. The deploy command also creates a dedicated MLflow run
linked to the source training run and logs exported artifacts there.

You can disable either stage during iteration:

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

Deploy writes its outputs into the deploy MLflow run's artifact directory, under
`mlruns/<experiment_group_name>/<experiment_name>/<deploy_run_id>/artifacts/exports/`.
The same run records the resolved config and CLI overrides under `config/`, and a
`run_metadata.json` listing the source checkpoints, so an export can always be traced
back to the training run it came from. A deploy that fails after its run was created
leaves the run marked `FAILED`.

Without MLflow logging (`~logger`), outputs land in the config's `experiment_run_dir`,
`<working_dir>/tasks/<experiment_group_name>/<experiment_name>/<experiment_uid>/`.
The output location is not configurable on its own.

## Export Modules

A model exports one or more deployable modules. Models without a dedicated
export split produce a single module named `end_to_end`.

Artifact files are named after the module. Every exported module has an entry
under `deploy.onnx.modules`: `{}` exports it with the shared `deploy.onnx`
settings, and any key in the entry overrides them. A module without an entry
fails the deploy before the deploy run is created, so a module added to a model
cannot silently export without its `metainfo` or `modify_graph`, and a
misspelled entry shows up as the missing one. Entries of modules the model does
not export are left alone, so a model whose modules depend on its config, as
BEVFusion's on `camera_network`, lists the entries of every variant in one base
config.

## Multi-head exports

Multi-head export models can expose multiple deployable modules from one
configured model. `--weights` can merge checkpoints from independently
trained parts of that model into a single export:

```bash
autoware-ml deploy \
    --config-name detection3d/ptv3/nuscenes/voxel005_b16_30e_nuscenes_51m \
    --weights mlruns/ptv3_seg3d/NuScenesDataset-base-v1.0.0/nuscenes/voxel005_b16_50e_nuscenes_51m/<run_id>/artifacts/checkpoints/best.ckpt \
    --weights mlruns/ptv3/NuScenesDataset-base-v1.0.0/nuscenes/voxel005_b16_30e_nuscenes_51m/<run_id>/artifacts/checkpoints/best.ckpt
```

Checkpoints are applied in the order they appear on the command line, and
later checkpoints overwrite any keys already set by earlier ones. Each
checkpoint only contributes the state-dict keys that exist on the export
model and match its tensor shapes. Keys missing from the model are skipped.
Keys with a matching name but mismatching shape raise an error immediately.

**Full coverage is enforced.** After all checkpoints are loaded, deploy
verifies that every parameter in the export model has been covered by at
least one of the supplied `--weights`. If any parameter is left
uninitialized, the command fails up front with the list of missing keys
instead of producing an ONNX or engine that contains untrained layers. Add
or replace `--weights` entries until every key is covered.

## ONNX Metadata

Every exported module carries its identity and provenance inside the ONNX
file: producer, git commit of the export, release (encoded into
`model_version` as `major * 10000 + minor * 100 + patch`), the config name,
export date, and the linked deploy run.

Per-module inference parameters come from the deploy config's `metainfo`
block. The pipeline serializes whatever the config declares without
interpreting it:

```yaml
deploy:
  onnx:
    modules:
      <module>:
        metainfo:
          class_names: ${taxonomy.detection3d.class_names}
```

Each `metainfo` value is stamped as one compact JSON document (scalars,
strings, lists and mappings, nested as declared), while the provenance
properties above are plain strings, so a consumer reads a parameter with any
JSON parser. Non-finite floats, values JSON cannot represent, and keys that
collide with the automatically stamped properties fail at export time.

## Configuration

### ONNX Settings

Shared settings live at `deploy.onnx`. Each exported module lists an entry
under `deploy.onnx.modules.<module>`, `{}` to keep the shared settings or the
keys that override them:

```yaml
deploy:
  onnx:
    enabled: true
    dynamo: true
    opset_version: 21
    precision: fp32
    modules:
      end_to_end:
        precision: fp16
```

**Tensor names**: the export spec names the graph inputs and outputs, either
the names a model's [deployment graphs](#deployment-graphs-verification-and-evaluation) declare or
the names a hand-written spec lists; the export log prints them. A spec names every one of
its inputs and at least one output, or it is refused when built. `deploy.onnx`
carries no `input_names` or `output_names`; an entry fails the deploy before its
MLflow run is created or the model loads. The keys of
`deploy.tensorrt.input_shapes` are these names. A model that lets a config
choose its names exposes them as a model parameter, as PTv3's
`export_output_names`.

**Dynamic axes**: which axes vary between runs is a property of the model, so
the export spec declares them once (`ExportSpec.dynamic_axes`, keyed by its
input and output names) and both exporters derive their arguments from that
declaration. `deploy.onnx` carries no `dynamic_axes` or `dynamic_shapes`; an
entry fails the deploy like a configured name does. The spec refuses an axis of
a tensor it does not name. The deployment settings that do vary, the TensorRT
shape ranges, stay in `deploy.tensorrt.input_shapes`.

**precision**: `fp32` exports the model unchanged. `fp16` halves the weights and
internal tensors but keeps graph inputs and outputs fp32, so consumers keep their
fp32 buffers either way. Calibrated precisions such as `int8` are out of scope here.

An fp16 export only works if the inference engine honors its dtypes instead of
reassigning them. In Autoware, that means `trt_precision: strongly-typed` on the
node.

**dynamo**: `true` exports through `torch.export`, which gets one `Dim` per
axis name the spec declares over the inputs. Set `dynamo: false` for models
that export through the TorchScript exporter and its ONNX symbolic functions;
it takes the spec's `dynamic_axes` directly.

### TensorRT Settings

```yaml
deploy:
  tensorrt:
    enabled: true
    workspace_size: 8589934592  # 8 GiB
    input_shapes:
      input:
        min_shape: [1, 3, 224, 224]
        opt_shape: [1, 3, 256, 256]   # Optimized for this
        max_shape: [1, 3, 512, 512]
```

!!! tip
    TensorRT optimizes most aggressively for `opt_shape`. Set this to your typical inference resolution.

Engines are always built strongly typed: the builder uses exactly the precisions
the ONNX carries and never picks its own. Choose the engine's numerics with
`deploy.onnx.precision`.

## Plugin graphs (sparse convolutions)

Graphs with sparse convolutions — BEVFusion's lidar encoder, PTv3's cpe convs — export
them as ONNX nodes in the `autoware` domain (`autoware::GetIndicePairsImplicitGemm`,
`autoware::ImplicitGemm`), the operators `autoware_universe`'s `autoware_tensorrt_plugins`
implements. TensorRT loads the library before parsing the ONNX:

```yaml
deploy:
  tensorrt:
    plugin_libraries: [/opt/plugins/libautoware_tensorrt_plugins.so]
```

A plugin node with a bias or quantization scales has seven inputs: the five sparse
tensors, then a per-channel `weight_scale` and the `bias`, both FP32 and zero-length when
absent (`autoware_ml/ops/spconv/contract.py`). spconv's plugin and t4sparse read this
layout alike, so one exported graph builds against either library.

The Docker image builds the plugin (`docker/tensorrt_plugins/`, see its README for a
rebuild against another `autoware_universe` branch). ONNX Runtime cannot run plugin
nodes, so such graphs declare `torch_fallback_backends=(Backend.ONNX,)`: the `onnx`
backend runs them in PyTorch (starred in the evaluation table) while `tensorrt` runs the
engine. The exported graph is ready for the runtime as it is.

The FP16 cast of a plugin or Q/DQ graph is island-aware
(`autoware_ml/deployment/onnx/precision.py`): plugin nodes, integer index paths and
`Range` stay fp32, Linear Q/DQ regions keep fp32 mini-islands, and the graph I/O stays fp32.
Plain graphs go through the upstream `onnxconverter_common` cast unchanged.

## Model-Owned Export Wrappers

Export logic lives inside the model. A new model declares its deployment with two
methods: `deploy_graphs()` returns its `GraphStage`s (one exported module each,
named after the graph, with the tensor names it takes and returns), and
`deployed_forward(batch, graphs)` runs the deployed inference, the glue in plain
code and each graph called as `graphs[name](input_name=tensor)`.
`build_export_specs()` runs that function once on the example batch in eval
mode and returns one export spec per `GraphStage`, traced on the tensors the
function passed it. Every declared graph is called exactly once, and a graph's
inputs must not be modified in place after the call, since export traces the
graph on them; the run rejects a function that breaks either rule. The model
outputs are what `deployed_forward()` returns; PyTorch post-processing of the
deployed outputs belongs in `build_deployed_eval_output()`. Each `GraphStage`
module is export-ready as declared, the live submodule or the copy its
`prepare_for_export()` makes; the framework never alters it, and it declares
the graphs again after replacing a module rather than reusing an earlier
declaration. CenterPoint, BEVFusion (lidar-only) and the PTv3 models declare
their modules this way;
[Deployment graphs: verification and evaluation](#deployment-graphs-verification-and-evaluation)
describes the declaration in full and what else runs on it.

Models that do not declare deployment graphs yet (TransFusion, FRNet, the
calibration-status classifier, camera-lidar BEVFusion) override
`build_export_spec()` or `build_export_specs()` instead and return an explicit
export module plus example tensor inputs. That path exports, but nothing
verifies or scores its artifacts.

## Optional Graph Modification

Post-export ONNX graph modification is still available as a fallback, shared
across modules or per module:

```yaml
deploy:
  onnx:
    modules:
      <module>:
        modify_graph:
          _target_: my_module.OnnxGraphModifier
          # modifier-specific parameters
```

Use for operator replacement, shape inference fixes, or custom plugin insertion.
Modifiers run before the precision conversion and the metadata stamp, so the
shipped file reflects them.

## Deployment graphs: verification and evaluation

Export alone produces artifacts; it does not prove they compute what the PyTorch
model computes. A model can opt into that proof by declaring its deployment as the
graphs it exports (`GraphStage`, through `ModuleBaseModel.deploy_graphs()`) plus one
function that runs them with the PyTorch glue between them
(`ModuleBaseModel.deployed_forward()`):

```python
def deploy_graphs(self):
    return (
        GraphStage("pts_voxel_encoder_centerpoint", module=..., inputs=("input_features",),
                   outputs=("pillar_features",)),
        GraphStage("pts_backbone_neck_head_centerpoint", module=..., inputs=("spatial_features",),
                   outputs=("heatmap", "reg", ...)),
    )

def deployed_forward(self, batch, graphs):
    input_features = self.pts_voxel_encoder.encode(batch.voxels_data)   # glue reads the batch
    pillar_features = graphs["pts_voxel_encoder_centerpoint"](
        input_features=input_features)["pillar_features"]
    spatial_features = self.pts_middle_encoder(pillar_features.squeeze(1), ...)
    # The returned tensors are the model outputs.
    return graphs["pts_backbone_neck_head_centerpoint"](spatial_features=spatial_features)
```

Every `GraphStage` is one `deploy.onnx.modules.<name>` entry and one `<name>.onnx` /
`<name>.engine`; the export specs are derived from the declaration (the function runs once
in PyTorch on the example batch and each graph is traced with the tensors the function
passed it), so `build_export_specs()` needs no hand-written counterpart. The glue is plain
code, so the framework checks each graph call against its declaration instead (the input
names, tensors only, one call per run, every graph called) when the function runs; a
wiring mistake fails the first run, which export does. The model outputs are what the
function returns: the last graph's outputs in a chain like the one above, and every head's
outputs when several heads share one encoder (PTv3 seg + det). They are what a backend
hands to the model's
`build_deployed_eval_output(batch_inputs, outputs)`, which turns them into the flat dict
`build_eval_output` produces from `forward()`'s typed outputs — rebuilding the typed head
outputs when the graph emits them (CenterPoint, the PTv3 det head), or scoring what the
graph decoded itself (BEVFusion's packed boxes, the PTv3 seg head's labels and scores).
Models without deployment graphs keep their existing `build_export_specs()`; nothing below
applies to them.

The same function runs on three backends — `pytorch` (the modules), `onnx` (ONNX
Runtime sessions) and `tensorrt` (the engines) — with identical glue, so two backends
differ only by what executed the exported graphs. Two post-export steps use this, both
disabled by default (quantized checkpoints go through the same path, see
[Quantization](quantization.md)):

### Verification

```yaml
deploy:
  verification:
    enabled: true
    num_verify_batches: 1
    scenarios:
      - { ref: { backend: pytorch, device: cuda }, test: { backend: onnx, device: cuda }, tolerance: 0.01 }
      - { ref: { backend: pytorch, device: cuda }, test: { backend: tensorrt, device: cuda }, tolerance: 0.5 }
```

Each scenario runs the reference and test pipelines on the first predict batches and
compares the raw model outputs element wise; a failing scenario fails the deploy.
Tolerances are calibrated, not guessed: lossy backends (fp16 engines, int8) set an
explicit per-scenario `tolerance`, and a failing tensor's message suggests the gate that
would have passed so the observed value can be recorded in the config comment. Raw-logit
differences under fp16 are expected while the metrics below stay equal — the metrics are
the real gate; verification catches wiring and dtype mistakes.

Integer-typed outputs (class labels, indices) are decisions over the float outputs, so
they are gated on their mismatch *ratio* (5 %) rather than element-wise: a label flips
wherever two logits sit within backend noise of each other while the probabilities agree,
but a swapped or mis-wired integer output still fails.

Verification only makes sense for outputs whose positions mean the same thing on every
backend (a BEV cell, a point). A graph that selects proposals with a top-k inside the
export (BEVFusion, PTv3 detection: the TransFusion head) has no such outputs — near-tie
scores send different proposals to the same slot on different backends — so its configs
leave `deploy.verification` off and say why; enabling it fails on the first batch, and the
per-backend evaluation table is the comparison to read.

### Evaluation

```yaml
deploy:
  evaluation:
    enabled: true
    split: test          # or val
    num_samples: -1      # whole split; a positive number for a quick check
    batch_size: 1        # frames per batch; the latency below is per batch
    backends:
      pytorch: { enabled: true, device: cuda }
      onnx: { enabled: true, device: cuda }
      tensorrt: { enabled: true, device: cuda }
```

Every enabled backend with artifacts is scored on the split with the model's own metric
suites through its `build_deployed_eval_output`, and the latency of each graph and of the
glue is collected (`model_graphs` sums the exported graphs: pure GPU time for TensorRT). The split is read
at `batch_size` (one frame by default) rather than at its training-time dataloader
setting: the deployed graphs are single-frame, and a per-batch latency at batch 1 is the
per-frame figure the runtime sees. Evaluation runs
with `cpu_threads: 1` by default: an engine with data-dependent shapes (sparse
convolutions) waits on the host at its shape-sync points, and the BLAS / OpenMP workers a
preceding numpy or CPU-torch op leaves busy-waiting double its measured time (BEVFusion:
10-13 ms vs the true 5.5 ms).
Set it to 0 to keep the process default when only the metrics matter. Metric keys carry
the backend, `{split}/{backend}/{suite}/{metric}` (`test/tensorrt/det3d/mAP_0m_121m`),
latencies land under `latency/{backend}/{graph}_mean_ms` (and `glue_mean_ms`), and everything is logged to the
deploy run and written to `evaluation.json` next to the exports (every metric and latency of every
backend, with or without a tracker). The log ends with a cross-backend table of the suites'
headline metrics (det3d: `mAP` and NDS, reported as `map_based_nds`; seg3d: `mIoU` and
`fwIoU`); a backend column whose graphs ran in PyTorch (a declared fallback, e.g. a plugin
graph on ONNX Runtime) is starred.

## Overriding at Runtime

Override deployment settings from CLI:

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.input_shapes.input.opt_shape=[1,3,256,256] \
    deploy.tensorrt.workspace_size=8589934592
```
