---
icon: lucide/package
---

# Deployment

Autoware-ML exports trained models for production use, checks that the exported
graphs match the PyTorch model numerically, and scores every backend against
ground truth — all derived from one declaration on the model: its **stage graph**.

## Deployment Pipeline

```text
Checkpoint (.ckpt) -> export  (<stage>.onnx, <stage>.engine per exportable stage)
                   -> verify  (pytorch vs onnx vs tensorrt on raw outputs)
                   -> evaluate (GT metrics + latency per backend, same keys as `test`)
```

## Basic Usage

```bash
autoware-ml deploy \
    --config-name experiments/<task>/<model>/<config> \
    --weights mlruns/<...>/artifacts/checkpoints/best.ckpt
```

`deploy` operates on `experiments/` configs. `--weights` accepts one or more
checkpoint paths; later checkpoints overwrite earlier ones on overlapping keys, and
every model parameter must be covered. A **quantized** checkpoint produced by
`autoware-ml quantize` is detected automatically from its embedded description —
no `quantization` config is needed to deploy it (see
[Quantization](../framework/quantization.md)).

Artifacts land in the deploy run's MLflow artifact directory under `exports/`, one
`<stage_name>.onnx` / `<stage_name>.engine` per exportable stage. Disable either
exporter during iteration; verification and evaluation then reuse the artifacts
already present in that directory:

```bash
autoware-ml deploy --config-name experiments/<...> --weights <ckpt> \
    deploy.tensorrt.enabled=false
```

## The Stage Graph

A model declares its inference as an ordered list of stages over named tensors
(`MultiTaskBaseModel.build_stages()`):

- **`GraphStage`** — exportable: a module plus the names it reads (its ONNX inputs)
  and writes (its ONNX outputs). One `GraphStage` = one artifact.
- **`TorchStage`** — glue that is never exported (pillar decoration, BEV scatter);
  it runs in PyTorch on every backend.

```text
CenterPoint:  pillar_decorate -> pts_voxel_encoder -> scatter -> pts_backbone_neck_head
              (torch)           (graph)              (torch)    (graph)
```

From that one declaration the framework derives the export units and their trace
inputs, the artifact names, the per-backend inference pipeline, verification, and
the latency breakdown. `forward()` stays hand-written for training; a test pins it
to the staged PyTorch run.

## Configuration

Global options live under `deploy.onnx` / `deploy.tensorrt`; everything
shape-related is per stage under `deploy.stages.<stage_name>`, keyed by the names
the model declares (a name that does not match raises):

```yaml
deploy:
  onnx:
    enabled: true
    dynamo: false            # legacy exporter for models relying on symbolic functions
    opset_version: 17
    do_constant_folding: false
    # fp16 converts every exported stage WITHOUT Q/DQ nodes to mixed FP16 (ModelOpt
    # AutoCast); quantized stages keep the precision their checkpoint bakes in.
    # Engines always build strongly typed, so this is where FP16 is decided.
    precision: fp16
  tensorrt:
    enabled: true
    workspace_size: 4294967296
    # No precision knob here: engines build strongly typed (see deploy.onnx.precision).
  stages:
    pts_voxel_encoder:
      onnx:
        # Optional per-stage override of deploy.onnx.precision (fp32 | fp16) — for a
        # pipeline whose stages need different precisions; unset inherits the global.
        # precision: fp32
        dynamic_axes:
          input_features: { 0: num_voxels, 1: num_max_points }
          pillar_features: { 0: num_voxels }
      tensorrt:
        input_shapes:
          input_features:
            min_shape: [1000, 32, 11]
            opt_shape: [20000, 32, 11]
            max_shape: [96000, 32, 11]
    pts_backbone_neck_head:
      onnx:
        dynamic_axes:
          spatial_features: { 0: batch_size, 2: H, 3: W }
      tensorrt:
        input_shapes:
          spatial_features:
            min_shape: [1, 32, 1020, 1020]
            opt_shape: [1, 32, 1020, 1020]
            max_shape: [1, 32, 1020, 1020]
```

With `dynamo: true`, use `dynamic_shapes` (`{input: {dim: name | {name, min, max}}}`)
instead of `dynamic_axes`. Input and output *names* are never configured — they are
the stage declaration.

!!! tip
    TensorRT optimizes most aggressively for `opt_shape`. Set it to your typical
    inference shape.

### Verification

```yaml
deploy:
  verification:
    enabled: true
    tolerance: 0.01           # strict default for FP32 graphs
    num_verify_batches: 1
    scenarios:
      - { ref: { backend: pytorch, device: cuda }, test: { backend: onnx, device: cuda } }
      - { ref: { backend: onnx, device: cuda }, test: { backend: tensorrt, device: cuda }, tolerance: 1.0 }
```

Each scenario runs both backends on the same preprocessed batches and compares the
final raw graph outputs element-wise. Lossy backends (fp16 engines, INT8) set an
explicit per-scenario `tolerance` rather than loosening the default. A failing
scenario fails the deploy run.

### Evaluation

```yaml
deploy:
  evaluation:
    enabled: true
    num_samples: -1           # -1 = whole test split
    num_warmup: 2
    backends:
      pytorch: { enabled: true, device: cuda }
      onnx: { enabled: true, device: cuda }
      tensorrt: { enabled: true, device: cuda }
```

Every backend is scored with the model's own metric suites — the same code
`autoware-ml test` runs — and reported under the same keys:
`test/<backend>/<suite>/<metric>` (e.g. `test/tensorrt/detection3d/mAP`) plus
`latency/<backend>/<stage>_mean_ms`. PyTorch is one backend among three, so the three
columns line up in MLflow.

## Optional Graph Modification

Post-export ONNX graph modification is available as a fallback and applies to every
exported stage:

```yaml
deploy:
  onnx:
    modify_graph:
      _target_: my_module.OnnxGraphModifier
      # modifier-specific parameters
```

## Overriding at Runtime

```bash
autoware-ml deploy --config-name experiments/<...> --weights <ckpt> \
    deploy.stages.pts_backbone_neck_head.tensorrt.input_shapes.spatial_features.opt_shape=[1,32,1020,1020] \
    deploy.evaluation.num_samples=1000
```

## Adding Deployment to a Model

Implement these hooks on your `MultiTaskBaseModel` subclass, in the model's own
directory (`models/<task>/main_modules/<model>/`):

1. `build_stages()` — the stage graph (`stages.py`).
2. Turn the final stage's tensors — which the stage declares as `(onnx_name, key)`
   pairs — into predictions. Which hook depends on what the deployed graph emits:
   - the head's raw output maps: implement `assemble_outputs(fields)` and the default
     `assemble_predictions` decodes them with the model's own `decode_outputs`;
   - detections, because the runtime ABI decodes in-graph: override
     `assemble_predictions(fields)` instead. There is no head output to rebuild, so
     `assemble_outputs` stays unimplemented. Reuse the head's post-processing rather
     than restating it, so the deployed behaviour cannot drift from the model's
     (see `TransFusionHead.decode_detections`).
3. `build_eval_output_from_predictions(batch, predictions)` — pair predictions with
   ground truth for the metric suites. Training, validation and deployment all score
   through this one method.
4. `build_quantization_plan(config)` — only if the model supports INT8
   (`quantization.py`).

Nothing model-specific goes into `autoware_ml/deployment`, `autoware_ml/evaluation`,
or `autoware_ml/quantization`; a test asserts no model name appears there.
