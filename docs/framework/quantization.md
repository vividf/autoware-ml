# Deployment, evaluation, and quantization

This page covers the deployment stack (stage graph → export → verification →
evaluation), the INT8 quantization stack (PTQ / QAT), and how the three fit
together. Everything here is model-agnostic; CenterPoint is the reference model.

## Overview

```text
autoware-ml train    --config-name experiments/...            # FP training
autoware-ml quantize --config-name experiments/..._int8 \
    --weights <FP best.ckpt>                                  # PTQ or QAT -> self-describing checkpoint
autoware-ml deploy   --config-name experiments/..._int8 \
    --weights <ptq.ckpt / best.ckpt>                          # export + verify + evaluate
autoware-ml test     --config-name experiments/... \
    --weights <any .ckpt, FP or quantized>                    # trainer.test on the pytorch backend
```

A deployment run (`deploy`) is three peer stages sharing one exported-artifact
directory and one set of backend pipelines:

```text
export        one ONNX (+ TensorRT) artifact per exportable stage       [deploy.onnx / deploy.tensorrt]
verification  cross-backend numerical parity on the final raw outputs   [deploy.verification]
evaluation    per-backend GT metrics + latency, same keys as `test`     [deploy.evaluation]
```

`deploy` and `test` never read a `quantization` config section: a quantized
checkpoint carries its own description (see *Self-describing checkpoints*), and
`build_model` rebuilds the quantized tree from it.

## Architecture

```text
autoware_ml/deployment/            model-agnostic; no model name appears here
├── stages.py        Stage graph declaration: GraphStage (exportable) / TorchStage (glue)
├── config.py        DeployConfig — typed `deploy:` section, typo-guarded, per-stage layout
├── export.py        export_stages(): trace inputs + ONNX/TRT artifacts derived from the graph
├── onnx_export.py   the torch.onnx.export primitive + ONNX graph modification
├── pipeline.py      StagedPipeline (run the graph on pytorch / onnx / tensorrt) + PipelineCache
├── backends/        tensorrt_builder (ONNX -> .engine) + OnnxModuleRunner / TensorRTModuleRunner
└── verification/    OutputComparator + scenario-driven BackendVerifier

autoware_ml/evaluation/            one evaluate loop, PyTorch is a backend too
├── evaluator.py     evaluate_backend(model, dataloader, pipeline, device) -> EvaluationResult
└── latency.py       LatencyStats

autoware_ml/metrics/report.py      the metric-key convention: {split}/{backend}/{prefix}/{metric}
                                   shared by MetricEvalMixin (trainer) and evaluation/

autoware_ml/quantization/
├── config.py        typed `quantization:` section (typo guard, recipe-name guard, Precision enum,
│                    CalibrationConfig = the amax algorithm)
├── plan.py          QuantRules (per-model declaration) + QuantizationPlan (the stage interface)
│                    + PlacementRecord (recorded placement decisions)
├── checkpoint.py    self-describing checkpoints: config + placement record embedded next to state_dict
├── loader.py        rebuild + verify + load from that description
├── qat_callback.py  Lightning callback: frozen-amax STE fine-tuning; embeds the description on save
├── core/            engine on nvidia-modelopt: descriptor tables, in-place conversion through
│                    modelopt's QuantModuleRegistry (replace.py), BN fusion, Calibrator, quantizer state,
│                    the two modelopt bug patches (modelopt.py)
└── recipes/         architecture recipes, class-gated: residual_add (ResidualBlockSpec rows ->
                     QuantModule block classes, in place), ese (VoVNet eSE single-Q placement)
                     and maxpool (QuantBeforePool wrapper)

The engine owns no quantized module classes of its own: an `nn.Conv2d` becomes modelopt's
`QuantConv2d` by patching the instance's class in place (`QuantModuleRegistry.convert`), so the
object, its weights and `isinstance(m, nn.Conv2d)` all survive, and the calibrated scales live
under modelopt's names — `<layer>.input_quantizer._amax` / `<layer>.weight_quantizer._amax`
(SmoothQuant adds `input_quantizer._pre_quant_scale`). Residual blocks are converted the same
way through the recipes' `QuantBlockRegistry`.

Precision is wired end to end: `quantization.default_precision` flows from the plan
through the replace engine and the recipes into `core/descriptors.py`, the single
per-precision descriptor table. Adding a precision (e.g. FP8) = a `Precision` enum
member + one row per descriptor table (+ a capability check in the TensorRT builder).
The activation calibrator (histogram vs max) follows `quantization.calibration`; it changes
no state_dict key.

autoware_ml/models/detection3d/main_modules/centerpoint/   everything CenterPoint-specific
├── model.py         CenterPointDetectionModel (forward / loss / decode + the three hooks below)
├── stages.py        the stage graph + export wrappers + output-field table
└── quantization.py  CENTERPOINT_QUANT_RULES
```

### Model integration: three hooks

A model with residual / eSE blocks of its own declares them in its `QuantRules.residual_blocks`
/ `ese_blocks` (`ResidualBlockSpec(block_cls, quant_block_cls, share_from, fresh_if_downsample,
osa_concat)`, `ESEBlockSpec(block_cls, quant_block_cls)`); the framework ships the quantized
block classes for VoVNet `_OSA_module` / `eSEModule` (`recipes/quant_blocks.py`) and the spec
for the repo's own spconv `SparseBasicBlock` (the ConvNeXt block is parked on branch
`feat/quantization-convnext-recipe`). Recipes
fire only inside the submodules listed in `quantize_submodules` — a block whose convolutions
stay FP gets no residual, eSE or pool Q/DQ.

A model supports deployment and quantization by implementing three methods on
`MultiTaskBaseModel`:

| Hook | Returns | Used by |
| --- | --- | --- |
| `build_stages()` | ordered `Stage`s (`GraphStage` = one ONNX/TRT artifact, `TorchStage` = glue that always runs in PyTorch) | export (trace inputs, artifact names, I/O names), pipelines, verification, evaluation, latency breakdown |
| `assemble_predictions(fields)` | predictions from the final stage's tensors, keyed as the stage declares | evaluation (metrics on any backend). Default composes `assemble_outputs` + `decode_outputs`, so a model whose graph emits the head's raw maps implements only `assemble_outputs` |
| `build_quantization_plan(config)` | the model's `QuantizationPlan` | quantize (PTQ / QAT) and the checkpoint loader |

`forward()` stays hand-written for training. The stage graph is what deploys;
`tests/deployment/test_centerpoint_stages.py` pins the two to each other
(`forward == StagedPipeline(pytorch)`), so they cannot drift silently.

Adding a model = one directory `models/<task>/main_modules/<model>/{model,stages,quantization}.py`.
`grep -ri <model> autoware_ml/deployment autoware_ml/evaluation autoware_ml/quantization` must stay empty
(a test enforces it for identifiers and string literals).

### Stage graph

```text
CenterPoint:  pillar_decorate (torch) -> pts_voxel_encoder (graph) -> scatter (torch) -> pts_backbone_neck_head (graph)
```

- A `GraphStage` declares `inputs` / `outputs` — these *are* its ONNX input/output
  names — and, on the final stage, `output_fields` (`(onnx_name, dataclass_field)`).
- Artifacts are `<output_dir>/<stage_name>.onnx` and `.engine`; nothing is named twice.
- On the `onnx` / `tensorrt` backend each graph stage is replaced by its artifact's
  runner; glue stages run in PyTorch on every backend. Mixing is therefore visible
  data (`exportable`), not code hidden in a per-model pipeline.
- Verification compares the final stage's raw outputs; evaluation reassembles them
  via `assemble_predictions` and scores with the model's own metric suites.
- Export folds BatchNorm into the preceding conv/linear on a copy (an inference
  identity): deployed graphs never carry a BatchNormalization node, so the FP ONNX
  weights are BN-fused relative to the checkpoint — the same layout the quantized
  flow produces (whose plan folds BN before calibration).

## Config surface

```yaml
deploy:
  onnx:      { enabled: true, dynamo: false, opset_version: 17, do_constant_folding: false, precision: fp16, modify_graph: null }
  tensorrt:  { enabled: true, workspace_size: 4294967296, plugin_libraries: [] }
  stages:                                 # keyed by the model's stage names
    pts_voxel_encoder:
      onnx:     { dynamic_axes: { input_features: { 0: num_voxels, 1: num_max_points }, pillar_features: { 0: num_voxels } } }
      tensorrt: { input_shapes: { input_features: { min_shape: [1000, 32, 11], opt_shape: [20000, 32, 11], max_shape: [96000, 32, 11] } } }
    pts_backbone_neck_head:
      onnx:     { dynamic_axes: { spatial_features: { 0: batch_size, 2: H, 3: W }, heatmap: { 0: batch_size, 2: H, 3: W }, ... } }
      tensorrt: { input_shapes: { spatial_features: { min_shape: [1, 32, 1020, 1020], opt_shape: [...], max_shape: [...] } } }
  verification:
    enabled: true
    tolerance: 0.01                       # strict default for FP32 graphs; lossy backends set per-scenario tolerances
    num_verify_batches: 1
    scenarios:
      - { ref: { backend: pytorch, device: cuda }, test: { backend: onnx, device: cuda } }
      - { ref: { backend: onnx, device: cuda }, test: { backend: tensorrt, device: cuda }, tolerance: 1.0 }
  evaluation:
    enabled: true
    num_samples: -1                       # -1 = whole test split (predict dataloader)
    num_warmup: 2                         # extra re-runs of the first batch, discarded
    backends:
      pytorch: { enabled: true, device: cuda }
      onnx: { enabled: true, device: cuda }
      tensorrt: { enabled: true, device: cuda }

quantization:                             # read by `quantize` only
  enabled: true
  mode: ptq                               # ptq | qat  (a stage block under the wrong mode raises)
  fuse_bn: true
  default_precision: int8                 # int8 is the only supported value today
  skip_quantize: [pts_voxel_encoder]      # glob patterns, subtree match, zero-match warns
  calibration: mse                        # mse (default) | entropy | percentile | max | smoothquant;
                                          # or {method: percentile, percentile: 99.99} / {method: smoothquant, smoothquant_alpha: 0.5}
  disable_recipes: []                     # residual_add | ese | maxpool — an unknown name raises
  dry_run: false                          # true: log the placement record and exit (no GPU, no data)
  ptq: { calibrate_samples: 400, batch_size: 1, calib_seed: 0, calib_shuffle: false }
  # qat: { epochs: 3, lr: 1.0e-5, schedule: cosine, freeze_unquantized: true, val_check_interval: 0.25, calibrate_samples: 400 }
```

Every mapping rejects unknown keys — a misspelled option would otherwise silently fall
back to a default. `deploy.stages` names are checked against the model's declaration.

### Metric keys

Every metric is keyed `{split}/{backend}/{suite_prefix}/{metric}`, e.g.
`test/pytorch/detection3d/mAP` from `trainer.test` and `test/tensorrt/detection3d/mAP`
from `deploy`. Latency is `latency/{backend}/{stage}_mean_ms`. The same metric on
the same split therefore lines up in MLflow regardless of what ran the forward.
(Validation metrics follow the same rule — `val/pytorch/...`; checkpoint monitors
must use the full key.)

## Quantization

Reference PTQ recipe: **400 samples @ batch_size=1, seed 0, histogram + MSE
amax** (`quantization.calibration: mse`), calibrated on the **validation split** through
the clean test-time pipeline. `calibration` is part of the recipe and travels inside the
checkpoint: `entropy` / `percentile` are the other histogram estimators, `max` is the FP8
convention (and the fastest), `smoothquant` migrates activation outliers of every INT8
`Linear` into its weight (modelopt's SmoothQuant; convolutions unaffected; exports as a
`Mul` before the Q/DQ pair). QAT recipe (Wu et al. 2020 *Integer Quantization for Deep Learning
Inference* §7 / App. A.2; confirmed on the CenterPoint replay 2026-08-27):
`epochs` ≈ 10% of the original training, `lr` = the schedule **peak** ≈ 1% of the
original training's peak lr (1e-3 → **1e-5** here), `schedule: cosine` (start at
the peak, no warmup, anneal to 1/100). `one_cycle` (the CUDA-CenterPoint shape) is
available but its 1e-4 peak degraded the converged model within an epoch.
`freeze_unquantized: true` freezes the `skip_quantize` subtrees during QAT (without
it the model collapsed to mAP 0 — un-quantized layers take no STE masking and drift
past the frozen downstream amax); `val_check_interval: 0.25` validates several
times per epoch because QAT quality peaks early. `calibrate_samples` counts
**samples** in both modes: QAT calibrates at epoch 0 on the val dataloader.

Batch sizes: PTQ calibration runs at `quantization.ptq.batch_size` (default 1 — the
release reference; a larger batch slightly changes the histogram statistics). QAT
trains at the experiment config's training batch size. Deploy evaluation follows the
dataloader batch size on the PyTorch backend; the TensorRT backend is bound by the
engine's shape profiles (the shipped CenterPoint engines are built for batch 1).

### Self-describing checkpoints

A quantized checkpoint is `{"state_dict": ..., "quantization": {config, placement_record}}`
— for QAT, the Lightning checkpoint plus that same `quantization` entry, written by
`QATCallback.on_save_checkpoint`. There are no sidecar files: the calibrated
`_amax` buffers live in the `state_dict` like any other buffer.

Loading (`build_model` → `quantization.loader`):

1. detect the entry (`find_quantization`) — at most one `--weights` may carry it;
2. `model.build_quantization_plan(embedded config).prepare(model)` — the SAME plan
   the quantize step built;
3. verify the rebuilt placement record against the embedded one — structural drift
   is a hard failure, not a silent weight mis-map;
4. `load_state_dict(strict=False)` + full-coverage check, disable the
   `skip_quantize` quantizers, validate every enabled amax, configure ONNX export.

## Invariants (do not break these)

1. **Same-plan-everywhere.** PTQ, QAT, and the loader all build the quantized
   module tree through the model's one `build_quantization_plan(config).prepare(model)`
   (`tests/quantization/test_tree_parity.py`); the recorded placement record travels
   inside the checkpoint and is machine-checked on load.
2. **`fuse_bn` changes state_dict keys.** BN is fused across the whole model
   regardless of `skip_quantize` — quantize and load must fuse the exact same set;
   `skip_quantize` only subtracts from the *quantized* set.
3. **Precision lives in the ONNX; engines build strongly typed.** TensorRT reads
   INT8 from the QuantizeLinear/DequantizeLinear nodes modelopt bakes into the
   graph, and FP16 from the tensor types AutoCast writes into the non-quantized
   stages (`deploy.onnx.precision: fp16`). There are no builder precision flags
   (TensorRT deprecated weak typing in 10.12 and removed it in 11). Never bypass
   quantizers during ONNX tracing.
4. **QAT hard boundaries:** single device, `precision: 32-true`, no resume. The
   callback enforces these and fails loud.
5. **One amax health policy.** `validate_quantizer_amax` (PTQ, QAT, loader alike):
   `None` / NaN / Inf is fatal; a finite non-positive amax (dead channel) is clamped
   to 1e-8 with a warning.
6. **Frozen output ABI.** The stage names (`pts_voxel_encoder`,
   `pts_backbone_neck_head`) and the head output order
   (`heatmap, reg, height, dim, rot, vel`) are the Autoware runtime contract; they
   are declared once in `centerpoint/stages.py`.

## Installation

nvidia-modelopt (0.46.0) is a required dependency — every environment installs it
with the project; there is no separate `[quant]` extra.

## Evaluation semantics

Each backend is scored with a fresh clone of the model's own metric suites
(`Detection3DMetricSuite` — the same code `trainer.test` runs), so cross-backend
metric differences are pure backend differences. Latency is reported per stage; the
`model_graphs` entry sums only the exportable stages (pure GPU time for TensorRT),
and warmup re-runs are excluded from both metrics and timing. Verification and
evaluation share one `PipelineCache`, so every ONNX session / TensorRT engine is
loaded once per deploy run.
