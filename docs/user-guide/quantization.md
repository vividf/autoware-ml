<!-- cspell:ignore modelopt smoothquant amax -->

# Quantization

Autoware-ML quantizes a trained model in two steps that mirror deployment: `autoware-ml
quantize` turns the floating-point checkpoint into a **self-describing quantized
checkpoint**, and `autoware-ml deploy` / `autoware-ml test` take that checkpoint like any
other `--weights`. Nothing about the quantization travels in the deploy or test config —
the checkpoint carries the exact module tree it was calibrated on, so the export cannot
drift from the calibration.

```bash
autoware-ml quantize \
    --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2_int8 \
    --weights mlruns/.../best.ckpt
autoware-ml deploy \
    --config-name detection3d/centerpoint/voxel024_second_secfpn_120m_t4dataset_j6gen2_int8 \
    --weights <run>/checkpoints/ptq.ckpt
```

The engine is [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
(fake quantization in PyTorch, Q/DQ nodes in ONNX); TensorRT then builds strongly typed
engines whose INT8 / FP8 regions are exactly the ones the checkpoint describes.

## Configuration

The `quantization` section (defaults in `autoware_ml/configs/defaults/modules/quantization.yaml`,
schema in `autoware_ml/quantization/config.py`) is off unless a variant enables it:

```yaml
quantization:
  enabled: true
  mode: ptq                 # ptq | qat
  fuse_bn: true             # fold BatchNorm into the preceding conv before calibration
  default_precision: int8   # int8 | fp8, for every module the model's rules leave open
  skip_quantize:            # glob patterns over named_modules; the whole subtree stays FP
    - pts_voxel_encoder
    - pts_backbone.blocks.0
  calibration:
    method: mse             # mse | entropy | percentile | max | smoothquant
  ptq:
    calibrate_samples: 400
    batch_size: 1
    calib_seed: 0
    calib_shuffle: false
  dry_run: false            # print the placement table and stop
```

`quantization.dry_run=true` builds the weightless model on the CPU, resolves the placement
and prints one line per module — which transform it gets (`int8`, `fp8`, `spconv int8`,
skipped) and the rule that decided it. Use it before any calibration run; it is the
fastest way to check a `skip_quantize` pattern.

### Model rules and the plan

Which layers *may* be quantized is the model's business, not the config's. A model
declares `QuantRules` through `BaseModel.build_quantization_rules()`:

```python
CENTERPOINT_QUANT_RULES = QuantRules(
    quantize_submodules={
        "pts_backbone": ("conv",),
        "pts_neck": ("conv",),
        "bbox_head": ("conv",),
    },
)
```

Per submodule the rules list the layer kinds to quantize (`conv`, `linear`, `spconv`) and
optionally pin a precision (`{"spconv": "int8"}` — the sparse plugin has no FP8 path;
`{"linear": "fp8"}` for attention projections that INT8 damages). The framework binds the
rules to the config into a `QuantizationPlan` (`skip_quantize` and `default_precision`
applied), replaces the selected modules with their ModelOpt-quantized counterparts and
records the placement (`PlacementRecord`) into the checkpoint. `deploy` and `test` rebuild
the same tree from that record before loading the weights, so an FP checkpoint and a
quantized one are loaded through the same `load_model_weights`.

Recipes (`autoware_ml/quantization/recipes/`) are per-module-class adjustments applied
inside the quantized subtree — sharing an input quantizer across a residual add, keeping
the TransFusion decoder's packed attention weights FP — and are listed by class in the
placement table. `disable_recipes: [<name>]` opts out of one.

### Calibration

PTQ runs the model's test-time pipeline over `calibrate_samples` validation samples
(deterministic with `calib_seed` / `calib_shuffle`) and picks each quantizer's `amax` with
`calibration.method`: `mse` (default, histogram-based), `entropy`, `percentile`
(`calibration.percentile`), `max`, or `smoothquant` (`calibration.smoothquant_alpha`, default 0.5),
which migrates activation outliers into the weights of Linear layers before the
per-tensor scale is taken (PTv3's attention / FFN linears need it).

### QAT

`mode: qat` starts from the same prepared tree, calibrates once, then fine-tunes with the
quantizers live. The schedule is explicit — there is no silent default:

```yaml
quantization:
  mode: qat
  qat:
    epochs: 3
    lr: 1.0e-5
    schedule: { type: cosine, final_lr_ratio: 0.01 }
    freeze_unquantized: true   # un-quantized layers keep their FP weights
    val_check_interval: 0.25
    calibrate_samples: 400
```

`freeze_unquantized` matters: letting the FP layers drift while the quantized ones sit on
frozen `amax` values is the usual way a QAT run diverges. The run's `best.ckpt` is
self-describing like the PTQ one.

## What the export does with it

- **Dense layers** (Conv, Linear) export as `QuantizeLinear` / `DequantizeLinear` pairs
  (INT8) or `TRT_FP8QuantizeLinear` / `TRT_FP8DequantizeLinear` (FP8). The Q/DQ scales
  are folded into the graph and BatchNorm is folded at export
  (`autoware_ml/utils/bn_fusion.py`), matching what the calibration saw.
- **Sparse convolutions** (BEVFusion's encoder, PTv3's cpe convs) export as
  `autoware::ImplicitGemm` plugin nodes; a calibrated one carries `precision=1` plus its
  `channel_scale` / `bias_scaled` inputs instead of Q/DQ. See
  [Deployment › Plugin graphs](deployment.md#plugin-graphs-sparse-convolutions).
- **Precision.** Quantized graphs use `deploy.onnx.precision: fp16` with opset 19 (fp16-typed
  Q/DQ); the cast is island-aware, so Linear Q/DQ regions keep fp32 mini-islands where fp16
  scales would round differently from the calibration, and the graph I/O stays fp32.
- **Verification** tolerances in the `_int8` variants are measured values (raw-logit
  differences under INT8 are large while the metrics stay equal); each config comment
  records the observation the gate came from.

## Shipped variants

| Config | Recipe |
| --- | --- |
| `detection3d/centerpoint/..._j6gen2_int8` | SECOND (except `blocks.0`), FPN, dense head INT8; PFN fp16 |
| `detection3d/bevfusion/lidar_..._j6gen2_int8` | Dense towers INT8 (first backbone / neck stages and the decoder skipped); sparse encoder fp16 |
| `detection3d/bevfusion/lidar_..._j6gen2_int8_sparse` | As above plus the sparse encoder from stage 3.1 onwards as INT8 plugin nodes |
| `segmentation3d/ptv3/..._j6gen2_int8` | Attention / FFN linears INT8 with SmoothQuant; cpe sparse convs fp16 |
| `segmentation3d/ptv3/..._j6gen2_int8_sparse` | As above plus the twelve cpe sparse convs as INT8 plugin nodes |
| `segmentation3d/ptv3/..._j6gen2_fp8` | Linears FP8 (E4M3); cpe convs fp16 |

Measured accuracy and latency for each live in the model pages
([CenterPoint](../models/centerpoint.md#int8), [BEVFusion](../models/bevfusion.md#int8),
[PointTransformerV3](../models/ptv3.md#int8-and-fp8)).
