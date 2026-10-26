# PTv3 sparse-conv PTQ for t4sparse (PoC)

Frozen branch for RDDR-1286. It is the deployment stack tip (`recut/09-docs`, `08c0bf30`, base upstream
`1c355005`) plus this directory. It will not be force-pushed or rebased; the stack itself moves on in
`recut/*`. Not meant for upstream.

autoware-ml does the calibration: real validation-split samples through the full PTv3 preprocessing and
model. t4sparse's `tools/ptq.py` writes the t4sparse graph. The script here carries the calibrated
activation ranges from one to the other.

## Flow

```bash
# 1. Calibrate: 12 cpe sparse convs (10 encoder, 2 seg head) get an activation amax in ptq.ckpt.
#    fp8: method=max. int8: mse or percentile. The config's own default (smoothquant) max-calibrates the convs.
autoware-ml quantize \
  --config-name segmentation3d/ptv3/t4dataset/voxel012_tiny_b16_50e_t4dataset_122m_j6gen2_semaseg_int8_sparse \
  --weights <fp best.ckpt> \
  quantization.calibration.method=max quantization.ptq.calibrate_samples=400

# 2. FP16 export from this same branch (node and initializer names must match the checkpoint).
autoware-ml deploy \
  --config-name segmentation3d/ptv3/t4dataset/voxel012_tiny_b16_50e_t4dataset_122m_j6gen2_semaseg \
  --weights <fp best.ckpt>

# 3. Checkpoint -> the {node: {in_amax, out_amax}} JSON tools/ptq.py --calib reads.
#    Fails if a calibrated cpe conv has no ImplicitGemm node.
python poc/t4sparse/ptq_ckpt_to_t4sparse_calib.py ptq.ckpt ptv3_encoder.onnx ptv3_seg3d_head.onnx -o calib.json

# 4. In t4sparse: tools/fold_cpe_linear.py, then tools/ptq.py --calib calib.json --dtype {e4m3,int8} [--per-channel]
```

## Caveats

- Do not build the INT8 ONNX that `deploy` writes for a quantized checkpoint against libt4sparse. That graph
  is in the autoware plugin's INT8 form (`precision=1`, fp16 weights, `channel_scale` / `bias_scaled`
  inputs). t4sparse ignores `precision` and applies `input_scale` as alpha, so the engine runs and the outputs
  are wrong. Start from the FP16 export (step 2).
- The spconv INT8 arm is `autoware-ml deploy` of the same `ptq.ckpt`. Use the same checkpoint for both arms so
  the activation ranges are identical. It needs the INT8 build of `libautoware_tensorrt_plugins`, pinned to
  autoware_universe#13369 by `docker/tensorrt_plugins/build_plugin.sh`.
- Calibration needs PyTorch, spconv, nvidia-modelopt and a GPU, but no TensorRT plugin. This branch's pixi
  environment has modelopt.
- Checkpoints from this branch are not guaranteed to load on the upstream version. Re-run step 1 when moving.
