# TensorRT plugins for sparse-convolution graphs

Models whose deployed graph contains sparse convolutions (BEVFusion's `bevfusion_sparse`
stage today) export ONNX nodes in the `autoware` domain —
`autoware::GetIndicePairsImplicitGemm` and `autoware::ImplicitGemm`. TensorRT has no
implementation for them, so an engine build fails with

```text
IPluginRegistry::getCreator: Cannot find plugin: ImplicitGemm, version: 1, namespace:.
```

unless `libautoware_tensorrt_plugins.so` is loaded first. ONNX Runtime cannot run those
nodes at all — a plugin is a TensorRT concept — which is why such stages declare
`torch_fallback_backends=(Backend.ONNX,)` and fall back to PyTorch there.

## What is here

| File | Purpose |
| --- | --- |
| `build_spconv_cpp.sh` | Builds and installs the **cumm + spconv C++** packages from [autowarefoundation/spconv_cpp](https://github.com/autowarefoundation/spconv_cpp). The plugin's sparse kernels come from libspconv; the pip `spconv` wheel does not provide them. |
| `CMakeLists.txt` | Standalone build of [autoware_universe/perception/autoware_tensorrt_plugins](https://github.com/autowarefoundation/autoware_universe/tree/main/perception/autoware_tensorrt_plugins) without ament/colcon. |
| `build_plugin.sh` | Resolves TensorRT, fetches version-matched headers, builds the plugin, installs it to `/opt/plugins`. |

The image builds both (see `docker/Dockerfile`), so containers already carry
`/opt/plugins/libautoware_tensorrt_plugins.so`.

## Using it

Point the deploy config at the library; TensorRT loads it before parsing ONNX:

```yaml
deploy:
  tensorrt:
    plugin_libraries: [/opt/plugins/libautoware_tensorrt_plugins.so]
```

A stage that needed the plugin should also drop `Backend.TENSORRT` from its
`torch_fallback_backends`, otherwise the framework keeps running it in PyTorch and skips
its engine build.

## Rebuilding by hand

Inside a container, against a local `autoware.universe` checkout:

```bash
bash docker/tensorrt_plugins/build_spconv_cpp.sh        # only once per image
PLUGINS_SRC=/path/to/autoware.universe/perception/autoware_tensorrt_plugins \
  bash docker/tensorrt_plugins/build_plugin.sh
```

Without `PLUGINS_SRC` the script clones `autoware_universe@main`; `PLUGINS_REPO` /
`PLUGINS_REF` select a different source (a feature branch for an A/B build, say).

## Two traps worth knowing

1. **Header/runtime version match.** The pip TensorRT wheels ship libraries but no C++
   headers, and Ubuntu's `libnvinfer-dev` is usually a different major.minor. Building
   against mismatched headers while linking the pip libraries yields an
   `IPluginCreatorV3One` vtable mismatch that **segfaults inside `createPlugin` during ONNX
   parsing** — it does not look like a version problem. `build_plugin.sh` therefore
   refuses anything but headers matching the runtime, and fetches them from TensorRT OSS
   (`release/<major.minor>`) rather than apt.
2. **The `do_sort` attribute.** The graphs exported by `autoware_ml.ops.spconv` set
   `do_sort` on `GetIndicePairsImplicitGemm`. A plugin built from a source tree without
   that attribute rejects them; `build_plugin.sh` warns when the source lacks it.

## CUDA architectures

`CMakeLists.txt` compiles for `80;86;87;89;90;120` plus PTX for 120, matching
`TORCH_CUDA_ARCH_LIST` in the Dockerfile. Override with `-DCUDA_ARCHS=...` when targeting
something else (a Jetson build, say).
