#!/usr/bin/env bash
# Build libautoware_tensorrt_plugins.so — the TensorRT custom operators the deployed
# sparse-convolution graphs bind against (autoware::GetIndicePairsImplicitGemm,
# autoware::ImplicitGemm, ...). Without it TensorRT refuses to parse those graphs
# ("Cannot find plugin: ImplicitGemm") and ONNX Runtime cannot run them at all.
#
# Prerequisites: docker/tensorrt_plugins/build_spconv_cpp.sh (cumm + spconv C++), CUDA, cmake.
#
# Usage (inside the container):  bash docker/tensorrt_plugins/build_plugin.sh
# Environment overrides:
#   PLUGINS_SRC          local plugin source tree (skips the clone), e.g.
#                        /path/to/autoware.universe/perception/autoware_tensorrt_plugins
#   PLUGINS_REPO / _REF  repository and ref to clone the plugin source from
#   TensorRT_INCLUDE_DIR headers to build against (must match the linked runtime)
#   INSTALL_DIR          where the .so lands (default /opt/plugins)
set -euo pipefail

PLUGINS_REPO="${PLUGINS_REPO:-https://github.com/autowarefoundation/autoware_universe.git}"
PLUGINS_REF="${PLUGINS_REF:-main}"
PLUGINS_SRC="${PLUGINS_SRC:-}"
INSTALL_DIR="${INSTALL_DIR:-/opt/plugins}"
BUILD_DIR="${BUILD_DIR:-/tmp/autoware_trt_plugins_build}"
CLONE_DIR="${CLONE_DIR:-/tmp/autoware_trt_plugins_src}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SO_NAME=libautoware_tensorrt_plugins.so

log() { echo "[build_plugin] $*"; }

command -v cmake >/dev/null || {
    log "ERROR: cmake is required"
    exit 1
}

# Build with the system toolchain, not pixi's: spconv and cumm install their headers into
# /usr/local/include, which the conda-sysroot compilers do not search.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
[ -x "$CC" ] && [ -x "$CXX" ] || {
    log "ERROR: system gcc/g++ not found ($CC, $CXX)"
    exit 1
}

# TensorRT lives in the project environment, so its interpreter is the one that can
# report the runtime's location and version.
if [ -z "${PYTHON_BIN:-}" ]; then
    for candidate in python3 /workspace/.pixi/envs/dev/bin/python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import tensorrt" 2>/dev/null; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
[ -n "${PYTHON_BIN:-}" ] || {
    log "ERROR: no python with TensorRT found; set PYTHON_BIN to the project interpreter"
    exit 1
}
log "Toolchain: CC=$CC CXX=$CXX  python=$PYTHON_BIN"

# --- TensorRT runtime and headers ------------------------------------------------------
# The headers MUST match the runtime's major.minor. Building against a different version
# (Ubuntu's apt libnvinfer-dev, say) while linking these libraries produces an
# IPluginCreatorV3One vtable mismatch that segfaults inside createPlugin during ONNX
# parsing — a failure that looks nothing like a version problem.
read -r TRT_ROOT TRT_VERSION < <(
    "$PYTHON_BIN" - <<'PY'
import tensorrt
print(tensorrt.__path__[0], tensorrt.__version__)
PY
)
TRT_MM="${TRT_VERSION%.*}"
TRT_MM="${TRT_MM%.*}"
log "TensorRT runtime ${TRT_VERSION} (headers must be ${TRT_MM}) at ${TRT_ROOT}"

read -r TRT_HINTS NVINFER NVONNXPARSER < <(
    "$PYTHON_BIN" - <<'PY'
import glob, os, site

dirs = []
for base in site.getsitepackages() + [site.getusersitepackages()]:
    if not base or not os.path.isdir(base):
        continue
    for rel in ("tensorrt", "tensorrt/lib", "tensorrt_libs", "nvidia/tensorrt", "nvidia/tensorrt/lib"):
        candidate = os.path.join(base, rel)
        if os.path.isdir(candidate):
            dirs.append(candidate)

def pick(pattern):
    for directory in dirs:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if matches:
            return matches[0]
    return "NOTFOUND"

print(";".join(dict.fromkeys(dirs)), pick("libnvinfer.so*"), pick("libnvonnxparser.so*"))
PY
)
log "nvinfer: ${NVINFER}"
log "nvonnxparser: ${NVONNXPARSER}"
[ "$NVINFER" != NOTFOUND ] && [ "$NVONNXPARSER" != NOTFOUND ] || {
    log "ERROR: TensorRT libraries not found in site-packages"
    exit 1
}

header_version() { # prints major.minor recorded in a header directory, or nothing
    local header="$1/NvInferVersion.h" major minor
    [ -f "$header" ] || return 0
    major="$(awk '/#define[ \t]+NV_TENSORRT_MAJOR/{print $3; exit}' "$header")"
    minor="$(awk '/#define[ \t]+NV_TENSORRT_MINOR/{print $3; exit}' "$header")"
    [[ $major =~ ^[0-9]+$ && $minor =~ ^[0-9]+$ ]] && echo "${major}.${minor}"
}

if [ -n "${TensorRT_INCLUDE_DIR:-}" ]; then
    found="$(header_version "$TensorRT_INCLUDE_DIR")"
    [ "$found" = "$TRT_MM" ] || {
        log "ERROR: TensorRT_INCLUDE_DIR headers are ${found:-unknown}, runtime is ${TRT_MM}"
        exit 1
    }
    log "Using TensorRT_INCLUDE_DIR=${TensorRT_INCLUDE_DIR} (${found})"
else
    # The pip wheels ship no headers, so take the matching public ones from TensorRT OSS.
    OSS_DIR="${TRT_OSS_HEADERS_DIR:-/tmp/tensorrt_oss_headers}"
    if [ "$(header_version "$OSS_DIR/include")" != "$TRT_MM" ]; then
        log "Fetching TensorRT ${TRT_MM} public headers from NVIDIA/TensorRT"
        rm -rf "$OSS_DIR"
        git clone -q -b "release/${TRT_MM}" --depth 1 --filter=blob:none --sparse \
            https://github.com/NVIDIA/TensorRT.git "$OSS_DIR"
        (cd "$OSS_DIR" && git sparse-checkout set include >/dev/null)
    fi
    found="$(header_version "$OSS_DIR/include")"
    [ "$found" = "$TRT_MM" ] || {
        log "ERROR: fetched headers are ${found:-unknown}, runtime is ${TRT_MM}"
        exit 1
    }
    TensorRT_INCLUDE_DIR="$OSS_DIR/include"
    log "Using TensorRT OSS headers ${TensorRT_INCLUDE_DIR} (${found})"
fi

# --- plugin source ---------------------------------------------------------------------
if [ -n "$PLUGINS_SRC" ]; then
    [ -f "$PLUGINS_SRC/src/implicit_gemm_plugin.cpp" ] || {
        log "ERROR: PLUGINS_SRC=$PLUGINS_SRC has no src/implicit_gemm_plugin.cpp"
        exit 1
    }
    PLUGIN_SRC_DIR="$(cd "$PLUGINS_SRC" && pwd)"
    log "Using local plugin source ${PLUGIN_SRC_DIR}"
else
    if [ ! -f "$CLONE_DIR/perception/autoware_tensorrt_plugins/src/implicit_gemm_plugin.cpp" ]; then
        log "Cloning ${PLUGINS_REPO} @ ${PLUGINS_REF}"
        rm -rf "$CLONE_DIR"
        git clone --depth 1 --branch "$PLUGINS_REF" --filter=blob:none --sparse \
            "$PLUGINS_REPO" "$CLONE_DIR"
        (cd "$CLONE_DIR" && git sparse-checkout set perception/autoware_tensorrt_plugins)
    fi
    PLUGIN_SRC_DIR="$CLONE_DIR/perception/autoware_tensorrt_plugins"
fi

# The exported graphs carry the `do_sort` attribute on GetIndicePairsImplicitGemm; a source
# tree without it produces a plugin that rejects them.
if grep -q '"do_sort"' \
    "$PLUGIN_SRC_DIR/src/get_indices_pairs_implicit_gemm_plugin_creator.cpp" 2>/dev/null; then
    log "Source exposes the do_sort attribute"
else
    log "WARNING: source does NOT expose the do_sort attribute — engine builds will reject"
    log "         graphs exported by autoware_ml.ops.spconv."
fi

# --- build ----------------------------------------------------------------------------
mkdir -p "$BUILD_DIR"
cp "$SCRIPT_DIR/CMakeLists.txt" "$BUILD_DIR/CMakeLists.txt"
cd "$BUILD_DIR"
rm -f CMakeCache.txt
cmake . \
    -DPLUGIN_SRC_DIR="$PLUGIN_SRC_DIR" \
    -DTensorRT_ROOT="$TRT_ROOT" \
    -DTensorRT_INCLUDE_DIR="$TensorRT_INCLUDE_DIR" \
    -DTensorRT_EXTRA_HINT_DIRS="$TRT_HINTS" \
    -DNVINFER_LIBRARY="$NVINFER" \
    -DNVONNXPARSER_LIBRARY="$NVONNXPARSER" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

[ -f "$BUILD_DIR/$SO_NAME" ] || {
    log "ERROR: build produced no $SO_NAME"
    exit 1
}
mkdir -p "$INSTALL_DIR"
cp -a "$BUILD_DIR/$SO_NAME" "$BUILD_DIR/libcuda_ops.so" "$INSTALL_DIR/"
chmod 755 "$INSTALL_DIR/$SO_NAME"
# The plugin links libcuda_ops.so, which lands beside it, so the install directory has to
# be on the loader's search path — otherwise dlopen fails on the dependency, not on the
# plugin itself.
if [ -w /etc ]; then
    echo "$INSTALL_DIR" >/etc/ld.so.conf.d/autoware-tensorrt-plugins.conf
    ldconfig
else
    log "WARNING: /etc is not writable; add ${INSTALL_DIR} to LD_LIBRARY_PATH yourself"
fi
log "Installed ${INSTALL_DIR}/${SO_NAME}"
log "Point deploy.tensorrt.plugin_libraries at it to build engines for plugin graphs."
