#!/usr/bin/env bash
# Build and install the spconv / cumm C++ packages.
#
# The TensorRT sparse-convolution plugins link against libspconv, which is a different
# artifact from the pip `spconv` wheel: the wheel carries Python bindings and its own
# kernels, while the plugin needs the C++ package that `find_package(spconv)` resolves.
# Both come from autowarefoundation/spconv_cpp, are packaged as .deb by cpack, and are
# installed system-wide here.
#
# Usage (inside the container):  bash docker/tensorrt_plugins/build_spconv_cpp.sh
# Environment overrides: SPCONV_CPP_REPO, SPCONV_CPP_REF, SPCONV_CPP_DIR
set -euo pipefail

SPCONV_CPP_REPO="${SPCONV_CPP_REPO:-https://github.com/autowarefoundation/spconv_cpp.git}"
SPCONV_CPP_REF="${SPCONV_CPP_REF:-main}"
SPCONV_CPP_DIR="${SPCONV_CPP_DIR:-/opt/spconv_cpp}"

log() { echo "[build_spconv_cpp] $*"; }

for tool in cmake make cpack git; do
    command -v "$tool" >/dev/null || {
        log "ERROR: $tool is required"
        exit 1
    }
done

# Build with the system toolchain, not pixi's. cumm installs its headers into
# /usr/local/include, which the conda-sysroot compilers in the pixi environment do not
# search — spconv then fails to compile with "tensorview/cuda/device_ops.h: No such file".
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
[ -x "$CC" ] && [ -x "$CXX" ] || {
    log "ERROR: system gcc/g++ not found ($CC, $CXX)"
    exit 1
}
log "Toolchain: CC=$CC CXX=$CXX"

if [ ! -d "$SPCONV_CPP_DIR/.git" ]; then
    log "Cloning $SPCONV_CPP_REPO @ $SPCONV_CPP_REF into $SPCONV_CPP_DIR"
    rm -rf "$SPCONV_CPP_DIR"
    git clone --depth 1 --branch "$SPCONV_CPP_REF" "$SPCONV_CPP_REPO" "$SPCONV_CPP_DIR"
else
    log "Reusing existing checkout at $SPCONV_CPP_DIR"
fi

# cumm first: spconv's build depends on it.
for package in cumm spconv; do
    log "Building $package"
    mkdir -p "$SPCONV_CPP_DIR/$package/build-amd64"
    cd "$SPCONV_CPP_DIR/$package/build-amd64"
    cmake ..
    make -j"$(nproc)"
    cpack -G DEB
    deb="$(find "$SPCONV_CPP_DIR/$package/_packages" -name "${package}_*_amd64.deb" | head -1)"
    [ -n "$deb" ] || {
        log "ERROR: cpack produced no .deb for $package"
        exit 1
    }
    log "Installing $deb"
    apt-get update
    apt-get install -y --no-install-recommends "$deb"
done

ldconfig
log "Done. cumm and spconv C++ packages installed."
