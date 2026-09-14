# Standalone build of Autoware's TensorRT plugins (no ament / autoware_cmake).
#
# Provides the custom operators the deployed sparse-convolution graphs bind against
# (`autoware::GetIndicePairsImplicitGemm`, `autoware::ImplicitGemm`, ...). Source comes
# from autoware_universe/perception/autoware_tensorrt_plugins; see build_plugin.sh.
#
# Required cache variables:
#   PLUGIN_SRC_DIR        the plugin source tree (holding src/ and include/)
#   TensorRT_ROOT         TensorRT location, e.g. the pip package directory
#   TensorRT_INCLUDE_DIR  headers whose version matches the TensorRT runtime being linked
cmake_minimum_required(VERSION 3.14)
project(autoware_tensorrt_plugins LANGUAGES CXX CUDA)

set(PLUGIN_SRC_DIR "" CACHE PATH "Plugin source tree (with src/ and include/)")
set(TensorRT_ROOT "" CACHE PATH "TensorRT root (e.g. pip site-packages/tensorrt)")
set(TensorRT_INCLUDE_DIR "" CACHE PATH "Directory containing NvInferRuntime.h")
set(TensorRT_EXTRA_HINT_DIRS "" CACHE STRING "Extra directories to search for TensorRT libraries")
set(NVINFER_LIBRARY "" CACHE FILEPATH "Explicit path to libnvinfer(.so*)")
set(NVONNXPARSER_LIBRARY "" CACHE FILEPATH "Explicit path to libnvonnxparser(.so*)")
# Keep in sync with TORCH_CUDA_ARCH_LIST in docker/Dockerfile. The trailing PTX entry lets
# the kernels run on newer architectures than the ones compiled here.
set(CUDA_ARCHS "80;86;87;89;90;120" CACHE STRING "CUDA architectures to compile kernels for")
set(CUDA_PTX_ARCH "120" CACHE STRING "Architecture to also embed as PTX (forward compatibility)")

if(NOT PLUGIN_SRC_DIR OR NOT EXISTS "${PLUGIN_SRC_DIR}/src")
  message(FATAL_ERROR "PLUGIN_SRC_DIR must point to a plugin source tree with src/.")
endif()
if(NOT TensorRT_ROOT OR NOT EXISTS "${TensorRT_ROOT}")
  message(FATAL_ERROR "TensorRT_ROOT must point to a TensorRT installation.")
endif()

add_compile_options(-Wno-deprecated-declarations)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# TensorRT. The pip wheels ship only versioned libraries (libnvinfer.so.10) and no headers,
# so both are resolved explicitly by build_plugin.sh and passed in; the searches below are
# fallbacks for other layouts.
if(NOT TensorRT_INCLUDE_DIR)
  find_path(TensorRT_INCLUDE_DIR
    NAMES NvInferRuntime.h NvInfer.h
    HINTS "${TensorRT_ROOT}" "${TensorRT_ROOT}/include" ${TensorRT_EXTRA_HINT_DIRS}
    PATH_SUFFIXES include
  )
endif()
if(NOT TensorRT_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT headers (NvInferRuntime.h) not found under ${TensorRT_ROOT}.")
endif()

set(TRT_HINT_DIRS "${TensorRT_ROOT}" "${TensorRT_ROOT}/lib" ${TensorRT_EXTRA_HINT_DIRS})
list(REMOVE_DUPLICATES TRT_HINT_DIRS)
if(NOT NVINFER_LIBRARY)
  find_library(NVINFER_LIBRARY NAMES nvinfer libnvinfer.so.10 HINTS ${TRT_HINT_DIRS})
endif()
if(NOT NVONNXPARSER_LIBRARY)
  find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser libnvonnxparser.so.10 HINTS ${TRT_HINT_DIRS})
endif()
if(NOT NVINFER_LIBRARY OR NOT NVONNXPARSER_LIBRARY)
  message(FATAL_ERROR
    "TensorRT libraries not found (NVINFER_LIBRARY=${NVINFER_LIBRARY}, "
    "NVONNXPARSER_LIBRARY=${NVONNXPARSER_LIBRARY}) under ${TRT_HINT_DIRS}."
  )
endif()

# spconv / cumm C++ packages: the plugin's sparse kernels are libspconv's, not the ones in
# the pip spconv wheel. build_spconv_cpp.sh installs them.
find_package(cumm QUIET)
find_package(spconv QUIET)
if(NOT cumm_FOUND OR NOT spconv_FOUND)
  message(FATAL_ERROR
    "cumm and spconv C++ packages are required (cumm_FOUND=${cumm_FOUND}, "
    "spconv_FOUND=${spconv_FOUND}). Run docker/tensorrt_plugins/build_spconv_cpp.sh first."
  )
endif()

message(STATUS "TensorRT includes: ${TensorRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer:  ${NVINFER_LIBRARY}")
message(STATUS "TensorRT onnx:     ${NVONNXPARSER_LIBRARY}")
message(STATUS "CUDA architectures: ${CUDA_ARCHS} (+PTX ${CUDA_PTX_ARCH})")

set(PLUGIN_CUDA_FLAGS "--expt-relaxed-constexpr" "--extended-lambda" "-diag-suppress=1675")
foreach(arch IN LISTS CUDA_ARCHS)
  list(APPEND PLUGIN_CUDA_FLAGS "-gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()
if(CUDA_PTX_ARCH)
  list(APPEND PLUGIN_CUDA_FLAGS
    "-gencode=arch=compute_${CUDA_PTX_ARCH},code=compute_${CUDA_PTX_ARCH}")
endif()

add_definitions(-DTV_CUDA)
include_directories(
  "${PLUGIN_SRC_DIR}/include"
  "${PLUGIN_SRC_DIR}/src"
  "${TensorRT_INCLUDE_DIR}"
)

# CUDA kernels backing the non-spconv plugins (BEV pooling, argsort, ...).
add_library(cuda_ops SHARED
  "${PLUGIN_SRC_DIR}/src/argsort_ops/argsort.cu"
  "${PLUGIN_SRC_DIR}/src/bev_ops/bev_pool_cuda.cu"
  "${PLUGIN_SRC_DIR}/src/scatter_ops/segment_csr.cu"
  "${PLUGIN_SRC_DIR}/src/unique_ops/unique.cu"
  "${PLUGIN_SRC_DIR}/src/multi_scale_deform_attn_ops/ms_deform_attn_kernel.cu"
  "${PLUGIN_SRC_DIR}/src/rotate_ops/rotate_kernel.cu"
  "${PLUGIN_SRC_DIR}/src/select_and_pad_ops/select_and_pad_kernel.cu"
)
set_target_properties(cuda_ops PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(cuda_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${PLUGIN_CUDA_FLAGS}>)

add_library(${PROJECT_NAME} SHARED
  "${PLUGIN_SRC_DIR}/src/argsort_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/argsort_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/quick_cumsum_cuda_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/quick_cumsum_cuda_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/get_indices_pairs_implicit_gemm_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/get_indices_pairs_implicit_gemm_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/get_indices_pairs_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/get_indices_pairs_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/implicit_gemm_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/implicit_gemm_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/indice_conv_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/indice_conv_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/multi_scale_deformable_attention_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/multi_scale_deformable_attention_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/rotate_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/rotate_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/segment_csr_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/segment_csr_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/select_and_pad_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/select_and_pad_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/unique_plugin.cpp"
  "${PLUGIN_SRC_DIR}/src/unique_plugin_creator.cpp"
  "${PLUGIN_SRC_DIR}/src/plugin_registration.cpp"
  "${PLUGIN_SRC_DIR}/src/plugin_utils.cpp"
)
target_compile_definitions(${PROJECT_NAME} PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${PLUGIN_CUDA_FLAGS}>)
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${NVINFER_LIBRARY}
  ${NVONNXPARSER_LIBRARY}
  CUDA::cudart
  cuda_ops
  spconv::spconv
)
