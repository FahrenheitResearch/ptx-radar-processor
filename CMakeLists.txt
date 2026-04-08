cmake_minimum_required(VERSION 3.24)
project(ptx-radar-processor LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Match the upstream PTX build so the processor keeps the same native SASS/PTX mix.
set(CMAKE_CUDA_ARCHITECTURES "86;89;120;120-virtual"
    CACHE STRING "CUDA archs to compile real SASS + PTX fallback" FORCE)

include(FetchContent)

find_package(BZip2 QUIET)
if(NOT BZip2_FOUND)
    FetchContent_Declare(
        bzip2_src
        URL      https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz
        URL_HASH SHA256=ab5a03176ee106d3f0fa90e381da478ddae405918153cca248e682cd0c4a2269
    )
    FetchContent_MakeAvailable(bzip2_src)
    add_library(bzip2 STATIC
        ${bzip2_src_SOURCE_DIR}/blocksort.c
        ${bzip2_src_SOURCE_DIR}/huffman.c
        ${bzip2_src_SOURCE_DIR}/crctable.c
        ${bzip2_src_SOURCE_DIR}/randtable.c
        ${bzip2_src_SOURCE_DIR}/compress.c
        ${bzip2_src_SOURCE_DIR}/decompress.c
        ${bzip2_src_SOURCE_DIR}/bzlib.c
    )
    target_include_directories(bzip2 PUBLIC ${bzip2_src_SOURCE_DIR})
    add_library(BZip2::BZip2 ALIAS bzip2)
endif()

find_package(CUDAToolkit REQUIRED)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/ptx-radar/src/cuda/ultra_kernels.ptx"
     ULTRA_PTX_CONTENT)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ptx-radar/src/cuda/ultra_kernels_ptx.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/ultra_kernels_ptx.h"
               @ONLY)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/ptx-radar/src/cuda/ultra_kernels.ptx")

add_executable(ptx-radar-processor
    src/main.cpp
    src/processor.cpp
    src/png_writer.cpp
    ptx-radar/src/nexrad/level2_parser.cpp
    ptx-radar/src/net/downloader.cpp
    ptx-radar/src/cuda/renderer.cu
    ptx-radar/src/cuda/preprocess.cu
    ptx-radar/src/cuda/gpu_pipeline.cu
)

target_include_directories(ptx-radar-processor PRIVATE
    src
    ptx-radar/src
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

if(WIN32)
    target_compile_definitions(ptx-radar-processor PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(ptx-radar-processor PRIVATE
    BZip2::BZip2
    CUDA::cudart
    CUDA::cuda_driver
)

if(WIN32)
    target_link_libraries(ptx-radar-processor PRIVATE winhttp ole32 windowscodecs)
else()
    find_package(CURL REQUIRED)
    target_link_libraries(ptx-radar-processor PRIVATE CURL::libcurl dl pthread)
endif()

set_target_properties(ptx-radar-processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(ptx-radar-processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xptxas=-v
    >
)
