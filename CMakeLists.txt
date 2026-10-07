cmake_minimum_required(VERSION 3.20)
project(dcvc_uf_int16_cpp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Imported Conda libraries add their prefix to RUNPATH. That prefix can carry
# an older libstdc++ than the compiler linked (notably condition_variable's
# GLIBCXX_3.4.30 symbol). Isolate the compiler runtime so resolving it first does
# not also replace Conda's curl/OpenSSL/Tcl libraries with system versions.
set(UF_CXX_RUNTIME_LIBRARY "" CACHE FILEPATH "Explicit compiler-compatible libstdc++.so.6 runtime")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(NOT UF_CXX_RUNTIME_LIBRARY)
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so.6
            OUTPUT_VARIABLE UF_CXX_RUNTIME_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY)
    endif()
    if(IS_ABSOLUTE "${UF_CXX_RUNTIME_LIBRARY}" AND EXISTS "${UF_CXX_RUNTIME_LIBRARY}")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/compiler-runtime")
        file(CREATE_LINK "${UF_CXX_RUNTIME_LIBRARY}"
             "${CMAKE_CURRENT_BINARY_DIR}/compiler-runtime/libstdc++.so.6"
             SYMBOLIC COPY_ON_ERROR)
        list(PREPEND CMAKE_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}/compiler-runtime")
    endif()
endif()

option(UF_BUILD_FP16 "Include the native UF FP16 CUDA inference backend" ON)
set(UF_FP16_SM "" CACHE STRING "GPU compute capability for the FP16 backend (for example 89)")
set(UF_CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass-uf" CACHE PATH "Separate CUTLASS dependency checkout")

# Capability values become nvcc architecture names and C++ integer literals.
# Canonicalize numeric input before architecture selection and SM90/SM100
# special cases; leading zeroes otherwise produce octal/invalid macro values
# or miss the architecture-specific compiler/toolkit gates. The regular
# variable shadows the cache entry, preserving the user's supplied spelling.
if(UF_FP16_SM MATCHES "^[0-9]+$")
    string(REGEX REPLACE "^0+" "" UF_FP16_SM "${UF_FP16_SM}")
    if(UF_FP16_SM STREQUAL "")
        set(UF_FP16_SM "0")
    endif()
endif()

# The installed Torch package enables CUDA during discovery, then resets
# CMAKE_CUDA_ARCHITECTURES to OFF for its legacy CUDA_NVCC_FLAGS machinery.
# Preserve integrated-CUDA target inputs across that package lookup. A supplied
# CMake value (including OFF) wins; CUDAARCHS initializes an unset value. The
# configured UF GPU is the default for INT16 as well as the FP16 target.
set(UF_CUDA_ARCHITECTURE_SELECTION_PRESENT FALSE)
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(UF_SELECTED_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    set(UF_CUDA_ARCHITECTURE_SELECTION_PRESENT TRUE)
elseif(NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(UF_SELECTED_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")
    set(UF_CUDA_ARCHITECTURE_SELECTION_PRESENT TRUE)
elseif(UF_FP16_SM MATCHES "^[0-9]+$" AND UF_FP16_SM GREATER_EQUAL 80)
    set(UF_SELECTED_CUDA_ARCHITECTURES "${UF_FP16_SM}")
    set(UF_CUDA_ARCHITECTURE_SELECTION_PRESENT TRUE)
endif()
if(UF_CUDA_ARCHITECTURE_SELECTION_PRESENT)
    set(CMAKE_CUDA_ARCHITECTURES "${UF_SELECTED_CUDA_ARCHITECTURES}")
endif()

find_package(OpenSSL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
if(NOT DEFINED CMAKE_CUDA_FLAGS)
    # CUDA can first be enabled inside Torch discovery. Capture its initial
    # user flags now so CUDAFLAGS/FLAGS_INIT are not mistaken for Torch flags.
    string(STRIP "$ENV{CUDAFLAGS} ${CMAKE_CUDA_FLAGS_INIT}" UF_CUDA_INITIAL_FLAGS)
    set(CMAKE_CUDA_FLAGS "${UF_CUDA_INITIAL_FLAGS}" CACHE STRING
        "Flags used by the CUDA compiler during all build types.")
endif()
set(UF_CUDA_FLAGS_BEFORE_TORCH "${CMAKE_CUDA_FLAGS}")
find_package(Torch REQUIRED)
# Torch appends legacy -gencode pairs to the global CUDA flags. A target's
# CUDA_ARCHITECTURES property does not override those flags, so they would
# also compile the single-SM FP16 kernels for unrelated architectures. Keep
# user flags and Torch's non-architecture additions, and scope the appended
# architecture groups to INT16 only when UF has no explicit selection.
string(LENGTH "${UF_CUDA_FLAGS_BEFORE_TORCH}" UF_CUDA_FLAGS_PREFIX_LENGTH)
string(SUBSTRING "${CMAKE_CUDA_FLAGS}" 0 ${UF_CUDA_FLAGS_PREFIX_LENGTH}
       UF_CUDA_FLAGS_PREFIX_AFTER_TORCH)
if(NOT UF_CUDA_FLAGS_PREFIX_AFTER_TORCH STREQUAL UF_CUDA_FLAGS_BEFORE_TORCH)
    message(FATAL_ERROR "Torch replaced existing CMAKE_CUDA_FLAGS; cannot isolate its architecture flags while preserving user flags")
endif()
string(SUBSTRING "${CMAKE_CUDA_FLAGS}" ${UF_CUDA_FLAGS_PREFIX_LENGTH} -1
       UF_CUDA_FLAGS_ADDED_BY_TORCH)
set(UF_TORCH_ARCH_FLAG_PATTERN "(^|[ \t])(-gencode|--generate-code)(=|[ \t]+)[^ \t]+")
string(REGEX MATCHALL "${UF_TORCH_ARCH_FLAG_PATTERN}" UF_TORCH_ARCH_FLAG_GROUPS
       "${UF_CUDA_FLAGS_ADDED_BY_TORCH}")
set(UF_TORCH_DEFAULT_ARCH_OPTIONS "")
foreach(UF_TORCH_ARCH_FLAG_GROUP IN LISTS UF_TORCH_ARCH_FLAG_GROUPS)
    string(STRIP "${UF_TORCH_ARCH_FLAG_GROUP}" UF_TORCH_ARCH_FLAG_GROUP)
    # SHELL keeps each -gencode and its argument together during de-duplication.
    list(APPEND UF_TORCH_DEFAULT_ARCH_OPTIONS "SHELL:${UF_TORCH_ARCH_FLAG_GROUP}")
endforeach()
string(REGEX REPLACE "${UF_TORCH_ARCH_FLAG_PATTERN}" " "
       UF_CUDA_FLAGS_ADDED_BY_TORCH "${UF_CUDA_FLAGS_ADDED_BY_TORCH}")
set(CMAKE_CUDA_FLAGS "${UF_CUDA_FLAGS_BEFORE_TORCH}${UF_CUDA_FLAGS_ADDED_BY_TORCH}")
if(UF_CUDA_ARCHITECTURE_SELECTION_PRESENT)
    set(CMAKE_CUDA_ARCHITECTURES "${UF_SELECTED_CUDA_ARCHITECTURES}")
endif()
find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# torch.utils.cpp_extension.BuildExtension supplies these to both reference
# CUDA extensions in addition to their setup.py flags. In particular, CUTLASS
# also requires relaxed constexpr when nvcc instantiates its device templates.
# Keep them local to CUDA compilation rather than passing nvcc flags to C++.
set(UF_TORCH_NVCC_OPTIONS
    -D__CUDA_NO_HALF_OPERATORS__
    -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__
    --expt-relaxed-constexpr)

add_library(uf_archive STATIC
    src/archive.cpp
    src/bitstream.cpp
    src/bitstream_output.cpp
    src/image_archive.cpp
    src/image_pixels.cpp
    src/image_tiff.cpp
    src/image_avif.cpp
    src/media.cpp
    src/jetson.cpp
    src/process.cpp
    src/events.cpp
    src/dashboard.cpp
    src/frame_buffer.cpp
    src/reference_paths.cpp
    src/python_json.cpp
    src/python_path.cpp
    src/utf8.cpp
)
target_include_directories(uf_archive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(uf_archive PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(uf_archive PUBLIC OpenSSL::Crypto nlohmann_json::nlohmann_json Threads::Threads)

find_package(ZLIB QUIET)

option(UF_BUILD_NATIVE_IMAGES "Use installed native JPEG, PNG, WebP, TIFF and AVIF image decoders" ON)
if(UF_BUILD_NATIVE_IMAGES)
    find_package(JPEG QUIET)
    if(JPEG_FOUND)
        target_compile_definitions(uf_archive PRIVATE UF_HAS_JPEG=1)
        target_link_libraries(uf_archive PRIVATE JPEG::JPEG)
    endif()
    find_package(PNG QUIET)
    if(PNG_FOUND)
        target_compile_definitions(uf_archive PRIVATE UF_HAS_PNG=1)
        target_link_libraries(uf_archive PRIVATE PNG::PNG)
    endif()
    find_path(UF_WEBP_INCLUDE_DIR NAMES webp/decode.h)
    find_library(UF_WEBP_LIBRARY NAMES webp)
    find_library(UF_WEBP_DEMUX_LIBRARY NAMES webpdemux)
    if(UF_WEBP_INCLUDE_DIR AND EXISTS "${UF_WEBP_INCLUDE_DIR}/webp/demux.h" AND
       UF_WEBP_LIBRARY AND UF_WEBP_DEMUX_LIBRARY)
        target_compile_definitions(uf_archive PRIVATE UF_HAS_WEBP=1)
        target_include_directories(uf_archive PRIVATE ${UF_WEBP_INCLUDE_DIR})
        target_link_libraries(uf_archive PRIVATE ${UF_WEBP_DEMUX_LIBRARY} ${UF_WEBP_LIBRARY})
    endif()
    # Per-handle diagnostics avoid global libtiff error-handler changes.
    find_package(TIFF 4.5 QUIET)
    if(TIFF_FOUND)
        target_compile_definitions(uf_archive PRIVATE UF_HAS_TIFF=1)
        target_link_libraries(uf_archive PRIVATE TIFF::TIFF)
        # Pillow converts TIFF LAB mode through LittleCMS, using built-in
        # Lab v2 and sRGB profiles rather than the TIFF's embedded profile.
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(UF_LCMS2 QUIET IMPORTED_TARGET lcms2>=2.0)
        endif()
        if(TARGET PkgConfig::UF_LCMS2)
            target_compile_definitions(uf_archive PRIVATE UF_HAS_LCMS2=1)
            target_link_libraries(uf_archive PRIVATE PkgConfig::UF_LCMS2)
        else()
            find_path(UF_LCMS2_INCLUDE_DIR NAMES lcms2.h)
            find_library(UF_LCMS2_LIBRARY NAMES lcms2)
            if(UF_LCMS2_INCLUDE_DIR AND UF_LCMS2_LIBRARY)
                target_compile_definitions(uf_archive PRIVATE UF_HAS_LCMS2=1)
                target_include_directories(uf_archive PRIVATE ${UF_LCMS2_INCLUDE_DIR})
                target_link_libraries(uf_archive PRIVATE ${UF_LCMS2_LIBRARY})
            else()
                message(STATUS "Native TIFF CIELAB disabled: LittleCMS2 development files unavailable")
            endif()
        endif()
    endif()
    find_package(libavif 1.0 CONFIG QUIET)
    if(TARGET avif)
        target_compile_definitions(uf_archive PRIVATE UF_HAS_AVIF=1)
        target_link_libraries(uf_archive PRIVATE avif)
    elseif(TARGET avif_static)
        target_compile_definitions(uf_archive PRIVATE UF_HAS_AVIF=1)
        target_link_libraries(uf_archive PRIVATE avif_static)
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(UF_LIBAVIF QUIET IMPORTED_TARGET libavif>=1.0)
            if(TARGET PkgConfig::UF_LIBAVIF)
                target_compile_definitions(uf_archive PRIVATE UF_HAS_AVIF=1)
                target_link_libraries(uf_archive PRIVATE PkgConfig::UF_LIBAVIF)
            endif()
        endif()
        if(NOT TARGET PkgConfig::UF_LIBAVIF)
            find_path(UF_AVIF_INCLUDE_DIR NAMES avif/avif.h)
            find_library(UF_AVIF_LIBRARY NAMES avif)
            if(UF_AVIF_INCLUDE_DIR AND UF_AVIF_LIBRARY)
                target_compile_definitions(uf_archive PRIVATE UF_HAS_AVIF=1)
                target_include_directories(uf_archive PRIVATE ${UF_AVIF_INCLUDE_DIR})
                target_link_libraries(uf_archive PRIVATE ${UF_AVIF_LIBRARY})
            endif()
        endif()
    endif()
endif()

# FindPNG may also discover ZLIB. Apply text-inflation support after the
# image dependencies have been resolved so native PNG always reuses it.
if(ZLIB_FOUND)
    target_compile_definitions(uf_archive PRIVATE UF_HAS_ZLIB=1)
    target_link_libraries(uf_archive PRIVATE ZLIB::ZLIB)
endif()

add_library(uf_entropy STATIC src/entropy.cpp)
target_include_directories(uf_entropy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(uf_random STATIC src/pareto_random.cpp src/python_integer.cpp src/python_numeric.cpp
    src/python_fraction.cpp)
target_include_directories(uf_random PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(uf_random PUBLIC OpenSSL::Crypto)
target_link_libraries(uf_archive PUBLIC uf_random)

add_library(uf_cli STATIC src/cli.cpp)
target_include_directories(uf_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(uf_cli PUBLIC uf_random)

add_library(uf_frame_utils STATIC src/transforms.cpp src/uf_splines.cpp src/video_io.cpp)
target_include_directories(uf_frame_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(uf_frame_utils PUBLIC uf_archive ${TORCH_LIBRARIES})
target_compile_options(uf_frame_utils PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")

add_library(uf_int16 STATIC
    src/int16_ops.cpp src/prepared.cpp src/prepare.cpp
    src/torch_pickle.cpp src/torch_pickle_structure.cpp src/torch_pickle_reader.cpp
    src/checkpoint_storage_copy.hpp
    src/inference_environment.cpp src/inference_environment.hpp
    src/model_schema.cpp src/layers.cpp src/integer_model.cpp src/integer_codec.cpp)
target_include_directories(uf_int16 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(uf_int16 PUBLIC uf_archive uf_entropy uf_random ${TORCH_LIBRARIES} OpenSSL::Crypto nlohmann_json::nlohmann_json)

# The supported port is UF. Earlier family work is retained as opt-in source
# and does not participate in the default executable or library dependency graph.
option(UF_BUILD_DCVC2021 "Include the original DCVC family model and stream library" OFF)
option(UF_BUILD_DCVC_TCM "Include the DCVC-TCM image/video models and stream library" OFF)
option(UF_BUILD_DCVC_HEM "Include the DCVC-HEM image/video models and stream library" OFF)
option(UF_BUILD_DCVC_DC "Include the DCVC-DC image/video models, layers, entropy and stream library" OFF)
if(UF_BUILD_DCVC2021 OR UF_BUILD_DCVC_TCM OR UF_BUILD_DCVC_HEM OR UF_BUILD_DCVC_DC)
    add_library(uf_dcvc_legacy STATIC
        src/dcvc2021_layers.cpp
        src/dcvc2021_entropy.cpp
        src/dcvc2021_stream.cpp
        src/dcvc2021_image_layers.cpp
        src/dcvc2021_image_entropy.cpp
    )
    target_include_directories(uf_dcvc_legacy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(uf_dcvc_legacy PUBLIC uf_int16 ${TORCH_LIBRARIES} Threads::Threads)
    target_compile_options(uf_dcvc_legacy PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")
    if(UF_BUILD_DCVC_TCM OR UF_BUILD_DCVC_HEM OR UF_BUILD_DCVC_DC)
        target_sources(uf_dcvc_legacy PRIVATE src/dcvc_tcm_msssim.cpp)
    endif()
endif()
if(UF_BUILD_DCVC2021)
    add_library(uf_dcvc2021 STATIC
        src/dcvc2021_model.cpp
        src/dcvc2021_image_model.cpp
        src/dcvc2021_download.cpp
    )
    target_include_directories(uf_dcvc2021 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(uf_dcvc2021 PUBLIC uf_dcvc_legacy uf_cli ${TORCH_LIBRARIES} Threads::Threads)
    target_compile_options(uf_dcvc2021 PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")
endif()
if(UF_BUILD_DCVC_TCM)
    add_library(uf_dcvc_tcm STATIC
        src/dcvc_tcm_layers.cpp
        src/dcvc_tcm_entropy.cpp
        src/dcvc_tcm_image_layers.cpp
        src/dcvc_tcm_image_entropy.cpp
        src/dcvc_tcm_image_model.cpp
        src/dcvc_tcm_model.cpp
        src/dcvc_tcm_stream.cpp
    )
    target_include_directories(uf_dcvc_tcm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(uf_dcvc_tcm PUBLIC uf_dcvc_legacy ${TORCH_LIBRARIES} Threads::Threads)
    target_compile_options(uf_dcvc_tcm PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")
endif()

if(UF_BUILD_DCVC_HEM)
    add_library(uf_dcvc_hem STATIC
        src/dcvc_hem_layers.cpp
        src/dcvc_hem_entropy.cpp
        src/dcvc_hem_common.cpp
        src/dcvc_hem_stream.cpp
        src/dcvc_hem_image.cpp
        src/dcvc_hem_video.cpp
        src/dcvc_hem_download.cpp
    )
    target_include_directories(uf_dcvc_hem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(uf_dcvc_hem PUBLIC uf_dcvc_legacy uf_cli ${TORCH_LIBRARIES} Threads::Threads)
    target_compile_options(uf_dcvc_hem PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")
endif()

if(UF_BUILD_DCVC_DC)
    add_library(uf_dcvc_dc STATIC
        src/dcvc_dc_layers.cpp
        src/dcvc_dc_entropy.cpp
        src/dcvc_dc_common.cpp
        src/dcvc_dc_stream.cpp
        src/dcvc_dc_image.cpp
        src/dcvc_dc_video.cpp
        src/dcvc_dc_splines.cpp
        src/dcvc_dc_transforms.cpp
        src/dcvc_dc_media.cpp
        src/dcvc_dc_download.cpp
    )
    target_include_directories(uf_dcvc_dc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(uf_dcvc_dc PUBLIC uf_dcvc_legacy uf_cli ${TORCH_LIBRARIES} Threads::Threads)
    target_compile_options(uf_dcvc_dc PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")
endif()

option(UF_BUILD_EVALUATION "Include native benchmarks, quality metrics and BD-rate comparison" ON)
option(UF_BUILD_RD_PLOTS "Use installed Cairo development files for RD curve PNGs" ON)
if(UF_BUILD_EVALUATION)
    add_library(uf_evaluation STATIC
        src/evaluation_json.cpp
        src/evaluation_metrics.cpp
        src/bd_metrics.cpp
        src/bd_compare.cpp
        src/rd_plot.cpp
        src/evaluation_models.cpp
        src/evaluation_workers.cpp
        src/evaluation_cli.cpp
        src/evaluation_cuda_policy.cpp
    )
    target_include_directories(uf_evaluation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(uf_evaluation PUBLIC uf_archive uf_frame_utils uf_cli uf_random ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
    target_link_libraries(uf_evaluation PRIVATE ${CMAKE_DL_LIBS})
    target_compile_options(uf_evaluation PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")
    if(UF_BUILD_DCVC2021)
        target_sources(uf_evaluation PRIVATE src/dcvc2021_evaluation.cpp src/dcvc2021_evaluation_metrics.cpp)
        target_compile_definitions(uf_evaluation PRIVATE UF_HAS_DCVC2021=1)
        target_link_libraries(uf_evaluation PUBLIC uf_dcvc2021)
    endif()
    if(UF_BUILD_DCVC_TCM)
        target_sources(uf_evaluation PRIVATE src/dcvc_tcm_evaluation.cpp)
        target_compile_definitions(uf_evaluation PRIVATE UF_HAS_DCVC_TCM=1)
        target_link_libraries(uf_evaluation PUBLIC uf_dcvc_tcm)
    endif()
    if(UF_BUILD_DCVC_HEM)
        target_sources(uf_evaluation PRIVATE src/dcvc_hem_evaluation.cpp)
        target_compile_definitions(uf_evaluation PRIVATE UF_HAS_DCVC_HEM=1)
        target_link_libraries(uf_evaluation PUBLIC uf_dcvc_hem)
    endif()
    if(UF_BUILD_DCVC_DC)
        target_sources(uf_evaluation PRIVATE src/dcvc_dc_evaluation.cpp src/dcvc_dc_metrics.cpp)
        target_compile_definitions(uf_evaluation PRIVATE UF_HAS_DCVC_DC=1)
        target_link_libraries(uf_evaluation PUBLIC uf_dcvc_dc)
    endif()
    if(UF_BUILD_RD_PLOTS)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(UF_CAIRO QUIET IMPORTED_TARGET cairo)
        endif()
        if(TARGET PkgConfig::UF_CAIRO)
            target_compile_definitions(uf_evaluation PRIVATE UF_HAS_CAIRO=1)
            target_link_libraries(uf_evaluation PRIVATE PkgConfig::UF_CAIRO)
        else()
            find_path(UF_CAIRO_INCLUDE_DIR NAMES cairo.h PATH_SUFFIXES cairo)
            find_library(UF_CAIRO_LIBRARY NAMES cairo)
            if(UF_CAIRO_INCLUDE_DIR AND UF_CAIRO_LIBRARY)
                target_compile_definitions(uf_evaluation PRIVATE UF_HAS_CAIRO=1)
                target_include_directories(uf_evaluation PRIVATE ${UF_CAIRO_INCLUDE_DIR})
                target_link_libraries(uf_evaluation PRIVATE ${UF_CAIRO_LIBRARY})
            else()
                message(STATUS "RD PNG plotting disabled: Cairo development files unavailable")
            endif()
        endif()
    endif()
endif()

option(UF_BUILD_TRAINING "Include the differentiable LibTorch training components" ON)
if(UF_BUILD_TRAINING)
    add_library(uf_training STATIC
        src/training_layers.cpp
        src/training_recompute.cpp
        src/training_common.cpp
        src/training_adamw.cpp
        src/training_checkpoint.cpp
        src/training_image_model.cpp
        src/training_ld_model.cpp
        src/training_ht_model.cpp
        src/training_video_model.cpp
        src/training_epoch.cpp
        src/training_loader.cpp
        src/training_loader_process.cpp
        src/training_distributed.cpp
        src/training_workflow.cpp
        src/training_cli.cpp
        src/training_datasets.cpp
        src/training_schedules.cpp
    )
    target_include_directories(uf_training PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(uf_training PUBLIC uf_int16 uf_archive uf_cli uf_random ${TORCH_LIBRARIES} Threads::Threads)
    # Python scalar expressions round at each operation; do not contract
    # AdamW's host-side decay/bias arithmetic into multiply-add instructions.
    target_compile_options(uf_training PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")
    # Optimizer capture checks use the runtime API and do not require nvcc.
    if(TARGET torch_cuda)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            target_compile_definitions(uf_training PRIVATE UF_HAS_TRAINING_CUDA=1)
            target_link_libraries(uf_training PRIVATE CUDA::cudart)
        endif()
    endif()
    option(UF_BUILD_TRAINING_NCCL "Use NCCL-enabled LibTorch for native distributed training" OFF)
    if(UF_BUILD_TRAINING_NCCL)
        if(NOT TARGET torch_cuda)
            message(FATAL_ERROR "UF_BUILD_TRAINING_NCCL requires CUDA/NCCL-enabled LibTorch")
        endif()
        find_path(UF_NCCL_INCLUDE_DIR NAMES nccl.h)
        find_library(UF_NCCL_LIBRARY NAMES nccl)
        if(NOT UF_NCCL_INCLUDE_DIR OR NOT UF_NCCL_LIBRARY)
            message(FATAL_ERROR "Set UF_NCCL_INCLUDE_DIR and UF_NCCL_LIBRARY to the NCCL installation matching LibTorch")
        endif()
        target_compile_definitions(uf_training PRIVATE UF_HAS_TRAINING_NCCL=1 USE_C10D_NCCL=1)
        target_include_directories(uf_training PRIVATE ${UF_NCCL_INCLUDE_DIR})
        target_link_libraries(uf_training PRIVATE torch_cuda ${UF_NCCL_LIBRARY})
    endif()
endif()

add_library(uf_fp16 STATIC src/fp16_model.cpp src/fp16_codec.cpp)
target_include_directories(uf_fp16 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(uf_fp16 PUBLIC uf_int16 uf_frame_utils ${TORCH_LIBRARIES} OpenSSL::Crypto)
if(UF_BUILD_EVALUATION)
    target_link_libraries(uf_evaluation PUBLIC uf_fp16)
endif()

include(CheckLanguage)
check_language(CUDA)
# nvcc alone cannot supply LibTorch's CUDA stream/guard implementation. Match
# the backend gates used by training/setup so a CPU LibTorch SDK keeps the
# INT16 reference path even when a CUDA toolkit is installed on the host.
if(CMAKE_CUDA_COMPILER AND TARGET torch_cuda)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(uf_cuda STATIC src/cuda/conv.cu src/cuda/elementwise.cu)
    target_include_directories(uf_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(uf_cuda PUBLIC ${TORCH_LIBRARIES} CUDA::cudart)
    target_compile_options(uf_cuda PRIVATE
        "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${UF_TORCH_NVCC_OPTIONS}>")
    if(NOT UF_CUDA_ARCHITECTURE_SELECTION_PRESENT AND UF_TORCH_DEFAULT_ARCH_OPTIONS)
        target_compile_options(uf_cuda PRIVATE
            "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${UF_TORCH_DEFAULT_ARCH_OPTIONS}>")
    endif()
    set_target_properties(uf_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(uf_int16 PRIVATE UF_HAS_CUDA=1)
    target_link_libraries(uf_int16 PUBLIC uf_cuda)

    if(UF_BUILD_FP16 AND NOT UF_FP16_SM STREQUAL "")
        if(NOT UF_FP16_SM MATCHES "^[0-9]+$" OR UF_FP16_SM LESS 80)
            message(FATAL_ERROR "UF_FP16_SM must be a numeric compute capability of at least 80")
        endif()
        # The retained inference sources and optimization flags use NVIDIA's
        # CUDA extension interface. Their extended launch config/attributes
        # are needed even when programmatic serialization is disabled on SM80.
        if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
            message(FATAL_ERROR "UF FP16 native inference requires NVIDIA nvcc; select it as CMAKE_CUDA_COMPILER or set UF_BUILD_FP16=OFF")
        endif()
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8 OR
           CUDAToolkit_VERSION VERSION_LESS 11.8)
            message(FATAL_ERROR "UF FP16 native inference requires CUDA compiler and toolkit 11.8 or newer for cudaLaunchKernelEx and its launch attributes")
        endif()
        # These architecture-specific targets use the same 'a' code generation
        # as inference/setup.py. CUTLASS's retained architecture table requires
        # CUDA 12.0 for 90a and CUDA 12.8 for 100a.
        if(UF_FP16_SM STREQUAL "90" AND
           (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0 OR CUDAToolkit_VERSION VERSION_LESS 12.0))
            message(FATAL_ERROR "UF_FP16_SM=90 requires CUDA compiler and toolkit 12.0 or newer for compute_90a/sm_90a")
        elseif(UF_FP16_SM STREQUAL "100" AND
               (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8 OR CUDAToolkit_VERSION VERSION_LESS 12.8))
            message(FATAL_ERROR "UF_FP16_SM=100 requires CUDA compiler and toolkit 12.8 or newer for compute_100a/sm_100a")
        endif()
        if(NOT EXISTS "${UF_CUTLASS_ROOT}/include/cutlass/cutlass.h")
            message(FATAL_ERROR "UF_CUTLASS_ROOT must point to the separate CUTLASS checkout")
        endif()
        file(GLOB_RECURSE UF_FP16_NATIVE_SOURCES CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/src/fp16/native/*.cpp"
             "${CMAKE_CURRENT_SOURCE_DIR}/src/fp16/native/*.cu")
        add_library(uf_fp16_native STATIC ${UF_FP16_NATIVE_SOURCES})
        target_include_directories(uf_fp16_native PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/fp16/native
            ${UF_CUTLASS_ROOT}/include
            ${UF_CUTLASS_ROOT}/tools/util/include)
        target_link_libraries(uf_fp16_native PUBLIC ${TORCH_LIBRARIES} CUDA::cudart Threads::Threads)
        target_compile_definitions(uf_fp16_native PRIVATE CURRENT_DEVICE_SM=${UF_FP16_SM})
        target_compile_options(uf_fp16_native PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3;-Wno-deprecated-declarations>"
            "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${UF_TORCH_NVCC_OPTIONS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization>")
        set_target_properties(uf_fp16_native PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        if(UF_FP16_SM STREQUAL "90" OR UF_FP16_SM STREQUAL "100")
            set_target_properties(uf_fp16_native PROPERTIES CUDA_ARCHITECTURES OFF)
            target_compile_options(uf_fp16_native PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${UF_FP16_SM}a,code=sm_${UF_FP16_SM}a>)
            target_compile_definitions(uf_fp16_native PRIVATE CUTLASS_ENABLE_GDC_FOR_SM${UF_FP16_SM}=1)
        else()
            set_target_properties(uf_fp16_native PROPERTIES CUDA_ARCHITECTURES ${UF_FP16_SM})
        endif()
        target_include_directories(uf_fp16 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/fp16/native)
        target_compile_definitions(uf_fp16 PRIVATE UF_HAS_FP16=1 UF_FP16_SM=${UF_FP16_SM})
        target_link_libraries(uf_fp16 PUBLIC uf_fp16_native)
        if(UF_BUILD_EVALUATION)
            target_compile_definitions(uf_evaluation PRIVATE UF_HAS_EVALUATION_CUDA=1)
            target_link_libraries(uf_evaluation PRIVATE CUDA::cudart)
        endif()
    elseif(UF_BUILD_FP16)
        message(STATUS "FP16 native inference disabled: set UF_FP16_SM for the intended GPU")
    endif()
elseif(NOT TARGET torch_cuda)
    message(STATUS "UF CUDA inference disabled: selected LibTorch has no CUDA backend; INT16 uses its CPU reference path")
elseif(UF_BUILD_FP16)
    message(STATUS "FP16 native inference disabled: CUDA compiler unavailable")
endif()

add_library(uf_codec STATIC src/codec.cpp src/frame_source.cpp src/runtime_environment.cpp)
target_include_directories(uf_codec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                   PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(uf_codec PUBLIC uf_int16 uf_fp16)
# NumPy's display conversion uses separate Float32 ufunc operations.
target_compile_options(uf_codec PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")

add_library(uf_workflow STATIC src/workflow.cpp src/workers.cpp src/image_workers.cpp src/decode_workers.cpp src/remote.cpp src/stream_pipeline.cpp src/image_workflow.cpp src/shared_work.cpp src/presets.cpp src/pareto.cpp src/pareto_state.cpp src/lifecycle.cpp src/viewer.cpp)
target_include_directories(uf_workflow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(uf_workflow PUBLIC uf_codec uf_archive uf_random Threads::Threads)
find_package(CURL 7.85 REQUIRED)
target_link_libraries(uf_workflow PRIVATE CURL::libcurl)
# Preserve the scalar operation boundaries in display-resize coefficients.
target_compile_options(uf_workflow PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>")

option(UF_BUILD_VIEWER "Include the native Tcl/Tk archive viewer" ON)
if(UF_BUILD_VIEWER)
    find_package(TCL QUIET)
    if(TCL_FOUND AND TK_LIBRARY AND TK_INCLUDE_PATH)
        target_compile_definitions(uf_workflow PRIVATE UF_HAS_TK=1)
        target_include_directories(uf_workflow PRIVATE ${TCL_INCLUDE_PATH} ${TK_INCLUDE_PATH})
        target_link_libraries(uf_workflow PRIVATE ${TK_LIBRARY} ${TCL_LIBRARY})
    else()
        message(STATUS "Native UF viewer disabled: Tcl/Tk development files unavailable")
    endif()
endif()

set(UF_PARETO_BENCHMARK "" CACHE FILEPATH "Optional generated UF HT-L 144p Pareto report for target-PSNR selection")
if(UF_PARETO_BENCHMARK)
    if(NOT EXISTS "${UF_PARETO_BENCHMARK}")
        message(FATAL_ERROR "UF_PARETO_BENCHMARK must name an existing generated Pareto report")
    endif()
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/benchmarks")
    configure_file("${UF_PARETO_BENCHMARK}"
        "${CMAKE_CURRENT_BINARY_DIR}/benchmarks/uf-int16-htl-144p-pareto.json" COPYONLY)
else()
    message(STATUS "No Pareto report supplied; generate one before using --target-psnr")
endif()

# Embed the configured source/dependency paths for resource discovery and the
# passive setup report.
# Escape C++ string literals before generating the header, including paths
# containing quotes, backslashes, or whitespace. No dependency is fetched.
function(uf_setup_escape_cpp output value)
    string(REPLACE "\\" "\\\\" escaped "${value}")
    string(REPLACE "\"" "\\\"" escaped "${escaped}")
    string(REPLACE "\n" "\\n" escaped "${escaped}")
    string(REPLACE "\r" "\\r" escaped "${escaped}")
    string(REPLACE "\t" "\\t" escaped "${escaped}")
    set(${output} "${escaped}" PARENT_SCOPE)
endfunction()
get_filename_component(UF_SETUP_CUTLASS_ROOT "${UF_CUTLASS_ROOT}" ABSOLUTE
                       BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
uf_setup_escape_cpp(UF_SETUP_SOURCE_ROOT_ESCAPED "${CMAKE_CURRENT_SOURCE_DIR}")
uf_setup_escape_cpp(UF_SETUP_CUTLASS_ROOT_ESCAPED "${UF_SETUP_CUTLASS_ROOT}")
uf_setup_escape_cpp(UF_SETUP_COMPILER_ESCAPED "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
# Wheel metadata carries vendor/version suffixes absent from torch/version.h.
# Read literal values only; this never imports Python or executes version.py.
# A standalone LibTorch SDK falls back to its header and linked CUDA hooks.
set(UF_TORCH_PACKAGE_VERSION "")
set(UF_TORCH_CUDA_VERSION "")
set(UF_TORCH_CUDA_VERSION_KNOWN false)
if(EXISTS "${TORCH_INSTALL_PREFIX}/version.py")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                 "${TORCH_INSTALL_PREFIX}/version.py")
    file(STRINGS "${TORCH_INSTALL_PREFIX}/version.py" UF_TORCH_VERSION_LINES
         REGEX "^(__version__|cuda)")
    foreach(UF_TORCH_VERSION_LINE IN LISTS UF_TORCH_VERSION_LINES)
        if(UF_TORCH_VERSION_LINE MATCHES [=[^__version__ *= *['"]([^'"]+)['"]]=])
            set(UF_TORCH_PACKAGE_VERSION "${CMAKE_MATCH_1}")
        elseif(UF_TORCH_VERSION_LINE MATCHES [=[^cuda( *:[^=]*)? *= *['"]([^'"]+)['"]]=])
            set(UF_TORCH_CUDA_VERSION "${CMAKE_MATCH_2}")
            set(UF_TORCH_CUDA_VERSION_KNOWN true)
        elseif(UF_TORCH_VERSION_LINE MATCHES [=[^cuda( *:[^=]*)? *= *None([ ]|$)]=])
            set(UF_TORCH_CUDA_VERSION_KNOWN true)
        endif()
    endforeach()
endif()
uf_setup_escape_cpp(UF_TORCH_PACKAGE_VERSION_ESCAPED "${UF_TORCH_PACKAGE_VERSION}")
uf_setup_escape_cpp(UF_TORCH_CUDA_VERSION_ESCAPED "${UF_TORCH_CUDA_VERSION}")
configure_file(include/uf/setup_build_config.hpp.in
               "${CMAKE_CURRENT_BINARY_DIR}/generated/uf/setup_build_config.hpp" @ONLY)

add_library(uf_setup STATIC src/setup.cpp)
target_include_directories(uf_setup PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                   PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(uf_setup PUBLIC uf_fp16 uf_int16 uf_archive ${TORCH_LIBRARIES})
# Device properties require runtime headers, but no CUDA compiler or tensors.
if(TARGET torch_cuda AND NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit QUIET)
endif()
if(TARGET CUDA::cudart)
    target_compile_definitions(uf_setup PRIVATE UF_HAS_SETUP_CUDA=1)
    target_link_libraries(uf_setup PRIVATE CUDA::cudart)
    target_compile_definitions(uf_codec PRIVATE UF_HAS_ENVIRONMENT_CUDA=1)
    target_link_libraries(uf_codec PRIVATE CUDA::cudart)
endif()

add_library(uf_int16_validation STATIC src/int16_validation.cpp)
target_include_directories(uf_int16_validation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(uf_int16_validation PUBLIC uf_int16 uf_cli ${TORCH_LIBRARIES} OpenSSL::Crypto)
if(TARGET torch_cuda AND TARGET CUDA::cudart)
    target_compile_definitions(uf_int16_validation PRIVATE UF_HAS_INT16_VALIDATION_CUDA=1)
    target_link_libraries(uf_int16_validation PRIVATE CUDA::cudart)
endif()

add_executable(dcvc-uf src/main.cpp)
target_link_libraries(dcvc-uf PRIVATE uf_cli uf_setup uf_int16_validation uf_workflow uf_int16 uf_archive uf_entropy)
if(UF_BUILD_DCVC2021)
    target_compile_definitions(dcvc-uf PRIVATE UF_HAS_DCVC2021=1)
    target_link_libraries(dcvc-uf PRIVATE uf_dcvc2021)
endif()
if(UF_BUILD_DCVC_TCM)
    target_compile_definitions(dcvc-uf PRIVATE UF_HAS_DCVC_TCM=1)
    target_link_libraries(dcvc-uf PRIVATE uf_dcvc_tcm)
endif()
if(UF_BUILD_DCVC_HEM)
    target_compile_definitions(dcvc-uf PRIVATE UF_HAS_DCVC_HEM=1)
    target_link_libraries(dcvc-uf PRIVATE uf_dcvc_hem)
endif()
if(UF_BUILD_DCVC_DC)
    target_compile_definitions(dcvc-uf PRIVATE UF_HAS_DCVC_DC=1)
    target_link_libraries(dcvc-uf PRIVATE uf_dcvc_dc)
endif()
if(UF_BUILD_EVALUATION)
    target_compile_definitions(dcvc-uf PRIVATE UF_HAS_EVALUATION=1)
    target_link_libraries(dcvc-uf PRIVATE uf_evaluation)
endif()
if(UF_BUILD_TRAINING)
    target_compile_definitions(dcvc-uf PRIVATE UF_HAS_TRAINING=1)
    target_link_libraries(dcvc-uf PRIVATE uf_training)
endif()

# Supply LibTorch through CMAKE_PREFIX_PATH. Configure never fetches dependencies.

option(UF_BUILD_TESTS "Build small UF tests without model/checkpoint loading" OFF)
if(UF_BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(UF_TEST_REFERENCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../DCVC-UF-INT16"
        CACHE PATH "UF Python reference checkout used by the comparison tests")
    add_executable(uf-model-free-tests tests/model_free.cpp)
    target_include_directories(uf-model-free-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(uf-model-free-tests PRIVATE uf_int16 uf_archive uf_entropy uf_cli uf_random uf_workflow)
    if(TARGET uf_cuda)
        target_compile_definitions(uf-model-free-tests PRIVATE UF_TEST_CUDA=1)
    endif()
    if(TARGET uf_fp16_native)
        target_compile_definitions(uf-model-free-tests PRIVATE UF_TEST_FP16=1)
        target_include_directories(uf-model-free-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/fp16/native)
        target_link_libraries(uf-model-free-tests PRIVATE uf_fp16 uf_fp16_native)
    endif()
    if(UF_BUILD_TRAINING)
        target_compile_definitions(uf-model-free-tests PRIVATE UF_TEST_TRAINING=1)
        target_link_libraries(uf-model-free-tests PRIVATE uf_training)
    endif()
    if(TARGET uf_evaluation)
        target_compile_definitions(uf-model-free-tests PRIVATE UF_TEST_EVALUATION=1)
        target_link_libraries(uf-model-free-tests PRIVATE uf_evaluation)
    endif()
    add_test(NAME uf-reference-fixtures
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/reference_fixtures.py
                --reference-root ${UF_TEST_REFERENCE_ROOT}
                --output ${CMAKE_CURRENT_BINARY_DIR}/model-free-fixtures.json)
    set_tests_properties(uf-reference-fixtures PROPERTIES FIXTURES_SETUP uf_reference TIMEOUT 60)
    add_test(NAME uf-model-free
             COMMAND uf-model-free-tests ${CMAKE_CURRENT_BINARY_DIR}/model-free-fixtures.json)
    set_tests_properties(uf-model-free PROPERTIES FIXTURES_REQUIRED uf_reference)
    add_test(NAME uf-cli-smoke
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_smoke.py
                --binary $<TARGET_FILE:dcvc-uf>
                --report ${CMAKE_CURRENT_BINARY_DIR}/cli-smoke-results.json)
    add_test(NAME uf-report-integration
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/report_integration.py
                --binary $<TARGET_FILE:dcvc-uf>)
    add_test(NAME uf-archive-integration
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/archive_integration.py
                --binary $<TARGET_FILE:dcvc-uf>)
    add_test(NAME uf-lifecycle-integration
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/lifecycle_integration.py
                --reference-root ${UF_TEST_REFERENCE_ROOT}
                --binary $<TARGET_FILE:dcvc-uf>)
    add_test(NAME uf-dashboard-integration
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/dashboard_integration.py
                --binary $<TARGET_FILE:dcvc-uf>)
    add_test(NAME uf-viewer-startup
        COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/viewer_startup.py
                --binary $<TARGET_FILE:dcvc-uf>)
    set(UF_TEST_YTDLP "" CACHE FILEPATH "Existing yt-dlp executable for loopback integration tests")
    if(UF_TEST_YTDLP)
        add_test(NAME uf-extractor-loopback
            COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/extractor_loopback.py
                    --native-tests $<TARGET_FILE:uf-model-free-tests> --ytdlp ${UF_TEST_YTDLP})
        set_tests_properties(uf-extractor-loopback PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
            TIMEOUT 120)
    endif()
    if(TARGET uf_evaluation)
        get_target_property(UF_TEST_EVALUATION_DEFINITIONS uf_evaluation COMPILE_DEFINITIONS)
        if("UF_HAS_CAIRO=1" IN_LIST UF_TEST_EVALUATION_DEFINITIONS)
            add_test(NAME uf-rd-plot-integration
                COMMAND ${Python3_EXECUTABLE} -S ${CMAKE_CURRENT_SOURCE_DIR}/tests/rd_plot_integration.py
                        --binary $<TARGET_FILE:dcvc-uf>
                        --evidence-dir ${CMAKE_CURRENT_BINARY_DIR}/rd-plot-evidence)
            set_tests_properties(uf-rd-plot-integration PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
                TIMEOUT 120)
        endif()
    endif()
    set_tests_properties(uf-model-free uf-reference-fixtures uf-cli-smoke uf-report-integration uf-archive-integration uf-lifecycle-integration uf-dashboard-integration uf-viewer-startup PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
        TIMEOUT 120)
endif()
