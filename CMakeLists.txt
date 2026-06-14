cmake_minimum_required(VERSION 3.20)
project(btx-miner LANGUAGES CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_architectures.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BTX_MINER_BUILD_TESTS "Build unit tests" ON)
option(BTX_MINER_ENABLE_CUDA "Enable CUDA kernels (disable for CPU-only reference build)" ON)
option(BTX_MINER_STATIC "Try to link static where possible (CUDA is usually dynamic)" OFF)

if(NOT BTX_MINER_ENABLE_CUDA)
    message(STATUS "CUDA disabled - building CPU reference only")
endif()

# Find dependencies
find_package(Threads REQUIRED)

# For solo JSON-RPC we use libcurl (easy, widely available)
find_package(CURL QUIET)
if(NOT CURL_FOUND)
    message(STATUS "libcurl not found - solo mining RPC client will be a stub until installed")
endif()

# CUDA handling - do NOT enable the language unless the user asked for it and the toolkit exists
if(BTX_MINER_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        if(NOT CUDAToolkit_NVCC_EXECUTABLE)
            set(CUDAToolkit_NVCC_EXECUTABLE "${CUDAToolkit_BIN_DIR}/nvcc")
        endif()
        if(NOT EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
            set(_btx_nvcc_hint "nvcc")
        else()
            set(_btx_nvcc_hint "${CUDAToolkit_NVCC_EXECUTABLE}")
        endif()

        execute_process(
            COMMAND ${_btx_nvcc_hint} --version
            OUTPUT_VARIABLE _btx_nvcc_version_text
            ERROR_VARIABLE _btx_nvcc_version_text
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(_btx_nvcc_version "${CUDAToolkit_VERSION}")
        if(_btx_nvcc_version_text MATCHES "release ([0-9]+\\.[0-9]+)")
            set(_btx_nvcc_version "${CMAKE_MATCH_1}")
        endif()

        if(_btx_nvcc_version VERSION_LESS "12.0")
            message(FATAL_ERROR
                "CUDA 12.0+ required (nvcc reports ${_btx_nvcc_version} at ${_btx_nvcc_hint}). "
                "Mining rigs with RTX 3060/3070 (sm_86) need a recent toolkit — install cuda-compiler-12-4 "
                "and ensure /usr/local/cuda/bin is before /usr/bin in PATH.")
        endif()

        # Map nvcc version -> highest SM this compiler can target.
        if(_btx_nvcc_version VERSION_GREATER_EQUAL "12.8")
            set(_btx_nvcc_max_sm 120)
        elseif(_btx_nvcc_version VERSION_GREATER_EQUAL "12.0")
            set(_btx_nvcc_max_sm 90)
        elseif(_btx_nvcc_version VERSION_GREATER_EQUAL "11.8")
            set(_btx_nvcc_max_sm 89)
        elseif(_btx_nvcc_version VERSION_GREATER_EQUAL "11.1")
            set(_btx_nvcc_max_sm 86)
        elseif(_btx_nvcc_version VERSION_GREATER_EQUAL "11.0")
            set(_btx_nvcc_max_sm 80)
        else()
            set(_btx_nvcc_max_sm 75)
        endif()

        # Must be set BEFORE enable_language(CUDA). CMake's nvcc probe fails with
        # "CUDA_ARCHITECTURES is empty" if this is unset (common on Hive/container rigs).
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            if(DEFINED CUDA_ARCHS)
                set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHS}")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "${BTX_RELEASE_CUDA_ARCHITECTURES}")
            endif()
        endif()

        # Filter requested arches to what this nvcc actually supports.
        string(REPLACE ";" " " _btx_arch_tokens "${CMAKE_CUDA_ARCHITECTURES}")
        separate_arguments(_btx_arch_list UNIX_COMMAND "${_btx_arch_tokens}")
        set(_btx_arch_filtered "")
        set(_btx_arch_dropped "")
        foreach(_btx_arch IN LISTS _btx_arch_list)
            if(_btx_arch GREATER _btx_nvcc_max_sm)
                list(APPEND _btx_arch_dropped "${_btx_arch}")
            else()
                list(APPEND _btx_arch_filtered "${_btx_arch}")
            endif()
        endforeach()
        if(_btx_arch_dropped)
            message(FATAL_ERROR
                "nvcc ${_btx_nvcc_version} (${_btx_nvcc_hint}) cannot target GPU arch(s): "
                "${_btx_arch_dropped} (max sm_${_btx_nvcc_max_sm}). "
                "Install CUDA Toolkit 12.x and put /usr/local/cuda/bin before /usr/bin in PATH.")
        endif()
        if(NOT _btx_arch_filtered)
            message(FATAL_ERROR
                "No usable CUDA architectures after filtering for nvcc ${_btx_nvcc_version}. "
                "Pass -DCMAKE_CUDA_ARCHITECTURES=<sm_list> matching your GPU and toolkit.")
        endif()
        list(REMOVE_DUPLICATES _btx_arch_filtered)
        list(JOIN _btx_arch_filtered ";" CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA SM targets" FORCE)

        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "CUDA nvcc: ${_btx_nvcc_hint} (${_btx_nvcc_version}, max sm_${_btx_nvcc_max_sm})")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        message(STATUS "CUDA Toolkit found - enabling GPU kernels")
    else()
        message(WARNING "BTX_MINER_ENABLE_CUDA=ON but CUDA Toolkit not found - falling back to CPU-only")
        set(BTX_MINER_ENABLE_CUDA OFF)
    endif()
endif()

# JSON (header-only). We vendor a tiny copy or expect user to have nlohmann; for simplicity we embed a minimal one or use rapidjson later.
# For first cut we will use nlohmann/json if findable, else a simple replacement. Add as submodule or fetch in real setup.
# Here we assume a modern enough env or we will implement a tiny RPC layer.

# SHA256: prefer system crypto or OpenSSL for the reference path
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    message(STATUS "Using OpenSSL for SHA256 in CPU reference")
    set(HAVE_OPENSSL 1)
else()
    message(STATUS "OpenSSL not found - will use bundled SHA256 implementation")
    set(HAVE_OPENSSL 0)
endif()

# Sources
set(COMMON_SOURCES
    src/common/cli.cpp
    src/common/log.cpp
    src/common/work_dispatcher.cpp
    src/common/metrics.cpp
    src/common/dev_fee.cpp
    src/common/hardware.cpp
)

set(POW_SOURCES
    src/pow/field.cpp
    src/pow/matrix.cpp
    src/pow/noise.cpp
    src/pow/transcript.cpp
    src/pow/matmul_pow.cpp
    src/pow/uint256_stub.cpp
    src/pow/sha256_portable.cpp   # fallback when no OpenSSL
)

set(SOLO_SOURCES
    src/solo/rpc_client.cpp
    src/solo/solo_miner.cpp
)

set(STRATUM_SOURCES
    src/stratum/stratum_client.cpp
    src/stratum/stratum_protocol.cpp
)

set(CUDA_SOURCES
    src/cuda/cuda_device.cpp
    src/cuda/hashrate.cpp
    src/cuda/gpu_sha256.cu
    src/cuda/matmul_kernel.cu
    src/cuda/cuda_solver.cpp
)

set(MAIN_SOURCES
    src/main.cpp
)

add_executable(btx-miner
    ${MAIN_SOURCES}
    ${COMMON_SOURCES}
    ${POW_SOURCES}
    ${SOLO_SOURCES}
    ${STRATUM_SOURCES}
)

target_include_directories(btx-miner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(btx-miner PRIVATE Threads::Threads)
if(CURL_FOUND)
    target_link_libraries(btx-miner PRIVATE CURL::libcurl)
    target_compile_definitions(btx-miner PRIVATE HAVE_CURL=1)
endif()

if(OpenSSL_FOUND)
    target_link_libraries(btx-miner PRIVATE OpenSSL::Crypto)
    target_compile_definitions(btx-miner PRIVATE HAVE_OPENSSL=1)
endif()

# Always compile the CUDA host-side solver (it has CPU fallback when no CUDA)
target_sources(btx-miner PRIVATE src/cuda/cuda_solver.cpp src/cuda/cuda_device.cpp src/cuda/hashrate.cpp)

if(BTX_MINER_ENABLE_CUDA)
    target_sources(btx-miner PRIVATE src/cuda/gpu_sha256.cu src/cuda/matmul_kernel.cu)
    target_link_libraries(btx-miner PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(btx-miner PRIVATE BTX_MINER_HAS_CUDA=1)

    # CUDA specific flags for performance + debug
    set_target_properties(btx-miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    if(MSVC)
        target_compile_options(btx-miner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    else()
        target_compile_options(btx-miner PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>
        )
    endif()
endif()

# Install
install(TARGETS btx-miner DESTINATION bin)

# Optional tests (CPU reference correctness + vectors)
if(BTX_MINER_BUILD_TESTS)
    add_executable(btx-miner-test
        tests/test_pow.cpp
        ${POW_SOURCES}
        ${COMMON_SOURCES}
        src/stratum/stratum_protocol.cpp
    )
    target_include_directories(btx-miner-test PRIVATE src include)
    target_link_libraries(btx-miner-test PRIVATE Threads::Threads)
    if(OpenSSL_FOUND)
        target_link_libraries(btx-miner-test PRIVATE OpenSSL::Crypto)
        target_compile_definitions(btx-miner-test PRIVATE HAVE_OPENSSL=1)
    endif()

    add_executable(btx-miner-oracle-test
        tests/test_oracle_digest.cpp
        ${POW_SOURCES}
        src/stratum/stratum_protocol.cpp
    )
    target_include_directories(btx-miner-oracle-test PRIVATE src include)
    target_link_libraries(btx-miner-oracle-test PRIVATE Threads::Threads)
    if(OpenSSL_FOUND)
        target_link_libraries(btx-miner-oracle-test PRIVATE OpenSSL::Crypto)
        target_compile_definitions(btx-miner-oracle-test PRIVATE HAVE_OPENSSL=1)
    endif()

    enable_testing()
    add_test(NAME pow_reference COMMAND btx-miner-test)
    add_test(NAME oracle_digest COMMAND btx-miner-oracle-test)

    if(BTX_MINER_ENABLE_CUDA)
        add_executable(btx-miner-cuda-test
            tests/test_cuda_pow.cpp
            ${POW_SOURCES}
            src/cuda/gpu_sha256.cu
            src/cuda/matmul_kernel.cu
        )
        target_include_directories(btx-miner-cuda-test PRIVATE src include)
        target_link_libraries(btx-miner-cuda-test PRIVATE CUDA::cudart Threads::Threads)
        target_compile_definitions(btx-miner-cuda-test PRIVATE BTX_MINER_HAS_CUDA=1)
        set_target_properties(btx-miner-cuda-test PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        )
        add_test(NAME cuda_pow_reference COMMAND btx-miner-cuda-test)
    endif()
endif()

message(STATUS "Configured btx-nvidia-miner. CUDA=${BTX_MINER_ENABLE_CUDA}")
