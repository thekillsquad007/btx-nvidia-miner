cmake_minimum_required(VERSION 3.20)
project(btx-miner LANGUAGES CXX)

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
        if(CUDAToolkit_VERSION VERSION_LESS "12.0")
            message(FATAL_ERROR "CUDA 12.0 or newer is required for this miner (found ${CUDAToolkit_VERSION}). Old versions like 10.1 do not support the required C++17 / CUDA17 language standard.")
        endif()
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

if(BTX_MINER_ENABLE_CUDA)
    # Reasonable default arch list; user can override with -DCUDA_ARCHS="86;89;120"
    if(NOT DEFINED CUDA_ARCHS)
        set(CUDA_ARCHS "native")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})

    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Sources
set(COMMON_SOURCES
    src/common/cli.cpp
    src/common/log.cpp
    src/common/work_dispatcher.cpp
    src/common/metrics.cpp
    src/common/dev_fee.cpp
)

set(POW_SOURCES
    src/pow/field.cpp
    src/pow/matrix.cpp
    src/pow/noise.cpp
    src/pow/transcript.cpp
    src/pow/matmul_pow.cpp
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
target_sources(btx-miner PRIVATE src/cuda/cuda_solver.cpp src/cuda/cuda_device.cpp)

if(BTX_MINER_ENABLE_CUDA)
    target_sources(btx-miner PRIVATE src/cuda/matmul_kernel.cu)
    target_link_libraries(btx-miner PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(btx-miner PRIVATE BTX_MINER_HAS_CUDA=1)

    # CUDA specific flags for performance + debug
    set_target_properties(btx-miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    if(MSVC)
        target_compile_options(btx-miner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    else()
        target_compile_options(btx-miner PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
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
    )
    target_include_directories(btx-miner-test PRIVATE src include)
    target_link_libraries(btx-miner-test PRIVATE Threads::Threads)
    if(OpenSSL_FOUND)
        target_link_libraries(btx-miner-test PRIVATE OpenSSL::Crypto)
        target_compile_definitions(btx-miner-test PRIVATE HAVE_OPENSSL=1)
    endif()

    enable_testing()
    add_test(NAME pow_reference COMMAND btx-miner-test)
endif()

message(STATUS "Configured btx-nvidia-miner. CUDA=${BTX_MINER_ENABLE_CUDA}")
