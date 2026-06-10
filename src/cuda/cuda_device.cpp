#include "cuda/cuda_device.h"

#ifdef BTX_MINER_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <sstream>

namespace btx {
namespace cuda {

bool HasCudaSupport()
{
#ifdef BTX_MINER_HAS_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
#else
    return false;
#endif
}

std::vector<CudaDeviceInfo> EnumerateDevices()
{
    std::vector<CudaDeviceInfo> out;
#ifdef BTX_MINER_HAS_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        CudaDeviceInfo d;
        d.reason = "cudaGetDeviceCount failed (driver/toolkit mismatch?)";
        out.push_back(d);
        return out;
    }
    for (int i = 0; i < count; ++i) {
        CudaDeviceInfo info;
        info.index = i;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            info.name = prop.name;
            info.total_mem_bytes = prop.totalGlobalMem;
            info.major = prop.major;
            info.minor = prop.minor;
            // Very rough capability gate (Pascal+ is fine for our kernels)
            info.usable = (prop.major >= 6);
            if (!info.usable) info.reason = "compute capability too low (<6.0)";
        } else {
            info.reason = "cudaGetDeviceProperties failed";
        }
        out.push_back(info);
    }
#else
    CudaDeviceInfo d;
    d.reason = "built without CUDA support";
    out.push_back(d);
#endif
    return out;
}

} // namespace cuda
} // namespace btx
