#pragma once

#include <string>
#include <vector>

namespace btx {
namespace cuda {

struct CudaDeviceInfo {
    int index = -1;
    std::string name;
    size_t total_mem_bytes = 0;
    int major = 0;
    int minor = 0;
    bool usable = false;
    std::string reason;
};

std::vector<CudaDeviceInfo> EnumerateDevices();
std::vector<int> GetUsableDeviceIndices();
bool HasCudaSupport();

} // namespace cuda
} // namespace btx
