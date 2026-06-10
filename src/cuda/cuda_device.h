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
void WarmupDevices(const std::vector<int>& device_ids);
bool HasCudaSupport();

// Free VRAM on device after cudaSetDevice(device).
size_t GetDeviceFreeMemBytes(int device);

} // namespace cuda
} // namespace btx
