#pragma once

#include "pow/matmul_pow.h"

#include <cstdint>
#include <vector>

namespace btx {
namespace cuda {

// Result from device search. The host will always cross-check with CPU reference.
struct CudaSolution {
    bool found = false;
    uint64_t nonce = 0;
    uint32_t ntime = 0;
    uint256 digest;  // the matmul_digest
};

// Per-nonce GPU workspace bytes (for auto batch sizing).
size_t WorkspaceBytesPerNonce(const pow::MatMulJob& job);

// VRAM-tiered upper bound for auto batch (before free-VRAM fit).
int AutoBatchCapForDevice(int device);

// Pick a launch batch from free VRAM. Returns at least 1024, at most max_cap.
int AutoBatchSizeForDevice(int device, const pow::MatMulJob& job, int max_cap = 0);

struct BatchLaunchConfig {
    int global_batch = 0;              // used when per_device is empty; 0 = auto
    std::vector<int> per_device;     // one entry per active GPU index; 0 = auto
};

// Resolve launch batch for one GPU (active_index into GetActiveDevices()).
int ResolveLaunchBatch(
    int device,
    size_t active_index,
    const BatchLaunchConfig& config,
    const pow::MatMulJob& job);

// Print resolved/auto batch per active GPU (for install scripts / debugging).
void PrintGpuBatchPlan(const BatchLaunchConfig& config, const pow::MatMulJob& job);

// Largest resolved per-GPU launch batch; use for default job-chunk sizing.
int MaxResolvedLaunchBatch(const BatchLaunchConfig& config, const pow::MatMulJob& job);
int RecommendJobChunkSize(const BatchLaunchConfig& config, const pow::MatMulJob& job);

// Run a search for solutions on available CUDA devices.
std::vector<CudaSolution> SolveBatchCuda(
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    uint64_t max_tries,
    const BatchLaunchConfig& batch_config = {}
);

// Usable GPU indices: see cuda_device.h GetUsableDeviceIndices().

} // namespace cuda
} // namespace btx
