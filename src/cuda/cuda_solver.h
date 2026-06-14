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

// Pick a launch batch from free VRAM. Returns at least 64, at most max_cap.
int AutoBatchSizeForDevice(int device, const pow::MatMulJob& job, int max_cap = 65536);

// Run a search for solutions on available CUDA devices.
// max_batch_size <= 0 selects AutoBatchSizeForDevice per GPU.
std::vector<CudaSolution> SolveBatchCuda(
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    uint64_t max_tries,
    int max_batch_size = 0
);

// Usable GPU indices: see cuda_device.h GetUsableDeviceIndices().

} // namespace cuda
} // namespace btx
