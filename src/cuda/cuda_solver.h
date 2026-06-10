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

// Run a search for solutions on available CUDA devices.
// Uses the CPU reference for prep (noise etc for correctness) and for final cross-check.
// Splits work across devices for multi-GPU.
// Returns any solutions found in the [start_nonce, start_nonce + max_tries) range.
std::vector<CudaSolution> SolveBatchCuda(
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    uint64_t max_tries,
    int max_batch_size = 128  // tune per GPU mem
);

// Usable GPU indices: see cuda_device.h GetUsableDeviceIndices().

} // namespace cuda
} // namespace btx
