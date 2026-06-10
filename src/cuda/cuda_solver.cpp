#include "cuda/cuda_solver.h"
#include "cuda/cuda_device.h"
#include "pow/matmul_pow.h"

#ifdef BTX_MINER_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstring>
#include <vector>
#include <thread>

namespace btx {
namespace cuda {

namespace {

// Host-side prep for a sub-batch: using the trusted CPU reference, prepare
// per-nonce the 4 small noise rects (E_L etc). This keeps CPU light (r=8)
// and guarantees bit-exact noise with the reference.
// We upload the rects + fixed A/B + sigma params, and let the kernel do
// on-device low-rank + full transcript matmul.

struct PerNonceNoise {
    // We will flatten the 4 rects into one buffer per nonce for upload.
    // For simplicity in v1, the solver will compute full A' B' on host using
    // reference (still cheap for prep of small batches) and upload those.
    // This guarantees the device only has to match the transcript part.
};

} // namespace

std::vector<int> GetUsableDeviceIndices() {
    std::vector<int> ids;
    auto devs = EnumerateDevices();
    for (auto& d : devs) {
        if (d.usable && d.index >= 0) ids.push_back(d.index);
    }
    return ids;
}

#ifdef BTX_MINER_HAS_CUDA

// Forward declaration of the launcher implemented in .cu
// It takes A, B (once), then for the batch a list of A_prime / B_prime or noise,
// runs the kernel, returns per-nonce digests for those that met target.
extern "C" bool LaunchMatMulTranscriptBatch(
    int device,
    const pow::MatMulJob& job,
    const std::vector<uint64_t>& nonces,
    const std::vector<uint8_t>& target,  // the one to check against (share or block)
    std::vector<uint256>& out_digests,   // same size as nonces, only valid if found
    std::vector<bool>& out_found
);

#endif

std::vector<CudaSolution> SolveBatchCuda(
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    uint64_t max_tries,
    int max_batch_size
) {
    std::vector<CudaSolution> solutions;

    if (max_tries == 0) return solutions;

#ifdef BTX_MINER_HAS_CUDA
    auto usable = GetUsableDeviceIndices();
    if (!usable.empty()) {
        // For v1: use first device (multi-GPU round-robin can be added later)
        int dev = usable[0];

        uint64_t current_nonce = start_nonce;
        uint64_t remaining = max_tries;

        while (remaining > 0) {
            int batch = static_cast<int>(std::min<uint64_t>(remaining, max_batch_size));
            std::vector<uint64_t> batch_nonces;
            batch_nonces.reserve(batch);
            for (int i = 0; i < batch; ++i) {
                batch_nonces.push_back(current_nonce + i);
            }

            std::vector<uint256> digests(batch);
            std::vector<bool> found(batch, false);

            bool launched = LaunchMatMulTranscriptBatch(
                dev, job, batch_nonces, job.target,
                digests, found
            );

            if (!launched) {
                break;
            }

            // Cross-check every candidate reported by the device
            for (int i = 0; i < batch; ++i) {
                if (found[i]) {
                    uint256 cpu_d;
                    if (pow::VerifySolution(job, batch_nonces[i], job.time, cpu_d) &&
                        cpu_d == digests[i]) {
                        const uint8_t* dd = reinterpret_cast<const uint8_t*>(&digests[i]);
                        bool below = true;
                        for (size_t k = 0; k < job.target.size() && k < 32; ++k) {
                            if (dd[k] > job.target[k]) { below = false; break; }
                            if (dd[k] < job.target[k]) break;
                        }
                        if (below) {
                            CudaSolution sol;
                            sol.found = true;
                            sol.nonce = batch_nonces[i];
                            sol.ntime = job.time;
                            sol.digest = digests[i];
                            solutions.push_back(sol);
                        }
                    }
                }
            }

            current_nonce += batch;
            remaining -= batch;
        }

        return solutions;
    }
#endif

    // CPU reference fallback (used when no CUDA, no usable GPU, or CUDA not enabled)
    uint64_t current_nonce = start_nonce;
    uint64_t remaining = max_tries;

    while (remaining > 0) {
        uint256 d;
        if (pow::VerifySolution(job, current_nonce, job.time, d)) {
            const uint8_t* dd = reinterpret_cast<const uint8_t*>(&d);
            bool below = true;
            for (size_t k = 0; k < job.target.size() && k < 32; ++k) {
                if (dd[k] > job.target[k]) { below = false; break; }
                if (dd[k] < job.target[k]) break;
            }
            if (below) {
                CudaSolution sol;
                sol.found = true;
                sol.nonce = current_nonce;
                sol.ntime = job.time;
                sol.digest = d;
                solutions.push_back(sol);
            }
        }
        if (current_nonce == UINT64_MAX) break;
        ++current_nonce;
        --remaining;
    }
    return solutions;
}

} // namespace cuda
} // namespace btx
