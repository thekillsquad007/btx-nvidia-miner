#include "cuda/cuda_solver.h"
#include "cuda/cuda_device.h"
#include "cuda/hashrate.h"
#include "pow/matmul_pow.h"

#include <cstdlib>

#ifdef BTX_MINER_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace btx {
namespace cuda {

size_t WorkspaceBytesPerNonce(const pow::MatMulJob& job)
{
    const uint32_t n = job.n;
    const uint32_t r = job.r;
    const uint32_t bsz = job.b;
    const size_t nn = static_cast<size_t>(n) * n;
    const size_t noise_elems = 2 * (static_cast<size_t>(n) * r + static_cast<size_t>(r) * n);
    const size_t scratch = static_cast<size_t>(bsz) * bsz;
    const bool use_v2 = job.block_height >= pow::kMatMulSeedV2Height;
    const bool use_factored = (n % 32U) == 0U && ((n / bsz) % 2U) == 0U;
    const size_t blocks_per_axis = static_cast<size_t>(n / bsz);
    const size_t matrix_elems = use_v2 ? (2 * nn) : (use_factored ? (2 * nn) : 0);
    const size_t factored_rhs_elems =
        use_factored ? (static_cast<size_t>(bsz) * n * blocks_per_axis) : 0;
    const size_t legacy_scratch_elems = use_factored ? 0 : (scratch * 5);
    const size_t per_nonce_elems =
        matrix_elems + nn + noise_elems + scratch + factored_rhs_elems + legacy_scratch_elems;
    return per_nonce_elems * sizeof(uint32_t);
}

namespace {

size_t BatchedPoolBytesForCount(size_t batch, uint32_t n, uint32_t r, uint32_t bsz)
{
    const size_t matrix = batch * static_cast<size_t>(n) * n;
    const size_t noise_left = batch * static_cast<size_t>(n) * r;
    const size_t noise_right = batch * static_cast<size_t>(r) * n;
    const size_t compress = batch * static_cast<size_t>(bsz) * bsz;
    const size_t words = batch * static_cast<size_t>(n / bsz) * (n / bsz);
    return (batch * 32 * 3) +
           (batch * 4 * 32) +
           (batch * 32) +
           (matrix * 4 * 4) +
           ((noise_left + noise_right) * 4 * 4) +
           (compress * 4) +
           (words * 4) +
           (batch * 32) +
           (batch * sizeof(int32_t)) +
           64;
}

size_t ExpectedPassedCap(size_t batch, int epsilon_bits)
{
    if (epsilon_bits <= 0) {
        return batch;
    }
    const unsigned shift = static_cast<unsigned>(std::min(epsilon_bits, 20));
    const size_t estimated = std::max<size_t>(1, batch >> shift);
    const size_t cap = std::max<size_t>(64, std::min(batch, estimated * 4));
    return std::max<size_t>(64, cap);
}

size_t LaunchBatchBytes(const pow::MatMulJob& job, size_t batch)
{
    const bool use_v2 = job.block_height >= pow::kMatMulSeedV2Height;
    if (!use_v2) {
        return batch * WorkspaceBytesPerNonce(job);
    }

    const size_t gate_bytes = batch * (sizeof(uint32_t) + 96) + 256;
    const size_t passed_cap = ExpectedPassedCap(batch, job.epsilon_bits);
    const size_t batched_bytes =
        BatchedPoolBytesForCount(passed_cap, job.n, job.r, job.b);
    const size_t legacy_ws = WorkspaceBytesPerNonce(job);
    return gate_bytes + batched_bytes + legacy_ws + batch * 33;
}

} // namespace

int AutoBatchSizeForDevice(int device, const pow::MatMulJob& job, int max_cap)
{
    const size_t per_launch = LaunchBatchBytes(job, 1);
    if (per_launch == 0) {
        return 256;
    }

    constexpr size_t kReserveBytes = 192 * 1024 * 1024;
    const size_t free_bytes = GetDeviceFreeMemBytes(device);
    if (free_bytes <= kReserveBytes) {
        return 256;
    }

    const size_t usable = static_cast<size_t>((free_bytes - kReserveBytes) * 0.85);
    int batch = static_cast<int>(usable / LaunchBatchBytes(job, 1));
    batch = std::max(1024, batch);
    if (max_cap > 0) {
        batch = std::min(batch, max_cap);
    }
    return batch;
}

namespace {

#ifdef BTX_MINER_HAS_CUDA
extern "C" bool LaunchMatMulTranscriptBatch(
    int device,
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    size_t batch_count,
    const std::vector<uint8_t>& target,
    std::vector<uint256>& out_digests,
    std::vector<bool>& out_found
);
#endif

bool CpuVerifySharesEnabled()
{
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("BTX_CUDA_CPU_VERIFY");
        cached = (env && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

std::vector<CudaSolution> CollectHits(
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    const std::vector<uint256>& digests,
    const std::vector<bool>& found)
{
    std::vector<CudaSolution> solutions;
    solutions.reserve(4);
    const bool cpu_verify = CpuVerifySharesEnabled();
    for (size_t i = 0; i < found.size(); ++i) {
        if (!found[i]) continue;

        const uint64_t nonce = start_nonce + i;
        const uint32_t ntime = job.time;
        uint256 digest = digests[i];
        if (cpu_verify) {
            uint256 cpu_digest;
            if (!pow::VerifySolution(job, nonce, ntime, cpu_digest) ||
                cpu_digest != digests[i]) {
                continue;
            }
            digest = cpu_digest;
        }

        CudaSolution sol;
        sol.found = true;
        sol.nonce = nonce;
        sol.ntime = ntime;
        sol.digest = digest;
        solutions.push_back(sol);
    }
    return solutions;
}

#ifdef BTX_MINER_HAS_CUDA
std::vector<CudaSolution> SolveOnDevice(
    int dev,
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    uint64_t count,
    int max_batch_size)
{
    std::vector<CudaSolution> solutions;
    if (count == 0) return solutions;

    const int launch_batch = max_batch_size > 0
        ? max_batch_size
        : AutoBatchSizeForDevice(dev, job);

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t current_nonce = start_nonce;
    uint64_t remaining = count;

    while (remaining > 0) {
        const size_t batch = static_cast<size_t>(
            std::min<uint64_t>(remaining, static_cast<uint64_t>(launch_batch)));

        std::vector<uint256> digests(batch);
        std::vector<bool> found(batch, false);

        if (!LaunchMatMulTranscriptBatch(
                dev, job, current_nonce, batch, job.target, digests, found)) {
            break;
        }

        auto hits = CollectHits(job, current_nonce, digests, found);
        solutions.insert(solutions.end(), hits.begin(), hits.end());

        current_nonce += batch;
        remaining -= batch;
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    RecordGpuWork(dev, count - remaining, elapsed_ms);
    return solutions;
}
#endif

std::vector<CudaSolution> SolveOnCpu(
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    uint64_t max_tries)
{
    const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const unsigned num_threads = std::min(hw_threads, 16u);

    struct ThreadResult {
        std::mutex mu;
        std::vector<CudaSolution> found;
        uint64_t tried = 0;
    } shared;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    const uint64_t per_thread = (max_tries + num_threads - 1) / num_threads;
    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            const uint64_t thread_start = start_nonce + static_cast<uint64_t>(t) * per_thread;
            const uint64_t thread_end = std::min(start_nonce + max_tries, thread_start + per_thread);
            uint64_t local_tried = 0;
            for (uint64_t current = thread_start; current < thread_end; ++current) {
                ++local_tried;
                uint256 d;
                if (pow::VerifySolution(job, current, job.time, d) &&
                    pow::DigestMeetsTarget(d, job.target)) {
                    CudaSolution sol;
                    sol.found = true;
                    sol.nonce = current;
                    sol.ntime = job.time;
                    sol.digest = d;
                    std::lock_guard<std::mutex> lk(shared.mu);
                    shared.found.push_back(sol);
                }
            }
            std::lock_guard<std::mutex> lk(shared.mu);
            shared.tried += local_tried;
        });
    }

    for (auto& w : workers) {
        if (w.joinable()) w.join();
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    RecordGpuWork(-1, shared.tried, elapsed_ms);
    return shared.found;
}

} // namespace

std::vector<CudaSolution> SolveBatchCuda(
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    uint64_t max_tries,
    int max_batch_size
) {
    if (max_tries == 0) return {};
    if (job.block_height >= pow::kMatMulSeedV3Height && !job.has_parent_mtp) {
        return {};
    }

#ifdef BTX_MINER_HAS_CUDA
    auto usable = GetActiveDevices();
    if (!usable.empty()) {
        if (usable.size() == 1) {
            return SolveOnDevice(usable[0], job, start_nonce, max_tries, max_batch_size);
        }

        // Multi-GPU: split the nonce range across devices and run in parallel.
        const size_t n = usable.size();
        const uint64_t per_dev = (max_tries + n - 1) / n;

        struct DevResult {
            std::mutex mu;
            std::vector<CudaSolution> solutions;
        } merged;

        std::vector<std::thread> workers;
        workers.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const uint64_t dev_start = start_nonce + static_cast<uint64_t>(i) * per_dev;
            const uint64_t dev_count = std::min(per_dev, start_nonce + max_tries - dev_start);
            if (dev_count == 0) continue;
            workers.emplace_back([&, i, dev_start, dev_count]() {
                auto sols = SolveOnDevice(usable[i], job, dev_start, dev_count, max_batch_size);
                std::lock_guard<std::mutex> lk(merged.mu);
                merged.solutions.insert(merged.solutions.end(), sols.begin(), sols.end());
            });
        }
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
        return merged.solutions;
    }
#endif

    return SolveOnCpu(job, start_nonce, max_tries);
}

} // namespace cuda
} // namespace btx