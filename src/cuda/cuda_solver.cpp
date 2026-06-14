#include "cuda/cuda_solver.h"
#include "cuda/cuda_device.h"
#include "cuda/hashrate.h"
#include "pow/matmul_pow.h"

#include <atomic>
#include <cstdlib>
#include <iostream>

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

int AutoBatchCapForDevice(int device)
{
    const size_t total = GetDeviceTotalMemBytes(device);
    if (total >= 28ULL * 1024 * 1024 * 1024) {
        return 262144;
    }
    if (total >= 20ULL * 1024 * 1024 * 1024) {
        return 262144;
    }
    if (total >= 12ULL * 1024 * 1024 * 1024) {
        return 131072;
    }
    if (total >= 8ULL * 1024 * 1024 * 1024) {
        return 65536;
    }
    return 32768;
}

int AutoBatchSizeForDevice(int device, const pow::MatMulJob& job, int max_cap)
{
    constexpr size_t kReserveBytes = 128 * 1024 * 1024;
    const size_t free_bytes = GetDeviceFreeMemBytes(device);
    if (free_bytes <= kReserveBytes) {
        return 1024;
    }

    const int cap = max_cap > 0 ? max_cap : AutoBatchCapForDevice(device);
    const size_t usable = static_cast<size_t>((free_bytes - kReserveBytes) * 0.85);

    // Gate-path VRAM is mostly fixed pools + O(batch) gate buffers — not
    // LaunchBatchBytes(job,1) per nonce. Binary-search the largest batch that fits.
    int lo = 1024;
    int hi = cap;
    int best = 1024;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (LaunchBatchBytes(job, static_cast<size_t>(mid)) <= usable) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return best;
}

int ResolveLaunchBatch(
    int device,
    size_t active_index,
    const BatchLaunchConfig& config,
    const pow::MatMulJob& job)
{
    int requested = config.global_batch;
    if (!config.per_device.empty()) {
        if (active_index < config.per_device.size()) {
            requested = config.per_device[active_index];
        } else {
            requested = 0;
        }
    }
    if (requested <= 0) {
        return AutoBatchSizeForDevice(device, job);
    }
    return requested;
}

int MaxResolvedLaunchBatch(const BatchLaunchConfig& config, const pow::MatMulJob& job)
{
    const auto active = GetActiveDevices();
    int max_batch = 65536;
    for (size_t i = 0; i < active.size(); ++i) {
        max_batch = std::max(max_batch, ResolveLaunchBatch(active[i], i, config, job));
    }
    return max_batch;
}

int RecommendJobChunkSize(const BatchLaunchConfig& config, const pow::MatMulJob& job)
{
    return std::max(131072, MaxResolvedLaunchBatch(config, job) * 2);
}

void PrintGpuBatchPlan(const BatchLaunchConfig& config, const pow::MatMulJob& job)
{
    const auto active = GetActiveDevices();
    if (active.empty()) {
        return;
    }
    for (size_t i = 0; i < active.size(); ++i) {
        const int dev = active[i];
        const int launch = ResolveLaunchBatch(dev, i, config, job);
        const auto devices = EnumerateDevices();
        std::string name = "GPU";
        for (const auto& info : devices) {
            if (info.index == dev) {
                name = info.name;
                break;
            }
        }
        const bool is_auto =
            (config.per_device.empty() && config.global_batch <= 0) ||
            (!config.per_device.empty() && i < config.per_device.size() &&
             config.per_device[i] <= 0);
        std::cout << "GPU " << dev << " " << name
                  << " launch batch=" << launch;
        if (is_auto) {
            std::cout << " (auto, cap=" << AutoBatchCapForDevice(dev) << ")";
        }
        std::cout << std::endl;
    }
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
extern "C" void CudaSetPrefetchGate(int device, uint64_t next_start_nonce, size_t next_batch);
extern "C" bool CudaCollectTranscriptBatch(
    int device,
    size_t launch_batch,
    std::vector<uint256>& out_digests,
    std::vector<bool>& out_found);
extern "C" void CudaFinalizeTranscriptPipeline(int device);
#endif

std::vector<CudaSolution> CollectHits(
    const pow::MatMulJob& job,
    uint64_t start_nonce,
    const std::vector<uint256>& digests,
    const std::vector<bool>& found)
{
    std::vector<CudaSolution> solutions;
    solutions.reserve(4);
    for (size_t i = 0; i < found.size(); ++i) {
        if (!found[i]) continue;

        const uint64_t nonce = start_nonce + i;
        CudaSolution sol;
        sol.found = true;
        sol.nonce = nonce;
        sol.ntime = job.time;
        sol.digest = digests[i];
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
    int launch_batch)
{
    std::vector<CudaSolution> solutions;
    if (count == 0) return solutions;
    if (launch_batch <= 0) {
        launch_batch = AutoBatchSizeForDevice(dev, job);
    }

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t current_nonce = start_nonce;
    uint64_t remaining = count;

    bool pipeline_pending = false;
    uint64_t pending_start = 0;
    size_t pending_batch = 0;
    std::vector<uint256> pending_digests;
    std::vector<bool> pending_found;

    while (remaining > 0) {
        const size_t batch = static_cast<size_t>(
            std::min<uint64_t>(remaining, static_cast<uint64_t>(launch_batch)));

        if (pipeline_pending) {
            if (!CudaCollectTranscriptBatch(
                    dev, pending_batch, pending_digests, pending_found)) {
                break;
            }
            auto hits = CollectHits(job, pending_start, pending_digests, pending_found);
            solutions.insert(solutions.end(), hits.begin(), hits.end());
            pipeline_pending = false;
        }

        const uint64_t next_start = current_nonce + batch;
        const uint64_t next_remaining = remaining - batch;
        if (next_remaining > 0) {
            const size_t next_batch = static_cast<size_t>(
                std::min<uint64_t>(next_remaining, static_cast<uint64_t>(launch_batch)));
            CudaSetPrefetchGate(dev, next_start, next_batch);
        }

        pending_digests.assign(batch, uint256{});
        pending_found.assign(batch, false);
        if (!LaunchMatMulTranscriptBatch(
                dev, job, current_nonce, batch, job.target,
                pending_digests, pending_found)) {
            break;
        }
        pipeline_pending = true;
        pending_start = current_nonce;
        pending_batch = batch;

        current_nonce += batch;
        remaining -= batch;
    }

    if (pipeline_pending) {
        if (CudaCollectTranscriptBatch(dev, pending_batch, pending_digests, pending_found)) {
            auto hits = CollectHits(job, pending_start, pending_digests, pending_found);
            solutions.insert(solutions.end(), hits.begin(), hits.end());
        }
    }
    CudaFinalizeTranscriptPipeline(dev);

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
    const BatchLaunchConfig& batch_config
) {
    if (max_tries == 0) return {};
    if (job.block_height >= pow::kMatMulSeedV3Height && !job.has_parent_mtp) {
        return {};
    }

#ifdef BTX_MINER_HAS_CUDA
    auto usable = GetActiveDevices();
    if (!usable.empty()) {
        if (usable.size() == 1) {
            const int launch_batch = ResolveLaunchBatch(usable[0], 0, batch_config, job);
            return SolveOnDevice(usable[0], job, start_nonce, max_tries, launch_batch);
        }

        // Multi-GPU: dynamic work queue — fast GPUs pull more nonce batches.
        const size_t n = usable.size();
        const uint64_t slice_end = start_nonce + max_tries;
        std::atomic<uint64_t> next_nonce{start_nonce};

        struct DevResult {
            std::mutex mu;
            std::vector<CudaSolution> solutions;
        } merged;

        std::vector<std::thread> workers;
        workers.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const int launch_batch = ResolveLaunchBatch(usable[i], i, batch_config, job);
            workers.emplace_back([&, i, launch_batch]() {
                std::vector<CudaSolution> local;
                while (true) {
                    const uint64_t batch_start =
                        next_nonce.fetch_add(static_cast<uint64_t>(launch_batch));
                    if (batch_start >= slice_end) {
                        break;
                    }
                    const uint64_t batch_count =
                        std::min<uint64_t>(static_cast<uint64_t>(launch_batch),
                                           slice_end - batch_start);
                    auto sols = SolveOnDevice(usable[i], job, batch_start, batch_count, launch_batch);
                    local.insert(local.end(), sols.begin(), sols.end());
                }
                if (!local.empty()) {
                    std::lock_guard<std::mutex> lk(merged.mu);
                    merged.solutions.insert(merged.solutions.end(), local.begin(), local.end());
                }
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