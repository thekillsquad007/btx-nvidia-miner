#include "cuda/cuda_solver.h"
#include "cuda/cuda_device.h"
#include "cuda/hashrate.h"
#include "pow/matmul_pow.h"

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

namespace {

#ifdef BTX_MINER_HAS_CUDA
extern "C" bool LaunchMatMulTranscriptBatch(
    int device,
    const pow::MatMulJob& job,
    const std::vector<uint64_t>& nonces,
    const std::vector<uint8_t>& target,
    std::vector<uint256>& out_digests,
    std::vector<bool>& out_found
);
#endif

std::vector<CudaSolution> CollectHits(
    const pow::MatMulJob& job,
    const std::vector<uint64_t>& batch_nonces,
    const std::vector<uint256>& digests,
    const std::vector<bool>& found)
{
    std::vector<CudaSolution> solutions;
    for (size_t i = 0; i < batch_nonces.size(); ++i) {
        if (!found[i]) continue;
        uint256 cpu_d;
        if (!pow::VerifySolution(job, batch_nonces[i], job.time, cpu_d) ||
            cpu_d != digests[i]) {
            continue;
        }
        const uint8_t* dd = reinterpret_cast<const uint8_t*>(&digests[i]);
        bool below = true;
        for (size_t k = 0; k < job.target.size() && k < 32; ++k) {
            if (dd[k] > job.target[k]) { below = false; break; }
            if (dd[k] < job.target[k]) break;
        }
        if (!below) continue;
        CudaSolution sol;
        sol.found = true;
        sol.nonce = batch_nonces[i];
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
    int max_batch_size)
{
    std::vector<CudaSolution> solutions;
    if (count == 0) return solutions;

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t current_nonce = start_nonce;
    uint64_t remaining = count;

    while (remaining > 0) {
        int batch = static_cast<int>(std::min<uint64_t>(remaining, max_batch_size));
        std::vector<uint64_t> batch_nonces;
        batch_nonces.reserve(batch);
        for (int i = 0; i < batch; ++i) {
            batch_nonces.push_back(current_nonce + i);
        }

        std::vector<uint256> digests(batch);
        std::vector<bool> found(batch, false);

        if (!LaunchMatMulTranscriptBatch(dev, job, batch_nonces, job.target, digests, found)) {
            break;
        }

        auto hits = CollectHits(job, batch_nonces, digests, found);
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
                if (pow::VerifySolution(job, current, job.time, d)) {
                    const uint8_t* dd = reinterpret_cast<const uint8_t*>(&d);
                    bool below = true;
                    for (size_t k = 0; k < job.target.size() && k < 32; ++k) {
                        if (dd[k] > job.target[k]) { below = false; break; }
                        if (dd[k] < job.target[k]) break;
                    }
                    if (below) {
                        CudaSolution sol;
                        sol.found = true;
                        sol.nonce = current;
                        sol.ntime = job.time;
                        sol.digest = d;
                        std::lock_guard<std::mutex> lk(shared.mu);
                        shared.found.push_back(sol);
                    }
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