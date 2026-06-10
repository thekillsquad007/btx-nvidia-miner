#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace btx {
namespace cuda {

struct GpuHashrateSample {
    int device = -1;
    std::string name;
    uint64_t nonces_tried = 0;
    int64_t elapsed_ms = 0;
    double slice_nonces_per_sec = 0.0;
    double average_nonces_per_sec = 0.0;
};

// Configure which GPU indices to use. Empty means all usable devices.
void SetActiveDevices(const std::vector<int>& device_ids);

// When true, SolveBatchCuda uses the CPU path even if CUDA GPUs exist.
void SetForceCpu(bool force);

// Returns the resolved list of usable GPU indices (after applying SetActiveDevices).
std::vector<int> GetActiveDevices();

// Record work completed on a GPU during a slice/batch.
void RecordGpuWork(int device, uint64_t nonces_tried, int64_t elapsed_ms);

// Snapshot current per-GPU hashrate (slice + session average).
std::vector<GpuHashrateSample> GetGpuHashrateSnapshot();

// One-line log suitable for periodic printing.
std::string FormatGpuHashrateLog();

} // namespace cuda
} // namespace btx