#include "cuda/hashrate.h"

#include "cuda/cuda_device.h"

// GetUsableDeviceIndices lives in cuda_device.cpp

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace btx {
namespace cuda {

namespace {

struct DeviceStats {
    std::string name;
    uint64_t session_nonces = 0;
    double session_seconds = 0.0;
    std::chrono::steady_clock::time_point session_start =
        std::chrono::steady_clock::now();

    // Last slice instantaneous rate.
    double last_slice_nps = 0.0;
};

std::mutex g_mu;
std::vector<int> g_requested_devices;
bool g_force_cpu = false;
std::unordered_map<int, DeviceStats> g_stats;

std::string ShortGpuName(const std::string& full)
{
    if (full.empty()) return "GPU";
    // Drop trailing whitespace from cudaDeviceProp names.
    size_t end = full.find_last_not_of(' ');
    return full.substr(0, end == std::string::npos ? full.size() : end + 1);
}

} // namespace

void SetActiveDevices(const std::vector<int>& device_ids)
{
    std::lock_guard<std::mutex> lk(g_mu);
    g_requested_devices = device_ids;
}

void SetForceCpu(bool force)
{
    std::lock_guard<std::mutex> lk(g_mu);
    g_force_cpu = force;
}

std::vector<int> GetActiveDevices()
{
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_force_cpu) return {};
    auto all = GetUsableDeviceIndices();
    if (g_requested_devices.empty()) return all;

    std::vector<int> out;
    for (int id : g_requested_devices) {
        if (std::find(all.begin(), all.end(), id) != all.end()) {
            out.push_back(id);
        }
    }
    return out;
}

void RecordGpuWork(int device, uint64_t nonces_tried, int64_t elapsed_ms)
{
    if (device < 0 || nonces_tried == 0) return;

    const double slice_sec = std::max(elapsed_ms, int64_t{1}) / 1000.0;
    const double slice_nps = static_cast<double>(nonces_tried) / slice_sec;

    std::lock_guard<std::mutex> lk(g_mu);
    auto& st = g_stats[device];
    if (st.name.empty()) {
        if (device < 0) {
            st.name = "CPU";
        } else {
            for (const auto& d : EnumerateDevices()) {
                if (d.index == device) {
                    st.name = ShortGpuName(d.name);
                    break;
                }
            }
            if (st.name.empty()) {
                st.name = "GPU" + std::to_string(device);
            }
        }
    }

    st.session_nonces += nonces_tried;
    st.session_seconds += slice_sec;
    st.last_slice_nps = slice_nps;
}

std::vector<GpuHashrateSample> GetGpuHashrateSnapshot()
{
    std::lock_guard<std::mutex> lk(g_mu);
    std::vector<GpuHashrateSample> out;
    out.reserve(g_stats.size());

    for (const auto& kv : g_stats) {
        GpuHashrateSample s;
        s.device = kv.first;
        s.name = kv.second.name;
        s.slice_nonces_per_sec = kv.second.last_slice_nps;
        if (kv.second.session_seconds > 0.0) {
            s.average_nonces_per_sec =
                static_cast<double>(kv.second.session_nonces) / kv.second.session_seconds;
        }
        out.push_back(s);
    }

    std::sort(out.begin(), out.end(),
              [](const GpuHashrateSample& a, const GpuHashrateSample& b) {
                  return a.device < b.device;
              });
    return out;
}

std::string FormatGpuHashrateLog()
{
    auto samples = GetGpuHashrateSnapshot();
    if (samples.empty()) return "hashrate: (no GPU work recorded yet)";

    std::ostringstream ss;
    ss << "hashrate:";
    double total_avg = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        if (i) ss << " |";
        ss << " " << (s.device >= 0 ? "GPU" + std::to_string(s.device) : "CPU")
           << " " << s.name
           << " " << std::fixed << std::setprecision(2) << s.slice_nonces_per_sec
           << " H/s (avg " << s.average_nonces_per_sec << " H/s)";
        total_avg += s.average_nonces_per_sec;
    }
    if (samples.size() > 1) {
        ss << " | total " << std::fixed << std::setprecision(2) << total_avg << " H/s";
    }
    return ss.str();
}

} // namespace cuda
} // namespace btx