#include "common/hardware.h"

#ifdef BTX_MINER_HAS_CUDA
#include "cuda/cuda_device.h"
#endif

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>
#include <unistd.h>

namespace btx {
namespace common {

namespace {

std::string RunCommand(const char* cmd)
{
    std::array<char, 4096> buf{};
    std::string out;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return {};
    while (true) {
        size_t n = fread(buf.data(), 1, buf.size(), pipe);
        if (n == 0) break;
        out.append(buf.data(), n);
    }
    pclose(pipe);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out;
}

std::string ReadFirstLineFromFile(const char* path, const char* prefix)
{
    std::ifstream f(path);
    if (!f) return {};
    std::string line;
    while (std::getline(f, line)) {
        if (prefix && line.rfind(prefix, 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos && pos + 1 < line.size()) {
                std::string val = line.substr(pos + 1);
                while (!val.empty() && val.front() == ' ') val.erase(val.begin());
                return val;
            }
        }
    }
    return {};
}

std::string CpuModel()
{
    std::string model = ReadFirstLineFromFile("/proc/cpuinfo", "model name");
    if (!model.empty()) return model;
    char host[256]{};
    if (gethostname(host, sizeof(host) - 1) == 0 && host[0]) {
        return host;
    }
    return "unknown";
}

int CpuThreadsTotal()
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<int>(n) : 0;
}

double RamGbTotal()
{
    std::ifstream f("/proc/meminfo");
    if (!f) return 0.0;
    std::string key;
    long kb = 0;
    while (f >> key >> kb) {
        if (key == "MemTotal:") {
            return static_cast<double>(kb) / (1024.0 * 1024.0);
        }
    }
    return 0.0;
}

double RamGbUsed()
{
    std::ifstream f("/proc/meminfo");
    if (!f) return 0.0;
    long total_kb = 0;
    long avail_kb = 0;
    std::string key;
    long val = 0;
    while (f >> key >> val) {
        if (key == "MemTotal:") total_kb = val;
        else if (key == "MemAvailable:") avail_kb = val;
    }
    if (total_kb > 0 && avail_kb >= 0) {
        return static_cast<double>(total_kb - avail_kb) / (1024.0 * 1024.0);
    }
    return 0.0;
}

std::string OsString()
{
    std::ifstream f("/etc/os-release");
    if (f) {
        std::string line;
        std::string pretty;
        while (std::getline(f, line)) {
            if (line.rfind("PRETTY_NAME=", 0) == 0) {
                pretty = line.substr(12);
                if (!pretty.empty() && pretty.front() == '"') pretty.erase(pretty.begin());
                if (!pretty.empty() && pretty.back() == '"') pretty.pop_back();
                break;
            }
        }
        if (!pretty.empty()) return pretty;
    }
    return "Linux";
}

std::string Hostname()
{
    char host[256]{};
    if (gethostname(host, sizeof(host) - 1) == 0 && host[0]) {
        return host;
    }
    return {};
}

struct GpuStatic {
    int index = -1;
    std::string model;
    double vram_gb = 0.0;
    std::string compute_capability;
    std::string pcie_link;
    std::string gpu_uuid;
};

struct GpuRuntime {
    std::string gpu_uuid;
    int util_pct = -1;
    double power_w = -1.0;
    int temp_c = -1;
};

std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> cells;
    std::string cell;
    for (char c : line) {
        if (c == ',') {
            while (!cell.empty() && cell.front() == ' ') cell.erase(cell.begin());
            while (!cell.empty() && cell.back() == ' ') cell.pop_back();
            cells.push_back(cell);
            cell.clear();
        } else {
            cell.push_back(c);
        }
    }
    while (!cell.empty() && cell.front() == ' ') cell.erase(cell.begin());
    while (!cell.empty() && cell.back() == ' ') cell.pop_back();
    cells.push_back(cell);
    return cells;
}

std::vector<GpuStatic> EnumerateNvidiaGpus()
{
    std::vector<GpuStatic> out;
    const char* cmd =
        "nvidia-smi --query-gpu=index,name,memory.total,compute_cap,"
        "pcie.link.gen.current,pcie.link.width.current,uuid "
        "--format=csv,noheader,nounits 2>/dev/null";
    std::string raw = RunCommand(cmd);
    if (raw.empty()) return out;

    std::istringstream ss(raw);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        auto cells = SplitCsvLine(line);
        if (cells.size() < 7) continue;
        GpuStatic g;
        g.index = std::atoi(cells[0].c_str());
        g.model = cells[1];
        try {
            double mb = std::stod(cells[2]);
            g.vram_gb = mb / 1024.0;
        } catch (...) {
            g.vram_gb = 0.0;
        }
        const std::string& cc = cells[3];
        if (!cc.empty() && cc != "[Not Supported]" && cc != "[N/A]") {
            std::string digits;
            for (char c : cc) {
                if (c == '.') continue;
                if (std::isdigit(static_cast<unsigned char>(c))) digits.push_back(c);
            }
            if (!digits.empty()) g.compute_capability = "sm_" + digits;
        }
        const std::string& gen = cells[4];
        const std::string& width = cells[5];
        if (!gen.empty() && gen != "[Not Supported]" && gen != "[N/A]" &&
            !width.empty() && width != "[Not Supported]" && width != "[N/A]") {
            g.pcie_link = "Gen" + gen + " x" + width;
        }
        g.gpu_uuid = cells[6];
        out.push_back(g);
    }
    return out;
}

std::vector<GpuStatic> EnumerateCudaFallbackGpus(const std::vector<int>& active_indices)
{
    std::vector<GpuStatic> out;
#ifndef BTX_MINER_HAS_CUDA
    (void)active_indices;
    return out;
#else
    const auto devs = cuda::EnumerateDevices();
    const std::string host = Hostname();
    for (const auto& d : devs) {
        if (!d.usable || d.index < 0) continue;
        if (!active_indices.empty()) {
            bool wanted = false;
            for (int id : active_indices) {
                if (id == d.index) { wanted = true; break; }
            }
            if (!wanted) continue;
        }
        GpuStatic g;
        g.index = d.index;
        g.model = d.name;
        g.vram_gb = static_cast<double>(d.total_mem_bytes) / (1024.0 * 1024.0 * 1024.0);
        if (d.major > 0 || d.minor > 0) {
            g.compute_capability = "sm_" + std::to_string(d.major) + std::to_string(d.minor);
        }
        g.gpu_uuid = "cuda-" + std::to_string(d.index) + "-" + host;
        out.push_back(g);
    }
    return out;
#endif
}

std::vector<GpuStatic> CollectStaticGpus(const std::vector<int>& active_indices)
{
    auto gpus = EnumerateNvidiaGpus();
    if (gpus.empty()) {
        return EnumerateCudaFallbackGpus(active_indices);
    }
    if (active_indices.empty()) return gpus;

    std::vector<GpuStatic> filtered;
    for (const auto& g : gpus) {
        for (int id : active_indices) {
            if (g.index == id) {
                filtered.push_back(g);
                break;
            }
        }
    }
    return filtered.empty() ? gpus : filtered;
}

void DriverAndCuda(std::string& driver, std::string& cuda_ver)
{
    std::string out = RunCommand(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null");
    if (!out.empty()) {
        driver = out.substr(0, out.find('\n'));
    }
    out = RunCommand("nvidia-smi 2>/dev/null");
    const auto pos = out.find("CUDA Version:");
    if (pos != std::string::npos) {
        size_t start = pos + 13;
        while (start < out.size() && out[start] == ' ') ++start;
        size_t end = start;
        while (end < out.size() && out[end] != ' ' && out[end] != '\n') ++end;
        cuda_ver = out.substr(start, end - start);
    }
}

std::vector<GpuRuntime> CollectGpuRuntime()
{
    std::vector<GpuRuntime> out;
    const char* cmd =
        "nvidia-smi --query-gpu=uuid,utilization.gpu,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits 2>/dev/null";
    std::string raw = RunCommand(cmd);
    if (raw.empty()) {
        for (const auto& g : CollectStaticGpus({})) {
            GpuRuntime r;
            r.gpu_uuid = g.gpu_uuid;
            out.push_back(r);
        }
        return out;
    }

    std::istringstream ss(raw);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        auto cells = SplitCsvLine(line);
        if (cells.size() < 4) continue;
        GpuRuntime r;
        r.gpu_uuid = cells[0];
        try { r.util_pct = static_cast<int>(std::stod(cells[1])); } catch (...) {}
        try { r.power_w = std::stod(cells[2]); } catch (...) {}
        try { r.temp_c = static_cast<int>(std::stod(cells[3])); } catch (...) {}
        out.push_back(r);
    }
    return out;
}

double CpuUtilPct()
{
    std::ifstream f("/proc/stat");
    if (!f) return -1.0;
    std::string cpu;
    long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0;
    f >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
    const long idle_a = idle + iowait;
    const long total_a = user + nice + system + idle + iowait + irq + softirq;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    f.close();
    f.open("/proc/stat");
    if (!f) return -1.0;
    long user_b = 0, nice_b = 0, system_b = 0, idle_b = 0, iowait_b = 0, irq_b = 0, softirq_b = 0;
    f >> cpu >> user_b >> nice_b >> system_b >> idle_b >> iowait_b >> irq_b >> softirq_b;
    const long idle_delta = (idle_b + iowait_b) - idle_a;
    const long total_delta = (user_b + nice_b + system_b + idle_b + iowait_b + irq_b + softirq_b) - total_a;
    if (total_delta <= 0) return -1.0;
    return 100.0 * (1.0 - static_cast<double>(idle_delta) / static_cast<double>(total_delta));
}

std::string JsonNumberOrNull(double v, bool valid)
{
    if (!valid) return "null";
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

std::string JsonIntOrNull(int v)
{
    if (v < 0) return "null";
    return std::to_string(v);
}

} // namespace

std::string JsonEscape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) break;
                out += static_cast<char>(c);
        }
    }
    return out;
}

std::string GenerateSessionId()
{
    std::array<unsigned char, 16> bytes{};
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (urandom) {
        urandom.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    } else {
        std::random_device rd;
        for (auto& b : bytes) b = static_cast<unsigned char>(rd());
    }
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(32);
    for (unsigned char b : bytes) {
        out.push_back(hex[b >> 4]);
        out.push_back(hex[b & 0x0f]);
    }
    return out;
}

std::string ExtractOperatorLabel(const std::string& user)
{
    const auto dot = user.rfind('.');
    if (dot == std::string::npos || dot + 1 >= user.size()) {
        return "worker";
    }
    return user.substr(dot + 1);
}

std::string BuildStaticHardwareJson(const std::string& miner_version,
                                    const std::vector<int>& active_gpu_indices)
{
    std::string driver;
    std::string cuda_ver;
    DriverAndCuda(driver, cuda_ver);

    const auto gpus = CollectStaticGpus(active_gpu_indices);
    std::ostringstream ss;
    ss << "{";
    ss << "\"cpu_model\":\"" << JsonEscape(CpuModel()) << "\"";
    ss << ",\"cpu_threads_total\":" << CpuThreadsTotal();
    ss << ",\"cpu_threads_allocated\":null";
    ss << ",\"ram_gb_total\":" << RamGbTotal();
    ss << ",\"os\":\"" << JsonEscape(OsString()) << "\"";
    ss << ",\"miner_version\":\"" << JsonEscape(miner_version) << "\"";
    ss << ",\"driver_version\":";
    ss << (driver.empty() ? "null" : ("\"" + JsonEscape(driver) + "\""));
    ss << ",\"cuda_version\":";
    ss << (cuda_ver.empty() ? "null" : ("\"" + JsonEscape(cuda_ver) + "\""));
    ss << ",\"gpus\":[";
    for (size_t i = 0; i < gpus.size(); ++i) {
        const auto& g = gpus[i];
        if (i) ss << ',';
        ss << "{";
        ss << "\"model\":\"" << JsonEscape(g.model) << "\"";
        ss << ",\"vram_gb\":" << g.vram_gb;
        ss << ",\"compute_capability\":";
        ss << (g.compute_capability.empty() ? "null" : ("\"" + JsonEscape(g.compute_capability) + "\""));
        ss << ",\"pcie_link\":";
        ss << (g.pcie_link.empty() ? "null" : ("\"" + JsonEscape(g.pcie_link) + "\""));
        ss << ",\"gpu_uuid\":\"" << JsonEscape(g.gpu_uuid) << "\"";
        ss << "}";
    }
    ss << "]";
    ss << ",\"host_hostname\":\"" << JsonEscape(Hostname()) << "\"";
    ss << ",\"is_containerized\":false";
    ss << ",\"cpu_threads_effective\":null";
    ss << ",\"rental_provider\":null";
    ss << ",\"power_cap_writable\":false";
    ss << ",\"numa\":null";
    ss << ",\"active_backend\":\"cuda\"";
    ss << ",\"cuda_arch_supported\":null";
    ss << "}";
    return ss.str();
}

std::string BuildRuntimeMetricsJson(const std::string& session_id,
                                    double solver_nps,
                                    int shares_session_total)
{
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    const double cpu_util = CpuUtilPct();
    const double ram_used = RamGbUsed();
    const auto gpus = CollectGpuRuntime();

    std::ostringstream ss;
    ss << "{";
    ss << "\"session_id\":\"" << JsonEscape(session_id) << "\"";
    ss << ",\"timestamp\":" << now;
    ss << ",\"cpu_util_pct\":" << JsonNumberOrNull(cpu_util, cpu_util >= 0.0);
    ss << ",\"ram_gb_used\":" << JsonNumberOrNull(ram_used, ram_used > 0.0);
    ss << ",\"gpus\":[";
    for (size_t i = 0; i < gpus.size(); ++i) {
        const auto& g = gpus[i];
        if (i) ss << ',';
        ss << "{";
        ss << "\"gpu_uuid\":\"" << JsonEscape(g.gpu_uuid) << "\"";
        ss << ",\"util_pct\":" << JsonIntOrNull(g.util_pct);
        ss << ",\"power_w\":" << JsonNumberOrNull(g.power_w, g.power_w >= 0.0);
        ss << ",\"temp_c\":" << JsonIntOrNull(g.temp_c);
        ss << "}";
    }
    ss << "]";
    ss << ",\"solver_nps\":" << JsonNumberOrNull(solver_nps, solver_nps > 0.0);
    ss << ",\"shares_session_total\":" << shares_session_total;
    ss << "}";
    return ss.str();
}

std::string HardwareSummaryFromJson(const std::string& hardware_json)
{
    std::ostringstream summary;
    auto extract = [&](const char* key) -> std::string {
        const std::string needle = std::string("\"") + key + "\":\"";
        auto pos = hardware_json.find(needle);
        if (pos == std::string::npos) return {};
        pos += needle.size();
        auto end = hardware_json.find('"', pos);
        if (end == std::string::npos) return {};
        return hardware_json.substr(pos, end - pos);
    };

    const std::string cpu = extract("cpu_model");
    if (!cpu.empty()) summary << "CPU=" << cpu << " ";
    const std::string host = extract("host_hostname");
    if (!host.empty()) summary << "host=" << host << " ";

    auto gpos = hardware_json.find("\"gpus\":[");
    if (gpos != std::string::npos) {
        summary << "GPUs=[";
        size_t pos = gpos + 9;
        bool first = true;
        while (pos < hardware_json.size()) {
            const auto mpos = hardware_json.find("\"model\":\"", pos);
            if (mpos == std::string::npos || mpos > hardware_json.find(']', pos)) break;
            if (!first) summary << ", ";
            first = false;
            pos = mpos + 9;
            auto mend = hardware_json.find('"', pos);
            if (mend == std::string::npos) break;
            summary << hardware_json.substr(pos, mend - pos);
            pos = mend + 1;
        }
        summary << "] ";
    }
    return summary.str();
}

} // namespace common
} // namespace btx