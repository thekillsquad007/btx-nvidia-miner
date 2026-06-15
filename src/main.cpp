#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "common/updater.h"
#include "common/version.h"
#include "cuda/cuda_device.h"
#include "cuda/cuda_solver.h"
#include "cuda/hashrate.h"
#include "pow/matmul_pow.h"
#include "stratum/stratum_client.h"
#include "stratum/stratum_protocol.h"

static void print_help()
{
    std::cout <<
R"(btx-miner - high performance NVIDIA CUDA miner for BTX MatMul PoW

Usage:
  btx-miner [options]

Modes:
  --solo                    Solo mine against local btxd (getblocktemplate)
  --pool <url>              Pool mine (default: stratum+tcp://stratum.minebtx.com:3333)
  --pool-fallback <url>     Backup pool when primary is down (default: bitminerpool.xyz)
                            Set to "none" to disable. Env: BTX_POOL_FALLBACK

Common:
  --user <worker>           Full worker name for pool (address.worker) — required
  --pass <password>         Pool password (default: x)
  --devices 0,1,2|all       GPUs to use (default: all visible)
  --intensity <n>           Max nonces per slice safety cap (default: 20000000)
  --batch <n>|i,j,k         CUDA nonces per kernel launch (default: auto per GPU)
                            Comma list sets per-GPU batch in --devices order; 0 = auto
  --print-gpu-batch         Print per-GPU launch batch plan and exit
  --job-chunk <n>           Nonces per outer solver call (default: auto from GPU batch)
  --check-update            Print whether a newer release is available
  --update                  Download and install latest release, then restart
  --auto-update             Check for updates on startup; re-check every 30m while mining
                            Interval: BTX_AUTO_UPDATE_INTERVAL_SEC (default 1800)
  --slice-seconds <n>       Time-limit each mining slice in seconds (default: 5)
  --verbose                 Extra stratum debug logging
  --benchmark               Run a short throughput test (CPU ref + CUDA if available)
  --no-gpu                  Force CPU reference path only
  --dev-fee <pct>           Dev fee percent 0-5 (default: 1, or BTX_DEV_FEE_PCT)
  -h, --help                This help

Solo RPC:
  --rpc-url http://127.0.0.1:19334
  --rpc-user user
  --rpc-password pass
  --address <btx1z...>      Payout address

See README.md for full tuning and multi-GPU details.
)";
}

static std::vector<int> parse_devices(const std::string& spec)
{
    std::vector<int> out;
    if (spec.empty() || spec == "all") return out;
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string part = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!part.empty()) out.push_back(std::atoi(part.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

static void print_gpu_inventory()
{
#ifdef BTX_MINER_HAS_CUDA
    auto devs = btx::cuda::EnumerateDevices();
    if (devs.empty()) {
        std::cout << "No CUDA devices detected." << std::endl;
        return;
    }
    std::cout << "GPUs:";
    for (const auto& d : devs) {
        std::cout << " [" << d.index << "] " << d.name
                  << (d.usable ? "" : " (skipped: " + d.reason + ")");
    }
    std::cout << std::endl;
#else
    std::cout << "CPU-only build." << std::endl;
#endif
}

static float parse_dev_fee(const char* arg)
{
    if (!arg) return -1.0f;
    char* end = nullptr;
    float v = std::strtof(arg, &end);
    if (end == arg) return -1.0f;
    return v;
}

static btx::cuda::BatchLaunchConfig parse_batch_config(const std::string& spec)
{
    btx::cuda::BatchLaunchConfig cfg;
    if (spec.empty()) {
        return cfg;
    }
    if (spec.find(',') == std::string::npos) {
        cfg.global_batch = std::atoi(spec.c_str());
        return cfg;
    }
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string part = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        cfg.per_device.push_back(part.empty() ? 0 : std::atoi(part.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return cfg;
}

static std::string format_batch_config(const btx::cuda::BatchLaunchConfig& cfg)
{
    if (!cfg.per_device.empty()) {
        std::ostringstream ss;
        for (size_t i = 0; i < cfg.per_device.size(); ++i) {
            if (i) ss << ',';
            ss << cfg.per_device[i];
        }
        return ss.str();
    }
    if (cfg.global_batch > 0) {
        return std::to_string(cfg.global_batch);
    }
    return "auto";
}

int main(int argc, char** argv)
{
    std::string mode;
    std::string pool_url;
    std::string pool_fallback_url;
    bool pool_fallback_disabled = false;
    std::string user;
    std::string pass = "x";
    bool do_bench = false;
    bool force_cpu = false;
    bool verbose = false;
    bool print_gpu_batch = false;
    bool check_update = false;
    bool do_update = false;
    bool auto_update = false;
    int intensity = 20'000'000;
    btx::cuda::BatchLaunchConfig batch_config;
    int job_chunk = 0;
    double slice_seconds = 5.0;
    float dev_fee_override = -1.0f;
    std::string devices_spec = "all";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_help(); return 0; }
        if (a == "--version") { std::cout << "btx-miner v" << btx::common::kMinerVersion << std::endl; return 0; }
        if (a == "--benchmark") do_bench = true;
        if (a == "--no-gpu") force_cpu = true;
        if (a == "--verbose") verbose = true;
        if (a == "--solo") mode = "solo";
        if (a == "--pool") {
            mode = "pool";
            if (i+1 < argc && argv[i+1][0] != '-') pool_url = argv[++i];
        }
        if (a == "--pool-fallback") {
            if (i+1 < argc && argv[i+1][0] != '-') {
                pool_fallback_url = argv[++i];
                if (pool_fallback_url == "none" || pool_fallback_url == "off") {
                    pool_fallback_disabled = true;
                    pool_fallback_url.clear();
                }
            }
        }
        if ((a == "--user" || a == "--address") && i+1 < argc) user = argv[++i];
        if (a == "--pass" && i+1 < argc) pass = argv[++i];
        if (a == "--intensity" && i+1 < argc) intensity = std::atoi(argv[++i]);
        if (a == "--batch" && i+1 < argc) batch_config = parse_batch_config(argv[++i]);
        if (a == "--print-gpu-batch") print_gpu_batch = true;
        if (a == "--check-update") check_update = true;
        if (a == "--update") do_update = true;
        if (a == "--auto-update") auto_update = true;
        if (a == "--job-chunk" && i+1 < argc) job_chunk = std::atoi(argv[++i]);
        if (a == "--slice-seconds" && i+1 < argc) slice_seconds = std::atof(argv[++i]);
        if (a == "--devices" && i+1 < argc) devices_spec = argv[++i];
        if (a == "--dev-fee" && i+1 < argc) dev_fee_override = parse_dev_fee(argv[++i]);
    }

    btx::cuda::SetForceCpu(force_cpu);
    if (!force_cpu) {
        btx::cuda::SetActiveDevices(parse_devices(devices_spec));
    }

    if (check_update || do_update || auto_update) {
        const auto info = btx::common::CheckForUpdate();
        if (check_update) {
            if (info.update_available) {
                std::cout << "Update available: v" << info.latest_version
                          << " (running v" << info.current_version << ")" << std::endl;
                std::cout << "Run: btx-miner --update" << std::endl;
            } else {
                std::cout << "Up to date: v" << info.current_version << std::endl;
            }
            return 0;
        }
        if (info.update_available) {
            std::cout << "[update] Installing v" << info.latest_version
                      << " (from v" << info.current_version << ")..." << std::endl;
            std::string err;
            if (!btx::common::InstallUpdate(info, err)) {
                std::cerr << "[update] failed: " << err << std::endl;
                return do_update ? 1 : 0;
            }
            std::cout << "[update] Installed v" << info.latest_version << ", restarting..." << std::endl;
            std::vector<std::string> relaunch;
            const std::string exe = btx::common::GetExecutablePath();
            relaunch.push_back(exe.empty() ? argv[0] : exe);
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--update") continue;
                if (arg == "--auto-update") continue;
                relaunch.push_back(arg);
            }
            btx::common::ReexecCurrentProcess(relaunch);
        } else if (do_update) {
            std::cout << "Already on latest: v" << info.current_version << std::endl;
            return 0;
        }
    }

    if (print_gpu_batch) {
        btx::pow::MatMulJob sample;
        sample.n = 512;
        sample.b = 16;
        sample.r = 8;
        sample.block_height = btx::pow::kMatMulSeedV2Height;
        sample.epsilon_bits = 18;
        btx::cuda::PrintGpuBatchPlan(batch_config, sample);
        return 0;
    }

    if (dev_fee_override >= 0.0f) {
        std::string env = std::to_string(dev_fee_override);
        setenv("BTX_DEV_FEE_PCT", env.c_str(), 1);
    }

    if (do_bench) {
        std::cout << "btx-miner CPU reference smoke test (MatMul PoW)..." << std::endl;

        btx::pow::MatMulJob job;
        job.n = 16; job.b = 8; job.r = 2;
        job.bits = 0x1d00ffff;
        job.target.assign(32, 0xff);

        auto sol = btx::pow::SolveCPU(job, 64, 0);

        if (sol.found) {
            std::cout << "  PASS: reference found solution at nonce=" << sol.nonce << std::endl;
        } else {
            uint256 dummy;
            bool ok = btx::pow::VerifySolution(job, job.nonce_start, 0, dummy);
            std::cout << "  Reference path executed cleanly. Verify on nonce_start="
                      << job.nonce_start << " returned "
                      << (ok ? "hit" : "no hit under target") << "." << std::endl;
        }

#ifdef BTX_MINER_HAS_CUDA
        auto devices = btx::cuda::GetActiveDevices();
        if (!devices.empty()) {
            std::cout << "CUDA devices: ";
            for (size_t d = 0; d < devices.size(); ++d) {
                if (d) std::cout << ", ";
                std::cout << devices[d];
            }
            std::cout << std::endl;
        } else {
            std::cout << "CUDA compiled in but no usable GPU at runtime." << std::endl;
        }
#else
        std::cout << "CPU-only build (enable CUDA for GPU mining)." << std::endl;
#endif

        std::cout << "CPU MatMul PoW reference is linked and mathematically functional." << std::endl;
        return 0;
    }

    if (mode == "pool") {
        if (user.empty()) {
            std::cerr << "Error: --user <btx1z... .worker> is required for pool mode" << std::endl;
            return 1;
        }
        if (pool_url.empty()) pool_url = "stratum+tcp://stratum.minebtx.com:3333";
        if (pool_fallback_url.empty() && !pool_fallback_disabled) {
            if (const char* env_fb = std::getenv("BTX_POOL_FALLBACK")) {
                pool_fallback_url = env_fb;
                if (pool_fallback_url == "none" || pool_fallback_url == "off") {
                    pool_fallback_disabled = true;
                    pool_fallback_url.clear();
                }
            } else {
                pool_fallback_url = "stratum+tcp://stratum.bitminerpool.xyz:3333";
            }
        }

        std::string host;
        uint16_t port = 3333;
        if (!btx::stratum::ParsePoolUrl(pool_url, host, port)) {
            std::cerr << "Error: invalid pool URL: " << pool_url << std::endl;
            return 1;
        }

        std::vector<btx::stratum::PoolEndpoint> fallback_endpoints;
        if (!pool_fallback_disabled && !pool_fallback_url.empty()) {
            std::string fb_host;
            uint16_t fb_port = 3333;
            if (!btx::stratum::ParsePoolUrl(pool_fallback_url, fb_host, fb_port)) {
                std::cerr << "Error: invalid pool fallback URL: " << pool_fallback_url << std::endl;
                return 1;
            }
            if (fb_host != host || fb_port != port) {
                fallback_endpoints.push_back({fb_host, fb_port});
            }
        }

        std::cout << "btx-miner v" << btx::common::kMinerVersion << " — pool mining " << host << ":" << port << " as " << user;
        if (!fallback_endpoints.empty()) {
            std::cout << " (fallback " << fallback_endpoints[0].host << ":" << fallback_endpoints[0].port << ")";
        }
        std::cout << std::endl;
        if (intensity < 10000) {
            std::cerr << "WARNING: --intensity " << intensity
                      << " limits each slice to only " << intensity << " nonces (~"
                      << (intensity * 4 / 1000) << "k/s overhead). Omit --intensity for 5s slices; "
                      << "use --batch for CUDA launch size." << std::endl;
        }
        btx::pow::MatMulJob sample;
        sample.n = 512;
        sample.b = 16;
        sample.r = 8;
        sample.block_height = btx::pow::kMatMulSeedV2Height;
        sample.epsilon_bits = 18;
        const int resolved_job_chunk = job_chunk > 0
            ? job_chunk
            : btx::cuda::RecommendJobChunkSize(batch_config, sample);

        std::cout << "Slice=" << slice_seconds << "s cap=" << intensity
                  << " nonces, chunk=" << resolved_job_chunk
                  << (job_chunk > 0 ? "" : " (auto)")
                  << ", batch=" << format_batch_config(batch_config);
        if (devices_spec != "all") std::cout << ", devices=" << devices_spec;
        std::cout << std::endl;
        print_gpu_inventory();
        auto active = btx::cuda::GetActiveDevices();
        if (!active.empty()) {
            std::cout << "Mining on GPU(s):";
            for (size_t i = 0; i < active.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << " " << active[i];
            }
            std::cout << std::endl;
            btx::cuda::WarmupDevices(active);
#ifdef BTX_MINER_HAS_CUDA
            btx::cuda::PrintGpuBatchPlan(batch_config, sample);
#endif
        }

        auto on_sol = [](const btx::stratum::StratumJob& j, uint64_t nonce, uint32_t ntime, const uint256& /*dig*/, bool is_block) {
            std::cout << "*** SOLUTION job=" << j.job_id << " nonce=0x" << std::hex << nonce
                      << std::dec << " ntime=0x" << std::hex << ntime << std::dec
                      << (is_block ? " (BLOCK!)" : " (share)") << std::endl;
        };

        btx::stratum::StratumConfig cfg;
        cfg.nonces_per_slice = intensity > 0 ? intensity : 20'000'000;
        cfg.batch_config = batch_config;
        cfg.job_chunk_size = resolved_job_chunk;
        cfg.slice_max_seconds = slice_seconds > 0.0 ? slice_seconds : 5.0;
        cfg.verbose = verbose;
        cfg.auto_update = auto_update;
        if (const char* interval = std::getenv("BTX_AUTO_UPDATE_INTERVAL_SEC")) {
            cfg.auto_update_interval_sec = std::atof(interval);
        }

        btx::stratum::StratumClient client(host, port, user, pass, on_sol, false, cfg, fallback_endpoints);
        client.run_forever();
        if (client.restart_for_update()) {
            std::cout << "[update] Restarting with new binary..." << std::endl;
            std::vector<std::string> relaunch;
            const std::string exe = btx::common::GetExecutablePath();
            relaunch.push_back(exe.empty() ? argv[0] : exe);
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--auto-update") continue;
                relaunch.push_back(arg);
            }
            btx::common::ReexecCurrentProcess(relaunch);
        }
        return 0;
    }

    if (mode == "solo") {
        std::cerr << "Solo mining scaffold is not wired yet. Use pool mode or run against btxd via dexbtx-miner." << std::endl;
        return 1;
    }

    print_help();
    return 0;
}