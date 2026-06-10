#include <cstdlib>
#include <iostream>
#include <string>

// Bump when making pool/stratum fixes so rigs can verify they got the latest build.
static constexpr const char* kMinerVersion = "0.2.8";

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
  --pool <url>              Pool mine (e.g. stratum+tcp://stratum.minebtx.com:3333)

Common:
  --user <worker>           Full worker name for pool (address.worker) — required
  --pass <password>         Pool password (default: x)
  --devices 0,1,2|all       GPUs to use (default: all visible)
  --intensity <n>           Nonces per work slice (default: 512)
  --batch <n>               CUDA nonces per kernel launch (default: 128)
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

int main(int argc, char** argv)
{
    std::string mode;
    std::string pool_url;
    std::string user;
    std::string pass = "x";
    bool do_bench = false;
    bool force_cpu = false;
    bool verbose = false;
    int intensity = 512;
    int batch = 128;
    float dev_fee_override = -1.0f;
    std::string devices_spec = "all";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_help(); return 0; }
        if (a == "--version") { std::cout << "btx-miner v" << kMinerVersion << std::endl; return 0; }
        if (a == "--benchmark") do_bench = true;
        if (a == "--no-gpu") force_cpu = true;
        if (a == "--verbose") verbose = true;
        if (a == "--solo") mode = "solo";
        if (a == "--pool") {
            mode = "pool";
            if (i+1 < argc && argv[i+1][0] != '-') pool_url = argv[++i];
        }
        if ((a == "--user" || a == "--address") && i+1 < argc) user = argv[++i];
        if (a == "--pass" && i+1 < argc) pass = argv[++i];
        if (a == "--intensity" && i+1 < argc) intensity = std::atoi(argv[++i]);
        if (a == "--batch" && i+1 < argc) batch = std::atoi(argv[++i]);
        if (a == "--devices" && i+1 < argc) devices_spec = argv[++i];
        if (a == "--dev-fee" && i+1 < argc) dev_fee_override = parse_dev_fee(argv[++i]);
    }

    btx::cuda::SetForceCpu(force_cpu);
    if (!force_cpu) {
        btx::cuda::SetActiveDevices(parse_devices(devices_spec));
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

        std::string host;
        uint16_t port = 3333;
        if (!btx::stratum::ParsePoolUrl(pool_url, host, port)) {
            std::cerr << "Error: invalid pool URL: " << pool_url << std::endl;
            return 1;
        }

        std::cout << "btx-miner v" << kMinerVersion << " — pool mining " << host << ":" << port << " as " << user << std::endl;
        std::cout << "Intensity=" << intensity << " nonces/slice, batch=" << batch;
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
        }

        auto on_sol = [](const btx::stratum::StratumJob& j, uint64_t nonce, uint32_t ntime, const uint256& /*dig*/, bool is_block) {
            std::cout << "*** SOLUTION job=" << j.job_id << " nonce=0x" << std::hex << nonce
                      << std::dec << " ntime=0x" << std::hex << ntime << std::dec
                      << (is_block ? " (BLOCK!)" : " (share)") << std::endl;
        };

        btx::stratum::StratumConfig cfg;
        cfg.nonces_per_slice = intensity > 0 ? intensity : 512;
        cfg.max_batch_size = batch > 0 ? batch : 128;
        cfg.verbose = verbose;

        btx::stratum::StratumClient client(host, port, user, pass, on_sol, false, cfg);
        client.run_forever();
        return 0;
    }

    if (mode == "solo") {
        std::cerr << "Solo mining scaffold is not wired yet. Use pool mode or run against btxd via dexbtx-miner." << std::endl;
        return 1;
    }

    print_help();
    return 0;
}