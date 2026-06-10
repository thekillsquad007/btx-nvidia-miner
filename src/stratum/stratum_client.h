#pragma once

#include "pow/matmul_pow.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace btx {
namespace stratum {

// Job from mining.notify (matmul extended)
struct StratumJob {
    std::string job_id;
    int version = 0;
    std::string prev_hash;
    std::string merkle_root;
    uint32_t time = 0;
    std::string bits;
    std::string share_target;   // easier target for shares
    bool clean = false;
    std::string seed_a;
    std::string seed_b;
    uint32_t block_height = 0;
    uint32_t matmul_n = 512;
    uint32_t matmul_b = 16;
    uint32_t matmul_r = 8;
    uint64_t nonce64_start = 0;
};

// Callback when a solution is found by the solver for a job.
using SolutionCallback = std::function<void(const StratumJob& job, uint64_t nonce, uint32_t ntime, const uint256& digest, bool is_block)>;

class StratumClient {
public:
    StratumClient(const std::string& host, uint16_t port,
                  const std::string& user, const std::string& pass,
                  SolutionCallback on_solution,
                  bool use_tls = false);

    ~StratumClient();

    void run_forever();   // blocks, reconnects on error

    void stop();

private:
    // Implementation details (sockets, reader, solver loop) in .cpp
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace stratum
} // namespace btx
