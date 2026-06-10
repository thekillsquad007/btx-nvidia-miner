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
    std::string target;         // share target (or block target)
    std::string share_target;   // alias for compatibility
    bool clean_jobs = false;
    bool clean = false;         // alias
    std::string seed_a;
    std::string seed_b;
    uint32_t block_height = 0;
    uint32_t matmul_n = 512;
    uint32_t matmul_b = 16;
    uint32_t matmul_r = 8;
    int epsilon_bits = 18;
    uint64_t nonce64_start = 0;
    // received_at etc. can be added if needed
};

// Callback when a solution is found by the solver for a job.
using SolutionCallback = std::function<void(const StratumJob& job, uint64_t nonce, uint32_t ntime, const uint256& digest, bool is_block)>;

struct StratumConfig {
    int nonces_per_slice = 256;
    int max_batch_size = 64;
    bool verbose = false;
};

class StratumClient {
public:
    StratumClient(const std::string& host, uint16_t port,
                  const std::string& user, const std::string& pass,
                  SolutionCallback on_solution,
                  bool use_tls = false,
                  StratumConfig config = {});

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
