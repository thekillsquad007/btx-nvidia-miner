#pragma once

#include <cstdint>
#include <vector>

#include "pow/uint256_stub.h"

namespace btx {
namespace pow {

// Minimal job description for both solo and stratum.
struct MatMulJob {
    uint32_t n = 512;
    uint32_t b = 16;
    uint32_t r = 8;
    uint256 seed_a;
    uint256 seed_b;
    // Difficulty target for the transcript hash (256-bit BE or as interpreted by node).
    // For solo this is the block target; for pool it can be a share target (easier).
    std::vector<uint8_t> target; // 32 bytes, big-endian comparison

    // Extra context needed for sigma derivation (header fields)
    int32_t version = 1;
    uint256 prev_hash;
    uint256 merkle_root;
    uint32_t time = 0;
    uint32_t bits = 0;           // compact bits for the *block* (important for sigma)
    uint64_t nonce_start = 0;    // where to begin scanning in this slice
};

// Result of a solve attempt.
struct MatMulSolution {
    bool found = false;
    uint64_t nonce = 0;
    uint32_t ntime = 0;          // the timestamp used (may be updated)
    uint256 digest;              // the successful matmul_digest
    bool meets_block_target = false;
};

// CPU reference implementation. Must match btxchain/btx exactly for the given header fields + seeds + nonce.
bool VerifySolution(const MatMulJob& job, uint64_t nonce, uint32_t ntime, uint256& out_digest);

// True when digest <= target using the node's UintToArith256 ordering.
bool DigestMeetsTarget(const uint256& digest, const std::vector<uint8_t>& target);

// Scan up to max_tries nonces starting at job.nonce_start.
// Returns on first solution that meets the job.target (share or block).
MatMulSolution SolveCPU(const MatMulJob& job, uint64_t max_tries, uint32_t ntime = 0);

} // namespace pow
} // namespace btx
