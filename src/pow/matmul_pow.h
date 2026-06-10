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
    // Share difficulty from pool notify (digest must be <= this).
    std::vector<uint8_t> target;
    // Block target derived from header nBits — used only for the pre-hash sigma gate.
    std::vector<uint8_t> block_target;

    // Extra context needed for sigma derivation (header fields)
    int32_t version = 1;
    uint256 prev_hash;
    uint256 merkle_root;
    uint32_t time = 0;
    uint32_t bits = 0;           // compact bits for the *block* (important for sigma)
    uint64_t nonce_start = 0;    // where to begin scanning in this slice
    uint32_t block_height = 0;
    uint32_t epsilon_bits = 0;   // pre-hash sigma gate: sigma must be <= target << N
};

// Saturating left-shift of a 32-byte arith_uint256 target (matches btx SaturatingLeftShift256).
std::vector<uint8_t> PreHashTargetShift(const std::vector<uint8_t>& target, uint32_t epsilon_bits);

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
