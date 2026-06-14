#pragma once

#include <cstdint>
#include <vector>

#include "pow/uint256_stub.h"

namespace btx {
namespace pow {

// Post-fork height: per-nonce seed_a/seed_b via DeterministicMatMulSeedV2.
constexpr uint32_t kMatMulSeedV2Height = 125000;
// BTX v0.32.10: parent MTP binds seeds at this height and above.
constexpr uint32_t kMatMulSeedV3Height = 130500;

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
    int64_t parent_mtp = 0;
    bool has_parent_mtp = false;
};

// Saturating left-shift of a 32-byte arith_uint256 target (matches btx SaturatingLeftShift256).
std::vector<uint8_t> PreHashTargetShift(const std::vector<uint8_t>& target, uint32_t epsilon_bits);

// BTX v0.32.3 per-nonce seed derivation (height >= kMatMulSeedV2Height).
uint256 DeterministicMatMulSeedV2(
    const uint256& prev_hash,
    int32_t height,
    int32_t version,
    const uint256& merkle_root,
    uint32_t time,
    uint32_t bits,
    uint64_t nonce64,
    uint16_t matmul_dim,
    uint8_t which);

// BTX v0.32.10 per-nonce seed derivation (height >= kMatMulSeedV3Height).
uint256 DeterministicMatMulSeedV3(
    const uint256& prev_hash,
    int64_t parent_mtp,
    int32_t height,
    int32_t version,
    const uint256& merkle_root,
    uint32_t time,
    uint32_t bits,
    uint64_t nonce64,
    uint16_t matmul_dim,
    uint8_t which);

// Pre-hash sigma gate: SHA sigma bytes vs arith-layout block target (MSB-first byte compare).
bool SigmaBelowPreHashTarget(const uint8_t sigma[32], const std::vector<uint8_t>& target_arith);

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
