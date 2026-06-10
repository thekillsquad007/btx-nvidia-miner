#pragma once

#include "pow/matrix.h"

#include <cstdint>
#include <vector>

class uint256;

namespace btx {
namespace pow {
namespace transcript {

std::vector<field::Element> DeriveCompressionVector(const uint256& sigma, uint32_t b);

struct CanonicalResult {
    Matrix C_prime{};
    uint256 transcript_hash{};
};

// The core algorithm: computes C' = A' * B' in b x b blocks while feeding the *running* accumulated block
// into the transcript hasher (exactly as the node does).
CanonicalResult CanonicalMatMul(const Matrix& A_prime, const Matrix& B_prime, uint32_t b, const uint256& sigma);

} // namespace transcript
} // namespace pow
} // namespace btx
