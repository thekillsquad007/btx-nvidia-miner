#include "pow/transcript.h"

#include "pow/field.h"
#include "pow/sha256_portable.h"

#include <cassert>
#include <vector>

namespace btx {
namespace pow {
namespace transcript {

// These are only used by the full node-compatible path. The miner reference
// implementation lives in matmul_pow.cpp (self-contained VerifySolution) so that
// we can get a compiling skeleton quickly. The functions below are minimal
// stubs so the build succeeds; they will be filled with the exact logic later
// or the callers will use the matmul_pow entry points.

std::vector<field::Element> DeriveCompressionVector(const uint256& /*sigma*/, uint32_t b)
{
    // Return a simple linear vector for stub purposes (not consensus).
    std::vector<field::Element> v(static_cast<size_t>(b) * b);
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<field::Element>(i + 1);
    return v;
}

field::Element CompressBlock(const Matrix& /*block_bb*/, const std::vector<field::Element>& /*v*/)
{
    return 0;
}

CanonicalResult CanonicalMatMul(const Matrix& A_prime, const Matrix& B_prime, uint32_t b, const uint256& /*sigma*/)
{
    // Not used in the current miner reference loop.
    (void)A_prime; (void)B_prime; (void)b;
    CanonicalResult r;
    // Cannot default-construct Matrix easily here without more work; return a dummy sized result.
    r.C_prime = Matrix(1, 1);
    return r;
}

} // namespace transcript
} // namespace pow
} // namespace btx
