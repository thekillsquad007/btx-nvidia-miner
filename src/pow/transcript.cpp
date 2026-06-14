#include "pow/transcript.h"

#include "pow/field.h"
#include "pow/sha256_portable.h"
#include "pow/uint256_stub.h"

#include <array>
#include <cassert>
#include <cstring>
#include <utility>
#include <vector>

namespace btx {
namespace pow {
namespace transcript {

namespace {

std::array<uint8_t, 32> ToCanonicalBytes(const uint256& value)
{
    std::array<uint8_t, 32> out;
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = value.data()[out.size() - 1 - i];
    }
    return out;
}

uint256 CanonicalBytesToUint256(const uint8_t bytes[32])
{
    uint256 out;
    for (size_t i = 0; i < 32; ++i) {
        out.data()[i] = bytes[31 - i];
    }
    return out;
}

uint256 DeriveCompressionSeed(const uint256& sigma)
{
    const auto sigma_bytes = ToCanonicalBytes(sigma);

    sha256_state hasher;
    sha256_init(&hasher);
    static const char kTag[] = "matmul-compress-v1";
    sha256_update(&hasher, reinterpret_cast<const uint8_t*>(kTag), sizeof(kTag) - 1);
    sha256_update(&hasher, sigma_bytes.data(), sigma_bytes.size());

    uint8_t digest[32];
    sha256_final(&hasher, digest);
    return CanonicalBytesToUint256(digest);
}

void WriteLE32(uint8_t out[4], uint32_t value)
{
    out[0] = static_cast<uint8_t>(value & 0xff);
    out[1] = static_cast<uint8_t>((value >> 8) & 0xff);
    out[2] = static_cast<uint8_t>((value >> 16) & 0xff);
    out[3] = static_cast<uint8_t>((value >> 24) & 0xff);
}

} // namespace

std::vector<field::Element> DeriveCompressionVector(const uint256& sigma, uint32_t b)
{
    const uint256 seed = DeriveCompressionSeed(sigma);
    const uint64_t len = static_cast<uint64_t>(b) * b;
    std::vector<field::Element> vec;
    vec.reserve(static_cast<size_t>(len));
    for (uint64_t k = 0; k < len; ++k) {
        vec.push_back(field::from_oracle(seed, static_cast<uint32_t>(k)));
    }
    return vec;
}

field::Element CompressBlock(const Matrix& block_bb, const std::vector<field::Element>& v)
{
    const uint64_t len = static_cast<uint64_t>(block_bb.rows()) * block_bb.cols();
    assert(len == v.size());
    return field::dot(block_bb.data(), v.data(), static_cast<uint32_t>(len));
}

CanonicalResult CanonicalMatMul(const Matrix& A_prime, const Matrix& B_prime, uint32_t b, const uint256& sigma)
{
    assert(A_prime.rows() == A_prime.cols());
    assert(B_prime.rows() == B_prime.cols());
    assert(A_prime.rows() == B_prime.rows());
    assert(b != 0 && (A_prime.rows() % b) == 0);

    const uint32_t n = A_prime.rows();
    const uint32_t blocks_per_axis = n / b;
    const auto compress_vec = DeriveCompressionVector(sigma, b);

    Matrix c_prime(n, n);
    sha256_state hasher;
    sha256_init(&hasher);

    for (uint32_t i = 0; i < blocks_per_axis; ++i) {
        for (uint32_t j = 0; j < blocks_per_axis; ++j) {
            for (uint32_t ell = 0; ell < blocks_per_axis; ++ell) {
                const Matrix product = A_prime.block(i, ell, b) * B_prime.block(ell, j, b);
                Matrix c_block = c_prime.block(i, j, b);
                c_block = c_block + product;
                c_prime.set_block(i, j, b, c_block);

                const field::Element compressed = CompressBlock(c_block, compress_vec);
                uint8_t bytes[4];
                WriteLE32(bytes, compressed);
                sha256_update(&hasher, bytes, sizeof(bytes));
            }
        }
    }

    uint8_t inner[32];
    sha256_final(&hasher, inner);
    sha256_state outer;
    sha256_init(&outer);
    sha256_update(&outer, inner, sizeof(inner));
    uint8_t digest[32];
    sha256_final(&outer, digest);

    CanonicalResult r;
    r.C_prime = std::move(c_prime);
    r.transcript_hash = uint256(digest);
    return r;
}

} // namespace transcript
} // namespace pow
} // namespace btx
