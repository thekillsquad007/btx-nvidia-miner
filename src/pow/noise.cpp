#include "pow/noise.h"

#include "pow/field.h"
#include "pow/matrix.h"
#include "pow/sha256_portable.h"

#include <array>
#include <cassert>
#include <cstdint>

namespace btx {
namespace pow {
namespace noise {

namespace {

std::array<uint8_t, 32> ToCanonicalBytes(const uint256& value)
{
    std::array<uint8_t, 32> out;
    for (size_t i = 0; i < 32; ++i) {
        out[i] = value.data()[31 - i];
    }
    return out;
}

Matrix FromSeedRect(const uint256& seed, uint32_t rows, uint32_t cols)
{
    Matrix out(rows, cols);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            out.at(r, c) = field::from_oracle(seed, r * cols + c);
        }
    }
    return out;
}

} // namespace

uint256 DeriveNoiseSeed(std::string_view domain_tag, const uint256& sigma)
{
    assert(domain_tag.size() == 18); // "matmul_noise_XX_v1"
    auto sigma_bytes = ToCanonicalBytes(sigma);

    sha256_state st;
    sha256_init(&st);
    sha256_update(&st, reinterpret_cast<const uint8_t*>(domain_tag.data()), domain_tag.size());
    sha256_update(&st, sigma_bytes.data(), 32);
    uint8_t digest[32];
    sha256_final(&st, digest);

    // Match the node's CanonicalBytesToUint256 exactly:
    // Store reversed so that from_oracle's internal swap recovers `digest` as the PRF seed bytes.
    uint256 out;
    for (size_t i = 0; i < 32; ++i) {
        out.data()[i] = digest[31 - i];
    }
    return out;
}

NoisePair Generate(const uint256& sigma, uint32_t n, uint32_t r)
{
    // Tags from the spec
    static constexpr std::string_view TAG_EL{"matmul_noise_EL_v1"};
    static constexpr std::string_view TAG_ER{"matmul_noise_ER_v1"};
    static constexpr std::string_view TAG_FL{"matmul_noise_FL_v1"};
    static constexpr std::string_view TAG_FR{"matmul_noise_FR_v1"};

    const uint256 tag_el = DeriveNoiseSeed(TAG_EL, sigma);
    const uint256 tag_er = DeriveNoiseSeed(TAG_ER, sigma);
    const uint256 tag_fl = DeriveNoiseSeed(TAG_FL, sigma);
    const uint256 tag_fr = DeriveNoiseSeed(TAG_FR, sigma);

    return {
        .E_L = FromSeedRect(tag_el, n, r),
        .E_R = FromSeedRect(tag_er, r, n),
        .F_L = FromSeedRect(tag_fl, n, r),
        .F_R = FromSeedRect(tag_fr, r, n),
    };
}

} // namespace noise
} // namespace pow
} // namespace btx
