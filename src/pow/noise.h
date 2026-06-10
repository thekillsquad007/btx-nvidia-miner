#pragma once

#include "pow/matrix.h"

#include <cstdint>
#include <string_view>

class uint256;

namespace btx {
namespace pow {
namespace noise {

struct NoisePair {
    Matrix E_L;
    Matrix E_R;
    Matrix F_L;
    Matrix F_R;
};

uint256 DeriveNoiseSeed(std::string_view domain_tag, const uint256& sigma);
NoisePair Generate(const uint256& sigma, uint32_t n, uint32_t r);

} // namespace noise
} // namespace pow
} // namespace btx
