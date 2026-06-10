#pragma once

#include <cstdint>
#include <string>

#include "pow/uint256_stub.h"

namespace btx {
namespace pow {
namespace field {

using Element = uint32_t;
constexpr Element MODULUS = 0x7FFFFFFFU;  // 2^31 - 1

Element add(Element a, Element b);
Element sub(Element a, Element b);
Element mul(Element a, Element b);
Element neg(Element a);
Element inv(Element a);                 // modular inverse (for tests/verification only)
Element from_uint32(uint32_t x);
Element from_oracle(const uint256& seed, uint32_t index);
Element dot(const Element* a, const Element* b, uint32_t len);

// Internal reduction exposed for tests
Element reduce64_for_test(uint64_t x);

} // namespace field
} // namespace pow
} // namespace btx
