#include "pow/field.h"

#include "pow/sha256_portable.h"  // for from_oracle when needed

#include <cassert>
#include <cstdint>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace btx {
namespace pow {
namespace field {

static_assert(sizeof(Element) == 4, "Element must be 32-bit");

namespace {

// Double Mersenne fold reduction for q = 2^31-1. Safe for all uint64 inputs.
static Element reduce64(uint64_t x)
{
    const uint64_t fold1 = (x & static_cast<uint64_t>(MODULUS)) + (x >> 31);
    const uint32_t lo = static_cast<uint32_t>(fold1 & MODULUS);
    const uint32_t hi = static_cast<uint32_t>(fold1 >> 31);
    uint32_t result = lo + hi;
    if (result >= MODULUS) result -= MODULUS;
    return result;
}

Element ScalarDot(const Element* a, const Element* b, uint32_t len)
{
    constexpr uint32_t REDUCE_INTERVAL = 4;
    uint64_t acc = 0;
    uint32_t pending = 0;
    for (uint32_t i = 0; i < len; ++i) {
        acc += static_cast<uint64_t>(a[i]) * b[i];
        if (++pending == REDUCE_INTERVAL) {
            acc = reduce64(acc);
            pending = 0;
        }
    }
    return reduce64(acc);
}

#if defined(__ARM_NEON)
Element NeonDot(const Element* a, const Element* b, uint32_t len)
{
    uint64_t acc = 0;
    uint32_t i = 0;
    for (; i + 4 <= len; i += 4) {
        const uint32x4_t va = vld1q_u32(a + i);
        const uint32x4_t vb = vld1q_u32(b + i);
        const uint64x2_t prod_lo = vmull_u32(vget_low_u32(va), vget_low_u32(vb));
        const uint64x2_t prod_hi = vmull_u32(vget_high_u32(va), vget_high_u32(vb));
        acc += vgetq_lane_u64(prod_lo, 0);
        acc += vgetq_lane_u64(prod_lo, 1);
        acc += vgetq_lane_u64(prod_hi, 0);
        acc += vgetq_lane_u64(prod_hi, 1);
        acc = reduce64(acc);
    }
    return add(reduce64(acc), ScalarDot(a + i, b + i, len - i));
}
#endif

} // namespace

Element add(Element a, Element b)
{
    assert(a < MODULUS && b < MODULUS);
    uint32_t s = a + b;
    if (s >= MODULUS) s -= MODULUS;
    return s;
}

Element sub(Element a, Element b)
{
    assert(a < MODULUS && b < MODULUS);
    if (a >= b) return a - b;
    return a + MODULUS - b;
}

Element mul(Element a, Element b)
{
    assert(a < MODULUS && b < MODULUS);
    return reduce64(static_cast<uint64_t>(a) * b);
}

Element neg(Element a)
{
    if (a == 0) return 0;
    return MODULUS - a;
}

Element from_uint32(uint32_t x)
{
    return reduce64(x);
}

Element inv(Element a)
{
    assert(a != 0);
    // Fermat: a^(q-2) mod q
    uint32_t exp = MODULUS - 2;
    Element result = 1;
    Element base = a;
    while (exp > 0) {
        if (exp & 1) result = mul(result, base);
        exp >>= 1;
        if (exp) base = mul(base, base);
    }
    return result;
}

Element from_oracle(const uint256& seed, uint32_t index)
{
    // Replicates btxchain/btx matmul/field.cpp exactly (little-endian seed bytes, LE index, rejection sampling)
    uint8_t seed_bytes[32];
    for (size_t i = 0; i < 32; ++i) {
        seed_bytes[i] = seed.data()[31 - i];  // uint256 is big-endian internal in many impls; match the node's ToCanonicalBytes
    }

    for (uint32_t retry = 0; retry < 256; ++retry) {
        // Use portable SHA256
        sha256_state st;
        sha256_init(&st);
        sha256_update(&st, seed_bytes, 32);

        uint8_t idx_le[4];
        idx_le[0] = index & 0xff;
        idx_le[1] = (index >> 8) & 0xff;
        idx_le[2] = (index >> 16) & 0xff;
        idx_le[3] = (index >> 24) & 0xff;
        sha256_update(&st, idx_le, 4);

        if (retry > 0) {
            uint8_t r_le[4];
            r_le[0] = retry & 0xff; r_le[1] = (retry>>8)&0xff; r_le[2]=(retry>>16)&0xff; r_le[3]=(retry>>24)&0xff;
            sha256_update(&st, r_le, 4);
        }

        uint8_t hash[32];
        sha256_final(&st, hash);

        const uint32_t candidate =
            (static_cast<uint32_t>(hash[0])) |
            (static_cast<uint32_t>(hash[1]) << 8) |
            (static_cast<uint32_t>(hash[2]) << 16) |
            (static_cast<uint32_t>(hash[3]) << 24);
        const uint32_t masked = candidate & MODULUS;
        if (masked < MODULUS) {
            return masked;
        }
    }

    // Fallback (extremely rare)
    sha256_state fb;
    sha256_init(&fb);
    sha256_update(&fb, seed_bytes, 32);
    uint8_t idx_le[4] = { (uint8_t)(index&0xff), (uint8_t)((index>>8)&0xff), (uint8_t)((index>>16)&0xff), (uint8_t)((index>>24)&0xff) };
    sha256_update(&fb, idx_le, 4);
    const char* tag = "oracle-fallback";
    sha256_update(&fb, (const uint8_t*)tag, 15);
    uint8_t fbh[32];
    sha256_final(&fb, fbh);
    const uint32_t v =
        (static_cast<uint32_t>(fbh[0])) |
        (static_cast<uint32_t>(fbh[1]) << 8) |
        (static_cast<uint32_t>(fbh[2]) << 16) |
        (static_cast<uint32_t>(fbh[3]) << 24);
    return v % MODULUS;
}

Element dot(const Element* a, const Element* b, uint32_t len)
{
#if defined(__ARM_NEON)
    return NeonDot(a, b, len);
#else
    return ScalarDot(a, b, len);
#endif
}

Element reduce64_for_test(uint64_t x)
{
    return reduce64(x);
}

} // namespace field
} // namespace pow
} // namespace btx
