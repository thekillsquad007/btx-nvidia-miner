#pragma once

#include <cstddef>
#include <cstdint>

// Very small public-domain style SHA256 implementation sufficient for the reference PoW path.
// Matches the usage pattern in the BTX node (Update + Finalize 32-byte digest).

struct sha256_state {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t data[64];
    uint32_t datalen;
};

void sha256_init(sha256_state* ctx);
void sha256_update(sha256_state* ctx, const uint8_t* data, size_t len);
void sha256_final(sha256_state* ctx, uint8_t hash[32]);

// Convenience one-shot
void sha256(const uint8_t* data, size_t len, uint8_t out[32]);
