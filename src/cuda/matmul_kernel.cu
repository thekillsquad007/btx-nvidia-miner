// CUDA implementation for BTX MatMul PoW transcript search.
// One CUDA block per nonce: full sigma derivation, noise, blocked matmul,
// running transcript compression, and target check on device.
// Host uploads constant A/B (FromSeed) once per job; CPU VerifySolution
// cross-checks any hits before submission.

#include "pow/matmul_pow.h"
#include "pow/matrix.h"
#include "pow/uint256_stub.h"

#ifdef __CUDACC__

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kFieldModulus = 0x7FFFFFFFU;
constexpr int kBlockThreads = 256;

struct CudaJobParams {
    int32_t version;
    uint32_t time;
    uint32_t bits;
    uint32_t n;
    uint32_t b;
    uint32_t r;
    uint32_t epsilon_bits;
    uint32_t block_height;
    uint8_t prev_hash[32];
    uint8_t merkle_root[32];
    uint8_t seed_a[32];
    uint8_t seed_b[32];
    uint8_t target[32];
    uint8_t pre_hash_target[32];
};

struct Sha256State {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t data[64];
    uint32_t datalen;
};

__device__ __forceinline__ uint32_t d_rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t d_ch(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t d_maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t d_ep0(uint32_t x)
{
    return d_rotr(x, 2) ^ d_rotr(x, 13) ^ d_rotr(x, 22);
}

__device__ __forceinline__ uint32_t d_ep1(uint32_t x)
{
    return d_rotr(x, 6) ^ d_rotr(x, 11) ^ d_rotr(x, 25);
}

__device__ __forceinline__ uint32_t d_sig0(uint32_t x)
{
    return d_rotr(x, 7) ^ d_rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t d_sig1(uint32_t x)
{
    return d_rotr(x, 17) ^ d_rotr(x, 19) ^ (x >> 10);
}

__device__ void d_sha256_transform(Sha256State* ctx, const uint8_t block[64])
{
    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    // PR#58 windowed schedule: 16-word sliding window (byte-identical, ~2x faster).
    uint32_t w[16];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }

    uint32_t a = ctx->state[0];
    uint32_t b = ctx->state[1];
    uint32_t c = ctx->state[2];
    uint32_t d = ctx->state[3];
    uint32_t e = ctx->state[4];
    uint32_t f = ctx->state[5];
    uint32_t g = ctx->state[6];
    uint32_t h = ctx->state[7];

    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = d_sig1(w[(t - 2) & 15]) + w[(t - 7) & 15] + d_sig0(w[(t - 15) & 15]) + w[(t - 16) & 15];
            w[t & 15] = wt;
        }
        uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + K[t] + wt;
        uint32_t t2 = d_ep0(a) + d_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

__device__ void d_sha256_init(Sha256State* ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

__device__ void d_sha256_update(Sha256State* ctx, const uint8_t* data, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64) {
            d_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void d_sha256_final(Sha256State* ctx, uint8_t hash[32])
{
    uint32_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) {
            ctx->data[i++] = 0;
        }
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) {
            ctx->data[i++] = 0;
        }
        d_sha256_transform(ctx, ctx->data);
        for (uint32_t j = 0; j < 56; ++j) {
            ctx->data[j] = 0;
        }
        i = 56;
    }

    ctx->bitlen += static_cast<uint64_t>(ctx->datalen) * 8;
    ctx->data[63] = static_cast<uint8_t>(ctx->bitlen);
    ctx->data[62] = static_cast<uint8_t>(ctx->bitlen >> 8);
    ctx->data[61] = static_cast<uint8_t>(ctx->bitlen >> 16);
    ctx->data[60] = static_cast<uint8_t>(ctx->bitlen >> 24);
    ctx->data[59] = static_cast<uint8_t>(ctx->bitlen >> 32);
    ctx->data[58] = static_cast<uint8_t>(ctx->bitlen >> 40);
    ctx->data[57] = static_cast<uint8_t>(ctx->bitlen >> 48);
    ctx->data[56] = static_cast<uint8_t>(ctx->bitlen >> 56);
    d_sha256_transform(ctx, ctx->data);

    for (i = 0; i < 4; ++i) {
        hash[i] = static_cast<uint8_t>((ctx->state[0] >> (24 - i * 8)) & 0xff);
        hash[i + 4] = static_cast<uint8_t>((ctx->state[1] >> (24 - i * 8)) & 0xff);
        hash[i + 8] = static_cast<uint8_t>((ctx->state[2] >> (24 - i * 8)) & 0xff);
        hash[i + 12] = static_cast<uint8_t>((ctx->state[3] >> (24 - i * 8)) & 0xff);
        hash[i + 16] = static_cast<uint8_t>((ctx->state[4] >> (24 - i * 8)) & 0xff);
        hash[i + 20] = static_cast<uint8_t>((ctx->state[5] >> (24 - i * 8)) & 0xff);
        hash[i + 24] = static_cast<uint8_t>((ctx->state[6] >> (24 - i * 8)) & 0xff);
        hash[i + 28] = static_cast<uint8_t>((ctx->state[7] >> (24 - i * 8)) & 0xff);
    }
}

__device__ __forceinline__ void d_set_sha_byte(uint32_t w[16], uint32_t offset, uint32_t byte_val)
{
    const uint32_t word_index = offset >> 2U;
    const uint32_t shift = (3U - (offset & 3U)) * 8U;
    w[word_index] |= byte_val << shift;
}

__device__ void d_sha256_compress_words(uint32_t state[8], uint32_t w[16])
{
    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    #pragma unroll
    for (uint32_t t = 0; t < 64; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = d_sig1(w[(t - 2) & 15U]) + w[(t - 7) & 15U] + d_sig0(w[(t - 15) & 15U]) + w[(t - 16) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + K[t] + wt;
        const uint32_t t2 = d_ep0(a) + d_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void d_sha256_init_words(uint32_t state[8])
{
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;
}

__device__ void d_sha256_block0_midstate(const uint8_t* message, uint32_t state[8])
{
    d_sha256_init_words(state);
    uint32_t w[16];
    for (uint32_t word = 0; word < 16; ++word) {
        uint32_t packed = 0;
        for (uint32_t byte = 0; byte < 4; ++byte) {
            packed = (packed << 8U) | message[word * 4U + byte];
        }
        w[word] = packed;
    }
    d_sha256_compress_words(state, w);
}

__device__ void d_sha256_bytes_from_midstate(
    const uint32_t midstate[8],
    const uint8_t* message,
    uint32_t message_len,
    uint8_t out[32])
{
    uint32_t state[8];
    for (uint32_t i = 0; i < 8; ++i) {
        state[i] = midstate[i];
    }

    const uint32_t total_blocks = (message_len + 9U + 63U) / 64U;
    const uint64_t bit_len = static_cast<uint64_t>(message_len) * 8U;
    for (uint32_t block = 1; block < total_blocks; ++block) {
        uint32_t w[16] = {};
        for (uint32_t word = 0; word < 16; ++word) {
            uint32_t packed = 0;
            for (uint32_t byte = 0; byte < 4; ++byte) {
                const uint32_t message_index = block * 64U + word * 4U + byte;
                uint8_t value = 0;
                if (message_index < message_len) {
                    value = message[message_index];
                } else if (message_index == message_len) {
                    value = 0x80U;
                } else {
                    const uint32_t length_start = total_blocks * 64U - 8U;
                    if (message_index >= length_start) {
                        const uint32_t shift = (7U - (message_index - length_start)) * 8U;
                        value = static_cast<uint8_t>((bit_len >> shift) & 0xffU);
                    }
                }
                packed = (packed << 8U) | value;
            }
            w[word] = packed;
        }
        d_sha256_compress_words(state, w);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        out[i * 4U] = static_cast<uint8_t>((state[i] >> 24U) & 0xffU);
        out[i * 4U + 1U] = static_cast<uint8_t>((state[i] >> 16U) & 0xffU);
        out[i * 4U + 2U] = static_cast<uint8_t>((state[i] >> 8U) & 0xffU);
        out[i * 4U + 3U] = static_cast<uint8_t>(state[i] & 0xffU);
    }
}

__device__ void d_to_canonical(const uint8_t* internal, uint8_t* canonical)
{
    for (int i = 0; i < 32; ++i) {
        canonical[i] = internal[31 - i];
    }
}

__device__ __forceinline__ uint32_t d_add(uint32_t a, uint32_t b)
{
    uint32_t s = a + b;
    return (s >= kFieldModulus) ? (s - kFieldModulus) : s;
}

__device__ __forceinline__ uint32_t d_mul(uint32_t a, uint32_t b)
{
    uint64_t p = static_cast<uint64_t>(a) * b;
    uint64_t fold = (p & kFieldModulus) + (p >> 31);
    uint32_t r = static_cast<uint32_t>((fold & kFieldModulus) + (fold >> 31));
    if (r >= kFieldModulus) {
        r -= kFieldModulus;
    }
    return r;
}

__device__ __forceinline__ uint32_t d_dot(const uint32_t* a, const uint32_t* b, uint32_t len)
{
    uint64_t acc = 0;
    uint32_t pending = 0;
    for (uint32_t i = 0; i < len; ++i) {
        acc += static_cast<uint64_t>(a[i]) * b[i];
        if (++pending == 4) {
            uint64_t fold = (acc & kFieldModulus) + (acc >> 31);
            acc = (fold & kFieldModulus) + (fold >> 31);
            pending = 0;
        }
    }
    uint64_t fold = (acc & kFieldModulus) + (acc >> 31);
    uint32_t r = static_cast<uint32_t>((fold & kFieldModulus) + (fold >> 31));
    if (r >= kFieldModulus) {
        r -= kFieldModulus;
    }
    return r;
}

__device__ void d_pack_oracle_midstate(const uint8_t* seed_internal, uint32_t mid[16])
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);

    uint32_t w[8] = {};
    for (uint32_t i = 0; i < 32; ++i) {
        d_set_sha_byte(w, i, seed_bytes[31U - i]);
    }

    uint32_t a = 0x6a09e667U;
    uint32_t b = 0xbb67ae85U;
    uint32_t c = 0x3c6ef372U;
    uint32_t d = 0xa54ff53aU;
    uint32_t e = 0x510e527fU;
    uint32_t f = 0x9b05688cU;
    uint32_t g = 0x1f83d9abU;
    uint32_t h = 0x5be0cd19U;

    const uint32_t K0[8] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5
    };
    #pragma unroll
    for (uint32_t t = 0; t < 8; ++t) {
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + K0[t] + w[t];
        const uint32_t t2 = d_ep0(a) + d_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    mid[0] = w[0];
    mid[1] = w[1];
    mid[2] = w[2];
    mid[3] = w[3];
    mid[4] = w[4];
    mid[5] = w[5];
    mid[6] = w[6];
    mid[7] = w[7];
    mid[8] = a;
    mid[9] = b;
    mid[10] = c;
    mid[11] = d;
    mid[12] = e;
    mid[13] = f;
    mid[14] = g;
    mid[15] = h;
}

__device__ uint32_t d_oracle_from_midstate(const uint32_t mid[16], uint32_t index)
{
    uint32_t w[16];
    w[0] = mid[0];
    w[1] = mid[1];
    w[2] = mid[2];
    w[3] = mid[3];
    w[4] = mid[4];
    w[5] = mid[5];
    w[6] = mid[6];
    w[7] = mid[7];
    #pragma unroll
    for (uint32_t i = 8; i < 16; ++i) {
        w[i] = 0U;
    }
    d_set_sha_byte(w, 32U, index & 0xffU);
    d_set_sha_byte(w, 33U, (index >> 8U) & 0xffU);
    d_set_sha_byte(w, 34U, (index >> 16U) & 0xffU);
    d_set_sha_byte(w, 35U, (index >> 24U) & 0xffU);
    d_set_sha_byte(w, 36U, 0x80U);
    w[15] = 36U * 8U;

    uint32_t a = mid[8];
    uint32_t b = mid[9];
    uint32_t c = mid[10];
    uint32_t d = mid[11];
    uint32_t e = mid[12];
    uint32_t f = mid[13];
    uint32_t g = mid[14];
    uint32_t h = mid[15];

    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    #pragma unroll
    for (uint32_t t = 8; t < 64; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = d_sig1(w[(t - 2) & 15U]) + w[(t - 7) & 15U] + d_sig0(w[(t - 15) & 15U]) + w[(t - 16) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + K[t] + wt;
        const uint32_t t2 = d_ep0(a) + d_maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    const uint32_t sum = 0x6a09e667U + a;
    const uint32_t candidate =
        ((sum & 0xffU) << 24) | (((sum >> 8U) & 0xffU) << 16) |
        (((sum >> 16U) & 0xffU) << 8) | ((sum >> 24U) & 0xffU);
    return candidate & kFieldModulus;
}

__device__ uint32_t d_from_oracle(const uint8_t* seed_internal, uint32_t index)
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);

    for (uint32_t retry = 0; retry < 256; ++retry) {
        Sha256State st;
        d_sha256_init(&st);
        d_sha256_update(&st, seed_bytes, 32);

        uint8_t idx_le[4] = {
            static_cast<uint8_t>(index & 0xff),
            static_cast<uint8_t>((index >> 8) & 0xff),
            static_cast<uint8_t>((index >> 16) & 0xff),
            static_cast<uint8_t>((index >> 24) & 0xff),
        };
        d_sha256_update(&st, idx_le, 4);

        if (retry > 0) {
            uint8_t r_le[4] = {
                static_cast<uint8_t>(retry & 0xff),
                static_cast<uint8_t>((retry >> 8) & 0xff),
                static_cast<uint8_t>((retry >> 16) & 0xff),
                static_cast<uint8_t>((retry >> 24) & 0xff),
            };
            d_sha256_update(&st, r_le, 4);
        }

        uint8_t hash[32];
        d_sha256_final(&st, hash);

        const uint32_t candidate = static_cast<uint32_t>(hash[0]) |
                                   (static_cast<uint32_t>(hash[1]) << 8) |
                                   (static_cast<uint32_t>(hash[2]) << 16) |
                                   (static_cast<uint32_t>(hash[3]) << 24);
        const uint32_t masked = candidate & kFieldModulus;
        if (masked < kFieldModulus) {
            return masked;
        }
    }

    Sha256State fb;
    d_sha256_init(&fb);
    d_sha256_update(&fb, seed_bytes, 32);
    uint8_t idx_le[4] = {
        static_cast<uint8_t>(index & 0xff),
        static_cast<uint8_t>((index >> 8) & 0xff),
        static_cast<uint8_t>((index >> 16) & 0xff),
        static_cast<uint8_t>((index >> 24) & 0xff),
    };
    d_sha256_update(&fb, idx_le, 4);
    const char* tag = "oracle-fallback";
    d_sha256_update(&fb, reinterpret_cast<const uint8_t*>(tag), 15);
    uint8_t fbh[32];
    d_sha256_final(&fb, fbh);
    const uint32_t v = static_cast<uint32_t>(fbh[0]) | (static_cast<uint32_t>(fbh[1]) << 8) |
                       (static_cast<uint32_t>(fbh[2]) << 16) | (static_cast<uint32_t>(fbh[3]) << 24);
    return v % kFieldModulus;
}

__device__ void d_derive_noise_seed(const char* domain_tag, const uint8_t* sigma_internal, uint8_t* out_internal)
{
    uint8_t sigma_canonical[32];
    d_to_canonical(sigma_internal, sigma_canonical);

    Sha256State st;
    d_sha256_init(&st);
    d_sha256_update(&st, reinterpret_cast<const uint8_t*>(domain_tag), 18);
    d_sha256_update(&st, sigma_canonical, 32);
    uint8_t digest[32];
    d_sha256_final(&st, digest);

    for (int i = 0; i < 32; ++i) {
        out_internal[i] = digest[31 - i];
    }
}

__device__ void d_write_compact_size(Sha256State& hasher, uint64_t val)
{
    if (val < 253) {
        const uint8_t c = static_cast<uint8_t>(val);
        d_sha256_update(&hasher, &c, 1);
    } else if (val < 0x10000) {
        const uint8_t buf[3] = {
            0xFD,
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8),
        };
        d_sha256_update(&hasher, buf, 3);
    } else if (val < 0x100000000ULL) {
        const uint8_t buf[5] = {
            0xFE,
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8),
            static_cast<uint8_t>(val >> 16),
            static_cast<uint8_t>(val >> 24),
        };
        d_sha256_update(&hasher, buf, 5);
    } else {
        const uint8_t buf[9] = {
            0xFF,
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8),
            static_cast<uint8_t>(val >> 16),
            static_cast<uint8_t>(val >> 24),
            static_cast<uint8_t>(val >> 32),
            static_cast<uint8_t>(val >> 40),
            static_cast<uint8_t>(val >> 48),
            static_cast<uint8_t>(val >> 56),
        };
        d_sha256_update(&hasher, buf, 9);
    }
}

__device__ void d_matmul_seed_v2(
    const CudaJobParams& job,
    uint64_t nonce64,
    uint8_t which,
    uint8_t out_internal[32])
{
    Sha256State hasher;
    d_sha256_init(&hasher);
    static const char kTag[] = "BTX_MATMUL_SEED_V2";
    d_write_compact_size(hasher, sizeof(kTag) - 1);
    d_sha256_update(&hasher, reinterpret_cast<const uint8_t*>(kTag), sizeof(kTag) - 1);
    d_sha256_update(&hasher, job.prev_hash, 32);

    uint8_t le[8];
    le[0] = static_cast<uint8_t>(job.block_height & 0xff);
    le[1] = static_cast<uint8_t>((job.block_height >> 8) & 0xff);
    le[2] = static_cast<uint8_t>((job.block_height >> 16) & 0xff);
    le[3] = static_cast<uint8_t>((job.block_height >> 24) & 0xff);
    d_sha256_update(&hasher, le, 4);
    le[0] = static_cast<uint8_t>(job.version & 0xff);
    le[1] = static_cast<uint8_t>((job.version >> 8) & 0xff);
    le[2] = static_cast<uint8_t>((job.version >> 16) & 0xff);
    le[3] = static_cast<uint8_t>((job.version >> 24) & 0xff);
    d_sha256_update(&hasher, le, 4);
    d_sha256_update(&hasher, job.merkle_root, 32);
    le[0] = static_cast<uint8_t>(job.time & 0xff);
    le[1] = static_cast<uint8_t>((job.time >> 8) & 0xff);
    le[2] = static_cast<uint8_t>((job.time >> 16) & 0xff);
    le[3] = static_cast<uint8_t>((job.time >> 24) & 0xff);
    d_sha256_update(&hasher, le, 4);
    le[0] = static_cast<uint8_t>(job.bits & 0xff);
    le[1] = static_cast<uint8_t>((job.bits >> 8) & 0xff);
    le[2] = static_cast<uint8_t>((job.bits >> 16) & 0xff);
    le[3] = static_cast<uint8_t>((job.bits >> 24) & 0xff);
    d_sha256_update(&hasher, le, 4);
    for (int i = 0; i < 8; ++i) {
        le[i] = static_cast<uint8_t>((nonce64 >> (i * 8)) & 0xff);
    }
    d_sha256_update(&hasher, le, 8);
    const uint16_t dim = static_cast<uint16_t>(job.n);
    le[0] = static_cast<uint8_t>(dim & 0xff);
    le[1] = static_cast<uint8_t>((dim >> 8) & 0xff);
    d_sha256_update(&hasher, le, 2);
    d_sha256_update(&hasher, &which, 1);
    d_sha256_final(&hasher, out_internal);
}

__device__ uint32_t d_build_seed_v2_message(
    const CudaJobParams& job,
    uint64_t nonce64,
    uint8_t which,
    uint8_t message[110])
{
    uint32_t offset = 0;
    static const char kTag[] = "BTX_MATMUL_SEED_V2";
    message[offset++] = static_cast<uint8_t>(sizeof(kTag) - 1);
    for (size_t i = 0; i < sizeof(kTag) - 1; ++i) {
        message[offset++] = static_cast<uint8_t>(kTag[i]);
    }
    for (int i = 0; i < 32; ++i) {
        message[offset++] = job.prev_hash[i];
    }
    message[offset++] = static_cast<uint8_t>(job.block_height & 0xff);
    message[offset++] = static_cast<uint8_t>((job.block_height >> 8) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.block_height >> 16) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.block_height >> 24) & 0xff);
    message[offset++] = static_cast<uint8_t>(job.version & 0xff);
    message[offset++] = static_cast<uint8_t>((job.version >> 8) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.version >> 16) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.version >> 24) & 0xff);
    for (int i = 0; i < 32; ++i) {
        message[offset++] = job.merkle_root[i];
    }
    message[offset++] = static_cast<uint8_t>(job.time & 0xff);
    message[offset++] = static_cast<uint8_t>((job.time >> 8) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.time >> 16) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.time >> 24) & 0xff);
    message[offset++] = static_cast<uint8_t>(job.bits & 0xff);
    message[offset++] = static_cast<uint8_t>((job.bits >> 8) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.bits >> 16) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.bits >> 24) & 0xff);
    for (int i = 0; i < 8; ++i) {
        message[offset++] = static_cast<uint8_t>((nonce64 >> (i * 8)) & 0xff);
    }
    const uint16_t dim = static_cast<uint16_t>(job.n);
    message[offset++] = static_cast<uint8_t>(dim & 0xff);
    message[offset++] = static_cast<uint8_t>((dim >> 8) & 0xff);
    message[offset++] = which;
    return offset;
}

__device__ uint32_t d_build_header_hash_message(
    const CudaJobParams& job,
    uint64_t nonce64,
    const uint8_t* seed_a,
    const uint8_t* seed_b,
    uint8_t message[150])
{
    uint32_t offset = 0;
    message[offset++] = static_cast<uint8_t>(job.version & 0xff);
    message[offset++] = static_cast<uint8_t>((job.version >> 8) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.version >> 16) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.version >> 24) & 0xff);
    for (int i = 0; i < 32; ++i) {
        message[offset++] = job.prev_hash[i];
    }
    for (int i = 0; i < 32; ++i) {
        message[offset++] = job.merkle_root[i];
    }
    message[offset++] = static_cast<uint8_t>(job.time & 0xff);
    message[offset++] = static_cast<uint8_t>((job.time >> 8) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.time >> 16) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.time >> 24) & 0xff);
    message[offset++] = static_cast<uint8_t>(job.bits & 0xff);
    message[offset++] = static_cast<uint8_t>((job.bits >> 8) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.bits >> 16) & 0xff);
    message[offset++] = static_cast<uint8_t>((job.bits >> 24) & 0xff);
    for (int i = 0; i < 8; ++i) {
        message[offset++] = static_cast<uint8_t>((nonce64 >> (i * 8)) & 0xff);
    }
    const uint16_t dim = static_cast<uint16_t>(job.n);
    message[offset++] = static_cast<uint8_t>(dim & 0xff);
    message[offset++] = static_cast<uint8_t>((dim >> 8) & 0xff);
    for (int i = 0; i < 32; ++i) {
        message[offset++] = seed_a[i];
    }
    for (int i = 0; i < 32; ++i) {
        message[offset++] = seed_b[i];
    }
    return offset;
}

__device__ void d_matmul_seed_v2_midstate(
    const CudaJobParams& job,
    uint64_t nonce64,
    uint8_t which,
    const uint32_t seed_midstate[8],
    uint8_t out_internal[32])
{
    uint8_t message[110];
    const uint32_t len = d_build_seed_v2_message(job, nonce64, which, message);
    d_sha256_bytes_from_midstate(seed_midstate, message, len, out_internal);
}

__device__ void d_derive_sigma_midstate(
    const CudaJobParams& job,
    uint64_t nonce64,
    const uint8_t* seed_a,
    const uint8_t* seed_b,
    const uint32_t header_midstate[8],
    uint8_t* sigma_internal)
{
    uint8_t message[150];
    const uint32_t len = d_build_header_hash_message(job, nonce64, seed_a, seed_b, message);
    uint8_t first[32];
    d_sha256_bytes_from_midstate(header_midstate, message, len, first);

    Sha256State h2;
    d_sha256_init(&h2);
    d_sha256_update(&h2, first, 32);
    d_sha256_final(&h2, sigma_internal);
}

__device__ void d_derive_sigma(
    const CudaJobParams& job,
    uint64_t nonce64,
    const uint8_t* seed_a,
    const uint8_t* seed_b,
    uint8_t* sigma_internal)
{
    Sha256State h;
    d_sha256_init(&h);

    uint8_t ver[4] = {
        static_cast<uint8_t>(job.version & 0xff),
        static_cast<uint8_t>((job.version >> 8) & 0xff),
        static_cast<uint8_t>((job.version >> 16) & 0xff),
        static_cast<uint8_t>((job.version >> 24) & 0xff),
    };
    uint8_t t[4] = {
        static_cast<uint8_t>(job.time & 0xff),
        static_cast<uint8_t>((job.time >> 8) & 0xff),
        static_cast<uint8_t>((job.time >> 16) & 0xff),
        static_cast<uint8_t>((job.time >> 24) & 0xff),
    };
    uint8_t bi[4] = {
        static_cast<uint8_t>(job.bits & 0xff),
        static_cast<uint8_t>((job.bits >> 8) & 0xff),
        static_cast<uint8_t>((job.bits >> 16) & 0xff),
        static_cast<uint8_t>((job.bits >> 24) & 0xff),
    };
    uint8_t n64[8];
    for (int i = 0; i < 8; ++i) {
        n64[i] = static_cast<uint8_t>((nonce64 >> (i * 8)) & 0xff);
    }
    const uint16_t dim = static_cast<uint16_t>(job.n);
    uint8_t dm[2] = {static_cast<uint8_t>(dim & 0xff), static_cast<uint8_t>((dim >> 8) & 0xff)};

    d_sha256_update(&h, ver, 4);
    d_sha256_update(&h, job.prev_hash, 32);
    d_sha256_update(&h, job.merkle_root, 32);
    d_sha256_update(&h, t, 4);
    d_sha256_update(&h, bi, 4);
    d_sha256_update(&h, n64, 8);
    d_sha256_update(&h, dm, 2);
    d_sha256_update(&h, seed_a, 32);
    d_sha256_update(&h, seed_b, 32);

    uint8_t first[32];
    d_sha256_final(&h, first);

    Sha256State h2;
    d_sha256_init(&h2);
    d_sha256_update(&h2, first, 32);
    d_sha256_final(&h2, sigma_internal);
}

__device__ bool d_sigma_below_prehash(const uint8_t sigma[32], const uint8_t target_arith[32])
{
    for (int i = 0; i < 32; ++i) {
        const uint8_t s = sigma[i];
        const uint8_t t = target_arith[31 - i];
        if (s < t) return true;
        if (s > t) return false;
    }
    return true;
}

__device__ void d_fill_rect(uint32_t* out, uint32_t rows, uint32_t cols, const uint8_t* seed_internal)
{
    // Must match amdbtx matmul_kernel.hip: full SHA oracle per cell (no midstate shortcut).
    const uint32_t total = rows * cols;
    for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        out[idx] = d_from_oracle(seed_internal, idx);
    }
}

__device__ void d_zero_u32(uint32_t* out, uint32_t count)
{
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        out[i] = 0;
    }
}

__device__ void d_mat_get(const uint32_t* m, uint32_t n, uint32_t row, uint32_t col, uint32_t* out, uint32_t bsz)
{
    const uint32_t cells = bsz * bsz;
    for (uint32_t idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        const uint32_t r = idx / bsz;
        const uint32_t c = idx % bsz;
        out[idx] = m[(row + r) * n + (col + c)];
    }
}

__device__ void d_block_lowrank(
    uint32_t* out,
    const uint32_t* L,
    const uint32_t* R,
    uint32_t l_rows,
    uint32_t l_cols,
    uint32_t r_cols,
    uint32_t row0,
    uint32_t col0,
    uint32_t bsz)
{
    (void)l_rows;
    const uint32_t cells = bsz * bsz;
    for (uint32_t idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        const uint32_t i = idx / bsz;
        const uint32_t j = idx % bsz;
        uint32_t acc = 0;
        for (uint32_t k = 0; k < l_cols; ++k) {
            const uint32_t a = L[(row0 + i) * l_cols + k];
            const uint32_t b = R[k * r_cols + (col0 + j)];
            acc = d_add(acc, d_mul(a, b));
        }
        out[idx] = acc;
    }
}

__device__ void d_block_matmul(uint32_t* out, const uint32_t* A, const uint32_t* B, uint32_t bsz)
{
    const uint32_t cells = bsz * bsz;
    for (uint32_t idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        const uint32_t i = idx / bsz;
        const uint32_t j = idx % bsz;
        uint32_t acc = 0;
        for (uint32_t k = 0; k < bsz; ++k) {
            acc = d_add(acc, d_mul(A[i * bsz + k], B[k * bsz + j]));
        }
        out[idx] = acc;
    }
}

__device__ void d_add_block(uint32_t* dst, const uint32_t* src, uint32_t bsz)
{
    const uint32_t n = bsz * bsz;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = d_add(dst[i], src[i]);
    }
}

__device__ void d_write_c_block(uint32_t* C, uint32_t job_n, uint32_t row0, uint32_t col0,
                                const uint32_t* cblk, uint32_t bsz)
{
    const uint32_t cells = bsz * bsz;
    for (uint32_t idx = threadIdx.x; idx < cells; idx += blockDim.x) {
        const uint32_t x = idx / bsz;
        const uint32_t y = idx % bsz;
        C[(row0 + x) * job_n + (col0 + y)] = cblk[idx];
    }
}

__device__ void d_canonicalize_sha256(const uint8_t raw[32], uint8_t out[32])
{
    for (int i = 0; i < 32; ++i) {
        out[i] = raw[31 - i];
    }
}

// Match btx Uint256LE / pool share_validator: data[31] is most significant.
__device__ bool d_uint256_le(const uint8_t digest[32], const uint8_t target[32])
{
    for (int i = 31; i >= 0; --i) {
        if (digest[i] < target[i]) {
            return true;
        }
        if (digest[i] > target[i]) {
            return false;
        }
    }
    return true;
}

__device__ void d_hash_matrix_words(const uint32_t* words, uint32_t count, uint8_t out[32])
{
    Sha256State hasher;
    d_sha256_init(&hasher);
    if (count > 0) {
        d_sha256_update(&hasher, reinterpret_cast<const uint8_t*>(words),
                        static_cast<size_t>(count) * sizeof(uint32_t));
    }
    uint8_t inner[32];
    d_sha256_final(&hasher, inner);

    Sha256State outer;
    d_sha256_init(&outer);
    d_sha256_update(&outer, inner, 32);
    d_sha256_final(&outer, out);
}

__device__ void d_finalize_product_digest(
    const uint8_t* sigma_data,
    const uint8_t* c_prime_data,
    uint32_t n,
    uint32_t bsz,
    uint8_t digest_out[32])
{
    Sha256State outer;
    d_sha256_init(&outer);
    static const char kTag[] = "matmul-product-digest-v3";
    d_sha256_update(&outer, reinterpret_cast<const uint8_t*>(kTag), sizeof(kTag) - 1);
    d_sha256_update(&outer, sigma_data, 32);
    d_sha256_update(&outer, c_prime_data, 32);
    uint8_t dim_le[4] = {
        static_cast<uint8_t>(n & 0xff),
        static_cast<uint8_t>((n >> 8) & 0xff),
        static_cast<uint8_t>((n >> 16) & 0xff),
        static_cast<uint8_t>((n >> 24) & 0xff),
    };
    uint8_t b_le[4] = {
        static_cast<uint8_t>(bsz & 0xff),
        static_cast<uint8_t>((bsz >> 8) & 0xff),
        static_cast<uint8_t>((bsz >> 16) & 0xff),
        static_cast<uint8_t>((bsz >> 24) & 0xff),
    };
    d_sha256_update(&outer, dim_le, 4);
    d_sha256_update(&outer, b_le, 4);

    uint8_t inner[32];
    d_sha256_final(&outer, inner);

    Sha256State outer2;
    d_sha256_init(&outer2);
    d_sha256_update(&outer2, inner, 32);
    d_sha256_final(&outer2, digest_out);
}

__device__ __forceinline__ uint32_t d_reduce64(uint64_t acc)
{
    const uint64_t fold1 = (acc & kFieldModulus) + (acc >> 31);
    const uint32_t lo = static_cast<uint32_t>(fold1 & kFieldModulus);
    const uint32_t hi = static_cast<uint32_t>(fold1 >> 31);
    uint32_t r = lo + hi;
    if (r >= kFieldModulus) {
        r -= kFieldModulus;
    }
    return r;
}

__device__ __forceinline__ uint64_t d_fold64(uint64_t acc)
{
    return static_cast<uint64_t>(d_reduce64(acc));
}

__device__ void d_add_lowrank_product(
    uint32_t* M,
    uint32_t n,
    uint32_t rank,
    const uint32_t* L,
    const uint32_t* R)
{
    const uint32_t total = n * n;
    for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const uint32_t i = idx / n;
        const uint32_t j = idx % n;
        uint32_t acc = 0;
        for (uint32_t k = 0; k < rank; ++k) {
            acc = d_add(acc, d_mul(L[i * rank + k], R[k * n + j]));
        }
        M[idx] = d_add(M[idx], acc);
    }
}

__device__ void d_build_factored_rhs(
    const uint32_t* B,
    const uint32_t* compress_v,
    uint32_t n,
    uint32_t bsz,
    uint32_t N,
    uint32_t* rhs)
{
    const uint32_t rhs_elems = bsz * n * N;
    for (uint32_t gid = threadIdx.x; gid < rhs_elems; gid += blockDim.x) {
        const uint32_t m = gid % n;
        const uint32_t jx = gid / n;
        const uint32_t x = jx % bsz;
        const uint32_t j = jx / bsz;
        const uint32_t* b_row = B + static_cast<size_t>(m) * n + j * bsz;
        const uint32_t* w_row = compress_v + x * bsz;
        uint64_t acc = 0;
        uint32_t pending = 0;
        for (uint32_t y = 0; y < bsz; ++y) {
            acc += static_cast<uint64_t>(w_row[y]) * b_row[y];
            if (++pending == 4) {
                acc = d_fold64(acc);
                pending = 0;
            }
        }
        rhs[gid] = d_reduce64(acc);
    }
}

__device__ void d_compute_factored_words(
    const uint32_t* A,
    const uint32_t* rhs,
    uint32_t n,
    uint32_t bsz,
    uint32_t N,
    uint32_t* output)
{
    const uint32_t tiles_per_axis = N >> 1U;
    const uint32_t tiles_per_request = tiles_per_axis * tiles_per_axis;
    const uint32_t num_warps = blockDim.x >> 5U;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp_id = threadIdx.x >> 5U;

    for (uint32_t tile_index = warp_id; tile_index < tiles_per_request; tile_index += num_warps) {
        const uint32_t j0 = (tile_index % tiles_per_axis) * 2U;
        const uint32_t i0 = (tile_index / tiles_per_axis) * 2U;
        uint64_t acc00 = 0;
        uint64_t acc01 = 0;
        uint64_t acc10 = 0;
        uint64_t acc11 = 0;
        uint32_t pending = 0;
        for (uint32_t x = 0; x < bsz; ++x) {
            const uint32_t* a_row0 = A + static_cast<size_t>(i0 * bsz + x) * n;
            const uint32_t* a_row1 = a_row0 + static_cast<size_t>(bsz) * n;
            const uint32_t* d_row0 = rhs + static_cast<size_t>(j0 * bsz + x) * n;
            const uint32_t* d_row1 = d_row0 + static_cast<size_t>(bsz) * n;
            for (uint32_t m = lane; m < n; m += 32U) {
                const uint64_t a0 = a_row0[m];
                const uint64_t a1 = a_row1[m];
                const uint64_t d0 = d_row0[m];
                const uint64_t d1 = d_row1[m];
                acc00 += a0 * d0;
                acc01 += a0 * d1;
                acc10 += a1 * d0;
                acc11 += a1 * d1;
                if (++pending == 4) {
                    acc00 = d_fold64(acc00);
                    acc01 = d_fold64(acc01);
                    acc10 = d_fold64(acc10);
                    acc11 = d_fold64(acc11);
                    pending = 0;
                }
            }
        }
        uint32_t value00 = d_reduce64(acc00);
        uint32_t value01 = d_reduce64(acc01);
        uint32_t value10 = d_reduce64(acc10);
        uint32_t value11 = d_reduce64(acc11);
        for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
            value00 = d_add(value00, __shfl_down_sync(0xffffffffU, value00, offset));
            value01 = d_add(value01, __shfl_down_sync(0xffffffffU, value01, offset));
            value10 = d_add(value10, __shfl_down_sync(0xffffffffU, value10, offset));
            value11 = d_add(value11, __shfl_down_sync(0xffffffffU, value11, offset));
        }
        if (lane == 0) {
            output[i0 * N + j0] = value00;
            output[i0 * N + j0 + 1U] = value01;
            output[(i0 + 1U) * N + j0] = value10;
            output[(i0 + 1U) * N + j0 + 1U] = value11;
        }
    }
}

__device__ void d_build_compress_vec(const uint8_t* sigma_internal, uint32_t bsz, uint32_t* compress_v)
{
    __shared__ uint8_t s_seed_internal[32];

    if (threadIdx.x == 0) {
        uint8_t sigma_canonical[32];
        d_to_canonical(sigma_internal, sigma_canonical);

        Sha256State st;
        d_sha256_init(&st);
        const char* tag = "matmul-compress-v1";
        d_sha256_update(&st, reinterpret_cast<const uint8_t*>(tag), 18);
        d_sha256_update(&st, sigma_canonical, 32);
        uint8_t seedb[32];
        d_sha256_final(&st, seedb);

        for (int i = 0; i < 32; ++i) {
            s_seed_internal[i] = seedb[31 - i];
        }
    }
    __syncthreads();

    const uint32_t len = bsz * bsz;
    for (uint32_t k = threadIdx.x; k < len; k += blockDim.x) {
        compress_v[k] = d_from_oracle(s_seed_internal, k);
    }
}

__device__ bool d_solve_nonce(
    const CudaJobParams& job,
    uint32_t* A,
    uint32_t* B,
    uint64_t nonce64,
    uint32_t* C,
    uint32_t* E_L,
    uint32_t* E_R,
    uint32_t* F_L,
    uint32_t* F_R,
    uint32_t* compress_v,
    uint32_t* ablk,
    uint32_t* bblk,
    uint32_t* noise_blk,
    uint32_t* prod,
    uint32_t* cblk,
    const uint8_t* reuse_seed_a,
    const uint8_t* reuse_seed_b,
    uint8_t* digest_out)
{
    __shared__ uint8_t s_sigma[32];
    __shared__ uint8_t s_seed_el[32];
    __shared__ uint8_t s_seed_er[32];
    __shared__ uint8_t s_seed_fl[32];
    __shared__ uint8_t s_seed_fr[32];
    __shared__ bool s_hit;
    __shared__ bool s_sigma_pass;
    __shared__ uint8_t s_seed_a[32];
    __shared__ uint8_t s_seed_b[32];

    if (threadIdx.x == 0) {
        if (reuse_seed_a && reuse_seed_b) {
            for (int i = 0; i < 32; ++i) {
                s_seed_a[i] = reuse_seed_a[i];
                s_seed_b[i] = reuse_seed_b[i];
            }
        } else if (job.block_height >= btx::pow::kMatMulSeedV2Height) {
            d_matmul_seed_v2(job, nonce64, 0, s_seed_a);
            d_matmul_seed_v2(job, nonce64, 1, s_seed_b);
        } else {
            for (int i = 0; i < 32; ++i) {
                s_seed_a[i] = job.seed_a[i];
                s_seed_b[i] = job.seed_b[i];
            }
        }
        d_derive_sigma(job, nonce64, s_seed_a, s_seed_b, s_sigma);
        s_sigma_pass = job.epsilon_bits == 0 ||
                       d_sigma_below_prehash(s_sigma, job.pre_hash_target);
    }
    __syncthreads();
    if (!s_sigma_pass) {
        return false;
    }

    if (threadIdx.x == 0) {
        d_derive_noise_seed("matmul_noise_EL_v1", s_sigma, s_seed_el);
        d_derive_noise_seed("matmul_noise_ER_v1", s_sigma, s_seed_er);
        d_derive_noise_seed("matmul_noise_FL_v1", s_sigma, s_seed_fl);
        d_derive_noise_seed("matmul_noise_FR_v1", s_sigma, s_seed_fr);
    }
    __syncthreads();

    d_fill_rect(E_L, job.n, job.r, s_seed_el);
    __syncthreads();
    d_fill_rect(E_R, job.r, job.n, s_seed_er);
    __syncthreads();
    d_fill_rect(F_L, job.n, job.r, s_seed_fl);
    __syncthreads();
    d_fill_rect(F_R, job.r, job.n, s_seed_fr);
    __syncthreads();

    const uint32_t bsz = job.b;
    const uint32_t N = job.n / bsz;
    const uint32_t word_count = N * N;

    d_build_compress_vec(s_sigma, job.b, compress_v);
    __syncthreads();

    // Factored compression is not used by amdbtx pool solver; keep disabled until
    // byte-identical with the blocked matmul reference (dexbtx PR#58 port pending).
    const bool use_factored = false;

    if (use_factored) {
        d_add_lowrank_product(A, job.n, job.r, E_L, E_R);
        __syncthreads();
        d_add_lowrank_product(B, job.n, job.r, F_L, F_R);
        __syncthreads();
        d_build_factored_rhs(B, compress_v, job.n, bsz, N, C);
        __syncthreads();
        d_compute_factored_words(A, C, job.n, bsz, N, C);
        __syncthreads();
    } else {
        d_zero_u32(C, word_count);
        __syncthreads();

        const uint32_t total_steps = N * N * N;
        for (uint32_t step = 0; step < total_steps; ++step) {
            const uint32_t bi = step / (N * N);
            const uint32_t rem = step % (N * N);
            const uint32_t bj = rem / N;
            const uint32_t ell = rem % N;
            const uint32_t row0 = bi * bsz;
            const uint32_t col0 = bj * bsz;
            const uint32_t mid0 = ell * bsz;

            d_mat_get(A, job.n, row0, mid0, ablk, bsz);
            __syncthreads();
            d_block_lowrank(noise_blk, E_L, E_R, job.n, job.r, job.n, row0, mid0, bsz);
            __syncthreads();
            d_add_block(ablk, noise_blk, bsz);
            __syncthreads();

            d_mat_get(B, job.n, mid0, col0, bblk, bsz);
            __syncthreads();
            d_block_lowrank(noise_blk, F_L, F_R, job.n, job.r, job.n, mid0, col0, bsz);
            __syncthreads();
            d_add_block(bblk, noise_blk, bsz);
            __syncthreads();

            d_block_matmul(prod, ablk, bblk, bsz);
            __syncthreads();

            if (threadIdx.x == 0) {
                const uint32_t word_idx = bi * N + bj;
                const uint32_t term = d_dot(prod, compress_v, bsz * bsz);
                C[word_idx] = d_add(C[word_idx], term);
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        uint8_t c_prime[32];
        d_hash_matrix_words(C, word_count, c_prime);
        d_finalize_product_digest(s_sigma, c_prime, job.n, bsz, digest_out);
        s_hit = d_uint256_le(digest_out, job.target);
    }
    __syncthreads();
    return s_hit;
}

__global__ void matmul_nonce_kernel(
    CudaJobParams params,
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ B,
    const uint64_t* __restrict__ nonces,
    uint32_t* __restrict__ workspaces,
    uint8_t* __restrict__ out_digests,
    uint8_t* __restrict__ out_found)
{
    const size_t idx = static_cast<size_t>(blockIdx.x);
    if (idx >= gridDim.x) {
        return;
    }

    const uint32_t n = params.n;
    const uint32_t r = params.r;
    const uint32_t bsz = params.b;
    const size_t nn = static_cast<size_t>(n) * n;
    const size_t noise_elems = 2 * (static_cast<size_t>(n) * r + static_cast<size_t>(r) * n);
    const size_t scratch_elems = static_cast<size_t>(bsz) * bsz;
    const size_t compress_elems = scratch_elems;
    const bool use_v2 = params.block_height >= btx::pow::kMatMulSeedV2Height;
    const size_t v2_base_elems = use_v2 ? (2 * nn) : 0;
    const size_t per_nonce_elems = v2_base_elems + nn + noise_elems + compress_elems + scratch_elems * 5;

    // Per-nonce workspace layout:
    // [A nn][B nn] (V2 only) [C nn][E_L n*r][E_R r*n][F_L n*r][F_R r*n][compress b*b][ablk][bblk][noise][prod][cblk]
    size_t off = idx * per_nonce_elems;

    uint32_t* A_local = use_v2 ? (workspaces + off) : nullptr;
    if (use_v2) off += nn;
    uint32_t* B_local = use_v2 ? (workspaces + off) : nullptr;
    if (use_v2) off += nn;

    uint32_t* C = workspaces + off;
    off += nn;
    uint32_t* E_L = workspaces + off;
    off += static_cast<size_t>(n) * r;
    uint32_t* E_R = workspaces + off;
    off += static_cast<size_t>(r) * n;
    uint32_t* F_L = workspaces + off;
    off += static_cast<size_t>(n) * r;
    uint32_t* F_R = workspaces + off;
    off += static_cast<size_t>(r) * n;
    uint32_t* compress_v = workspaces + off;
    off += compress_elems;
    uint32_t* ablk = workspaces + off;
    off += scratch_elems;
    uint32_t* bblk = workspaces + off;
    off += scratch_elems;
    uint32_t* noise_blk = workspaces + off;
    off += scratch_elems;
    uint32_t* prod = workspaces + off;
    off += scratch_elems;
    uint32_t* cblk = workspaces + off;

    __shared__ uint8_t s_v2_seed_a[32];
    __shared__ uint8_t s_v2_seed_b[32];

    if (use_v2) {
        if (threadIdx.x == 0) {
            d_matmul_seed_v2(params, nonces[idx], 0, s_v2_seed_a);
            d_matmul_seed_v2(params, nonces[idx], 1, s_v2_seed_b);
        }
        __syncthreads();
        d_fill_rect(A_local, n, n, s_v2_seed_a);
        __syncthreads();
        d_fill_rect(B_local, n, n, s_v2_seed_b);
        __syncthreads();
    }

    uint32_t* use_a = use_v2 ? A_local : const_cast<uint32_t*>(A);
    uint32_t* use_b = use_v2 ? B_local : const_cast<uint32_t*>(B);

    uint8_t digest[32];
    const uint8_t* reuse_a = use_v2 ? s_v2_seed_a : nullptr;
    const uint8_t* reuse_b = use_v2 ? s_v2_seed_b : nullptr;
    const bool hit = d_solve_nonce(
        params, use_a, use_b, nonces[idx], C, E_L, E_R, F_L, F_R, compress_v,
        ablk, bblk, noise_blk, prod, cblk, reuse_a, reuse_b, digest);

    if (threadIdx.x == 0) {
        if (hit) {
            out_found[idx] = 1;
            for (int i = 0; i < 32; ++i) {
                out_digests[idx * 32 + i] = digest[i];
            }
        }
    }
}

struct DeviceMatrixCache {
    int device = -1;
    uint8_t seed_a[32]{};
    uint8_t seed_b[32]{};
    uint32_t n = 0;
    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
};

DeviceMatrixCache g_matrix_cache[16];

struct DeviceLaunchPool {
    int device = -1;
    uint64_t* d_nonces = nullptr;
    uint32_t* d_workspace = nullptr;
    uint8_t* d_digests = nullptr;
    uint8_t* d_found = nullptr;
    size_t nonce_cap = 0;
    size_t ws_bytes = 0;
    size_t digest_cap = 0;
};

DeviceLaunchPool g_launch_pool[16];

void FreeLaunchPool(DeviceLaunchPool& pool)
{
    if (pool.d_nonces) cudaFree(pool.d_nonces);
    if (pool.d_workspace) cudaFree(pool.d_workspace);
    if (pool.d_digests) cudaFree(pool.d_digests);
    if (pool.d_found) cudaFree(pool.d_found);
    pool = {};
}

bool EnsureLaunchPool(int device, size_t batch, size_t ws_bytes)
{
    if (device < 0 || device >= 16) return false;
    DeviceLaunchPool& pool = g_launch_pool[device];
    if (pool.device != device) {
        FreeLaunchPool(pool);
        pool.device = device;
    }

    if (pool.nonce_cap < batch) {
        if (pool.d_nonces) cudaFree(pool.d_nonces);
        pool.d_nonces = nullptr;
        if (cudaMalloc(&pool.d_nonces, batch * sizeof(uint64_t)) != cudaSuccess) {
            FreeLaunchPool(pool);
            return false;
        }
        pool.nonce_cap = batch;
    }
    if (pool.ws_bytes < ws_bytes) {
        if (pool.d_workspace) cudaFree(pool.d_workspace);
        pool.d_workspace = nullptr;
        if (cudaMalloc(&pool.d_workspace, ws_bytes) != cudaSuccess) {
            FreeLaunchPool(pool);
            return false;
        }
        pool.ws_bytes = ws_bytes;
    }
    const size_t digest_bytes = batch * 32;
    if (pool.digest_cap < batch) {
        if (pool.d_digests) cudaFree(pool.d_digests);
        if (pool.d_found) cudaFree(pool.d_found);
        pool.d_digests = nullptr;
        pool.d_found = nullptr;
        if (cudaMalloc(&pool.d_digests, digest_bytes) != cudaSuccess ||
            cudaMalloc(&pool.d_found, batch) != cudaSuccess) {
            FreeLaunchPool(pool);
            return false;
        }
        pool.digest_cap = batch;
    }
    return pool.d_nonces && pool.d_workspace && pool.d_digests && pool.d_found;
}

void FreeMatrixCache(DeviceMatrixCache& cache)
{
    if (cache.d_A) {
        cudaFree(cache.d_A);
        cache.d_A = nullptr;
    }
    if (cache.d_B) {
        cudaFree(cache.d_B);
        cache.d_B = nullptr;
    }
}

bool EnsureMatricesOnDevice(
    int device,
    const btx::pow::MatMulJob& job,
    uint32_t** d_A,
    uint32_t** d_B)
{
    if (device < 0 || device >= 16) {
        return false;
    }

    // Post-fork jobs derive per-nonce A/B on device; skip CPU FromSeed + H2D.
    if (job.block_height >= btx::pow::kMatMulSeedV2Height) {
        *d_A = nullptr;
        *d_B = nullptr;
        return true;
    }

    DeviceMatrixCache& cache = g_matrix_cache[device];
    const bool same =
        cache.device == device &&
        cache.n == job.n &&
        std::memcmp(cache.seed_a, job.seed_a.data(), 32) == 0 &&
        std::memcmp(cache.seed_b, job.seed_b.data(), 32) == 0 &&
        cache.d_A != nullptr &&
        cache.d_B != nullptr;

    if (same) {
        *d_A = cache.d_A;
        *d_B = cache.d_B;
        return true;
    }

    if (cache.device != device) {
        FreeMatrixCache(cache);
        cache.device = device;
    } else {
        FreeMatrixCache(cache);
    }

    const btx::pow::Matrix A = btx::pow::FromSeed(job.seed_a, job.n);
    const btx::pow::Matrix B = btx::pow::FromSeed(job.seed_b, job.n);
    const size_t elems = static_cast<size_t>(job.n) * job.n;
    const size_t bytes = elems * sizeof(uint32_t);

    std::vector<uint32_t> h_A(elems);
    std::vector<uint32_t> h_B(elems);
    for (size_t i = 0; i < elems; ++i) {
        h_A[i] = A.data()[i];
        h_B[i] = B.data()[i];
    }

    if (cudaMalloc(&cache.d_A, bytes) != cudaSuccess ||
        cudaMalloc(&cache.d_B, bytes) != cudaSuccess) {
        FreeMatrixCache(cache);
        return false;
    }
    if (cudaMemcpy(cache.d_A, h_A.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(cache.d_B, h_B.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        FreeMatrixCache(cache);
        return false;
    }

    std::memcpy(cache.seed_a, job.seed_a.data(), 32);
    std::memcpy(cache.seed_b, job.seed_b.data(), 32);
    cache.n = job.n;
    cache.device = device;

    *d_A = cache.d_A;
    *d_B = cache.d_B;
    return true;
}

CudaJobParams MakeCudaJobParams(const btx::pow::MatMulJob& job, const std::vector<uint8_t>& target)
{
    CudaJobParams p{};
    p.version = job.version;
    p.time = job.time;
    p.bits = job.bits;
    p.n = job.n;
    p.b = job.b;
    p.r = job.r;
    p.epsilon_bits = job.epsilon_bits;
    p.block_height = job.block_height;
    std::memcpy(p.prev_hash, job.prev_hash.data(), 32);
    std::memcpy(p.merkle_root, job.merkle_root.data(), 32);
    std::memcpy(p.seed_a, job.seed_a.data(), 32);
    std::memcpy(p.seed_b, job.seed_b.data(), 32);
    const std::vector<uint8_t>& use_target =
        target.size() == 32 ? target : job.target;
    if (use_target.size() == 32) {
        std::memcpy(p.target, use_target.data(), 32);
        const std::vector<uint8_t>& prehash_base =
            job.block_target.size() == 32 ? job.block_target : use_target;
        const auto pre_hash = btx::pow::PreHashTargetShift(prehash_base, job.epsilon_bits);
        if (pre_hash.size() == 32) {
            std::memcpy(p.pre_hash_target, pre_hash.data(), 32);
        }
    }
    return p;
}

size_t WorkspaceUint32Count(uint32_t n, uint32_t r, uint32_t bsz, bool use_v2_seeds)
{
    const size_t nn = static_cast<size_t>(n) * n;
    const size_t noise_elems = 2 * (static_cast<size_t>(n) * r + static_cast<size_t>(r) * n);
    const size_t scratch = static_cast<size_t>(bsz) * bsz;
    const size_t v2_base = use_v2_seeds ? (2 * nn) : 0;
    return v2_base + nn + noise_elems + scratch + scratch * 5;
}

} // namespace

extern "C" bool LaunchMatMulTranscriptBatch(
    int device,
    const btx::pow::MatMulJob& job,
    const std::vector<uint64_t>& nonces,
    const std::vector<uint8_t>& target,
    std::vector<btx::uint256>& out_digests,
    std::vector<bool>& out_found)
{
    if (nonces.empty()) {
        return false;
    }

    if (cudaSetDevice(device) != cudaSuccess) {
        return false;
    }

    const bool use_v2 = job.block_height >= btx::pow::kMatMulSeedV2Height;
    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    // Post-v2 work derives A/B per nonce on device; skip host matrix upload.
    if (!use_v2 && !EnsureMatricesOnDevice(device, job, &d_A, &d_B)) {
        return false;
    }

    CudaJobParams h_params = MakeCudaJobParams(job, target);

    const size_t batch = nonces.size();
    const size_t ws_uint32_per_nonce = WorkspaceUint32Count(job.n, job.r, job.b, use_v2);
    const size_t ws_bytes = batch * ws_uint32_per_nonce * sizeof(uint32_t);

    out_digests.resize(batch);
    out_found.assign(batch, false);

    if (!EnsureLaunchPool(device, batch, ws_bytes)) {
        return false;
    }
    DeviceLaunchPool& pool = g_launch_pool[device];

    std::vector<uint8_t> h_found(batch, 0);
    std::vector<uint8_t> h_digests(batch * 32);

    if (cudaMemset(pool.d_found, 0, batch) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(pool.d_nonces, nonces.data(), batch * sizeof(uint64_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    matmul_nonce_kernel<<<static_cast<int>(batch), kBlockThreads>>>(
        h_params, d_A, d_B, pool.d_nonces, pool.d_workspace, pool.d_digests, pool.d_found);

    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(h_found.data(), pool.d_found, batch, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    bool any_hit = false;
    for (size_t i = 0; i < batch; ++i) {
        out_found[i] = h_found[i] != 0;
        if (out_found[i]) {
            any_hit = true;
        }
    }

    if (any_hit) {
        if (cudaMemcpy(h_digests.data(), pool.d_digests, batch * 32, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            return false;
        }
        for (size_t i = 0; i < batch; ++i) {
            if (out_found[i]) {
                std::memcpy(out_digests[i].data(), h_digests.data() + i * 32, 32);
            }
        }
    }
    return true;
}

extern "C" bool CudaVerifyAgainstCpu(
    const btx::pow::MatMulJob& job,
    uint64_t nonce,
    const std::vector<uint8_t>& target)
{
    if (cudaGetDeviceCount(nullptr) <= 0) {
        return true;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }

    std::vector<uint64_t> nonces = {nonce};
    std::vector<btx::uint256> digests;
    std::vector<bool> found;
    if (!LaunchMatMulTranscriptBatch(device, job, nonces, target, digests, found)) {
        return false;
    }

    btx::uint256 cpu_digest;
    btx::pow::VerifySolution(job, nonce, job.time, cpu_digest);
    const bool gpu_hit = !found.empty() && found[0];

    if (digests[0] != cpu_digest) {
        return false;
    }
    const bool cpu_hit = btx::pow::DigestMeetsTarget(cpu_digest, target);
    if (cpu_hit != gpu_hit) {
        return false;
    }
    return true;
}

#endif // __CUDACC__