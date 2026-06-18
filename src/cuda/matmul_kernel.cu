// CUDA implementation for BTX MatMul PoW transcript search.
// One CUDA block per nonce: full sigma derivation, noise, blocked matmul,
// running transcript compression, and target check on device.
// Host uploads constant A/B (FromSeed) once per job; CPU VerifySolution
// cross-checks any hits before submission.

#include "cuda/gpu_sha256.h"
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
    int64_t parent_mtp;
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

__device__ void d_to_canonical(const uint8_t* internal, uint8_t* canonical)
{
    for (int i = 0; i < 32; ++i) {
        canonical[i] = internal[31 - i];
    }
}


__device__ __constant__ uint32_t kSha256K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

__device__ __forceinline__ void d_set_sha_byte(uint32_t w[16], uint32_t offset, uint32_t byte)
{
    const uint32_t word_index = offset >> 2U;
    const uint32_t shift = (3U - (offset & 3U)) * 8U;
    w[word_index] |= (byte & 0xffU) << shift;
}

__device__ __forceinline__ uint32_t d_bswap32(uint32_t x)
{
    return ((x & 0x000000ffU) << 24U) |
           ((x & 0x0000ff00U) << 8U) |
           ((x & 0x00ff0000U) >> 8U) |
           ((x & 0xff000000U) >> 24U);
}

__device__ __forceinline__ void d_sha256_init_words(uint32_t state[8])
{
    state[0] = 0x6a09e667U;
    state[1] = 0xbb67ae85U;
    state[2] = 0x3c6ef372U;
    state[3] = 0xa54ff53aU;
    state[4] = 0x510e527fU;
    state[5] = 0x9b05688cU;
    state[6] = 0x1f83d9abU;
    state[7] = 0x5be0cd19U;
}

__device__ __forceinline__ void d_sha256_compress(uint32_t state[8], uint32_t w[16])
{
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
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + kSha256K[t] + wt;
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

__device__ void d_sha256_bytes(const uint8_t* message, uint32_t message_len, uint8_t out[32])
{
    uint32_t state[8];
    d_sha256_init_words(state);

    const uint32_t total_blocks = (message_len + 9U + 63U) / 64U;
    const uint64_t bit_len = static_cast<uint64_t>(message_len) * 8U;
    for (uint32_t block = 0; block < total_blocks; ++block) {
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
        d_sha256_compress(state, w);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        out[i * 4U] = static_cast<uint8_t>((state[i] >> 24U) & 0xffU);
        out[i * 4U + 1U] = static_cast<uint8_t>((state[i] >> 16U) & 0xffU);
        out[i * 4U + 2U] = static_cast<uint8_t>((state[i] >> 8U) & 0xffU);
        out[i * 4U + 3U] = static_cast<uint8_t>(state[i] & 0xffU);
    }
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
    d_sha256_compress(state, w);
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
        d_sha256_compress(state, w);
    }

    for (uint32_t i = 0; i < 8; ++i) {
        out[i * 4U] = static_cast<uint8_t>((state[i] >> 24U) & 0xffU);
        out[i * 4U + 1U] = static_cast<uint8_t>((state[i] >> 16U) & 0xffU);
        out[i * 4U + 2U] = static_cast<uint8_t>((state[i] >> 8U) & 0xffU);
        out[i * 4U + 3U] = static_cast<uint8_t>(state[i] & 0xffU);
    }
}

__device__ __forceinline__ void d_pack_seed_words(const uint8_t* seed_internal, uint32_t seed_w[8])
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        seed_w[i] = 0U;
    }
    for (uint32_t i = 0; i < 32; ++i) {
        d_set_sha_byte(seed_w, i, seed_bytes[i]);
    }
}

__device__ void d_compute_seed_midstate(const uint8_t* seed_internal, uint32_t midstate[16])
{
    uint32_t w[8];
    d_pack_seed_words(seed_internal, w);
    uint32_t a = 0x6a09e667U, b = 0xbb67ae85U, c = 0x3c6ef372U, d = 0xa54ff53aU;
    uint32_t e = 0x510e527fU, f = 0x9b05688cU, g = 0x1f83d9abU, h = 0x5be0cd19U;
    #pragma unroll
    for (uint32_t t = 0; t < 8; ++t) {
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + kSha256K[t] + w[t];
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
    midstate[0] = w[0];
    midstate[1] = w[1];
    midstate[2] = w[2];
    midstate[3] = w[3];
    midstate[4] = w[4];
    midstate[5] = w[5];
    midstate[6] = w[6];
    midstate[7] = w[7];
    midstate[8] = a;
    midstate[9] = b;
    midstate[10] = c;
    midstate[11] = d;
    midstate[12] = e;
    midstate[13] = f;
    midstate[14] = g;
    midstate[15] = h;
}

__device__ __forceinline__ uint32_t d_candidate_from_midstate(const uint32_t* mb, uint32_t index)
{
    uint32_t w[16];
    w[0] = mb[0];
    w[1] = mb[1];
    w[2] = mb[2];
    w[3] = mb[3];
    w[4] = mb[4];
    w[5] = mb[5];
    w[6] = mb[6];
    w[7] = mb[7];
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

    uint32_t a = mb[8], b = mb[9], c = mb[10], d = mb[11];
    uint32_t e = mb[12], f = mb[13], g = mb[14], h = mb[15];
    #pragma unroll
    for (uint32_t t = 8; t < 64; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            wt = d_sig1(w[(t - 2) & 15U]) + w[(t - 7) & 15U] + d_sig0(w[(t - 15) & 15U]) + w[(t - 16) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + kSha256K[t] + wt;
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
    return d_bswap32(0x6a09e667U + a) & kFieldModulus;
}

__device__ __forceinline__ uint32_t d_candidate_from_seed_index(
    const uint8_t* seed_internal,
    uint32_t index,
    bool with_retry,
    uint32_t retry)
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);

    uint32_t w[16] = {};
    for (uint32_t i = 0; i < 32; ++i) {
        d_set_sha_byte(w, i, seed_bytes[i]);
    }
    d_set_sha_byte(w, 32U, index & 0xffU);
    d_set_sha_byte(w, 33U, (index >> 8U) & 0xffU);
    d_set_sha_byte(w, 34U, (index >> 16U) & 0xffU);
    d_set_sha_byte(w, 35U, (index >> 24U) & 0xffU);

    uint32_t message_len = 36U;
    if (with_retry) {
        d_set_sha_byte(w, 36U, retry & 0xffU);
        d_set_sha_byte(w, 37U, (retry >> 8U) & 0xffU);
        d_set_sha_byte(w, 38U, (retry >> 16U) & 0xffU);
        d_set_sha_byte(w, 39U, (retry >> 24U) & 0xffU);
        message_len = 40U;
    }

    d_set_sha_byte(w, message_len, 0x80U);
    w[15] = message_len * 8U;

    uint32_t state[8];
    d_sha256_init_words(state);
    d_sha256_compress(state, w);
    return d_bswap32(state[0]) & kFieldModulus;
}

__device__ __forceinline__ uint32_t d_fallback_candidate(const uint8_t* seed_internal, uint32_t index)
{
    uint8_t seed_bytes[32];
    d_to_canonical(seed_internal, seed_bytes);

    uint32_t w[16] = {};
    for (uint32_t i = 0; i < 32; ++i) {
        d_set_sha_byte(w, i, seed_bytes[i]);
    }
    d_set_sha_byte(w, 32U, index & 0xffU);
    d_set_sha_byte(w, 33U, (index >> 8U) & 0xffU);
    d_set_sha_byte(w, 34U, (index >> 16U) & 0xffU);
    d_set_sha_byte(w, 35U, (index >> 24U) & 0xffU);

    constexpr uint8_t fallback_tag[15] = {
        'o', 'r', 'a', 'c', 'l', 'e', '-', 'f', 'a', 'l', 'l', 'b', 'a', 'c', 'k'
    };
    for (uint32_t i = 0; i < 15; ++i) {
        d_set_sha_byte(w, 36U + i, fallback_tag[i]);
    }

    d_set_sha_byte(w, 51U, 0x80U);
    w[15] = 51U * 8U;

    uint32_t state[8];
    d_sha256_init_words(state);
    d_sha256_compress(state, w);
    return d_bswap32(state[0]) % kFieldModulus;
}

__device__ __forceinline__ uint32_t d_from_oracle_fast(const uint8_t* seed_internal, uint32_t index)
{
    for (uint32_t retry = 0; retry < 256; ++retry) {
        const uint32_t candidate = retry == 0
            ? d_candidate_from_seed_index(seed_internal, index, false, 0U)
            : d_candidate_from_seed_index(seed_internal, index, true, retry);
        if (candidate < kFieldModulus) {
            return candidate;
        }
    }
    return d_fallback_candidate(seed_internal, index);
}

__device__ __forceinline__ uint32_t d_from_oracle_midstate(
    const uint32_t* midstate,
    const uint8_t* seed_internal,
    uint32_t index)
{
    const uint32_t candidate = d_candidate_from_midstate(midstate, index);
    return candidate < kFieldModulus ? candidate : d_from_oracle_fast(seed_internal, index);
}

__device__ __forceinline__ void d_append_byte(uint8_t* message, uint32_t& offset, uint8_t value)
{
    message[offset++] = value;
}

__device__ __forceinline__ void d_append_bytes(
    uint8_t* message,
    uint32_t& offset,
    const uint8_t* data,
    uint32_t size)
{
    for (uint32_t i = 0; i < size; ++i) {
        message[offset++] = data[i];
    }
}

__device__ __forceinline__ void d_append_le16(uint8_t* message, uint32_t& offset, uint16_t value)
{
    d_append_byte(message, offset, static_cast<uint8_t>(value & 0xffU));
    d_append_byte(message, offset, static_cast<uint8_t>((value >> 8U) & 0xffU));
}

__device__ __forceinline__ void d_append_le32(uint8_t* message, uint32_t& offset, uint32_t value)
{
    d_append_byte(message, offset, static_cast<uint8_t>(value & 0xffU));
    d_append_byte(message, offset, static_cast<uint8_t>((value >> 8U) & 0xffU));
    d_append_byte(message, offset, static_cast<uint8_t>((value >> 16U) & 0xffU));
    d_append_byte(message, offset, static_cast<uint8_t>((value >> 24U) & 0xffU));
}

__device__ __forceinline__ void d_append_le64(uint8_t* message, uint32_t& offset, uint64_t value)
{
    for (uint32_t i = 0; i < 8; ++i) {
        d_append_byte(message, offset, static_cast<uint8_t>((value >> (i * 8U)) & 0xffU));
    }
}

__device__ uint32_t d_build_matmul_seed_message(
    const CudaJobParams& job,
    uint64_t nonce64,
    uint8_t which,
    uint8_t message[128])
{
    uint32_t offset = 0;
    static const char kTagV2[] = "BTX_MATMUL_SEED_V2";
    static const char kTagV3[] = "BTX_MATMUL_SEED_V3";
    const bool use_v3 = job.block_height >= btx::pow::kMatMulSeedV3Height;
    const char* tag = use_v3 ? kTagV3 : kTagV2;
    d_append_byte(message, offset, 18U);
    d_append_bytes(message, offset, reinterpret_cast<const uint8_t*>(tag), 18U);
    d_append_bytes(message, offset, job.prev_hash, 32U);
    if (use_v3) {
        d_append_le64(message, offset, static_cast<uint64_t>(job.parent_mtp));
    }
    d_append_le32(message, offset, job.block_height);
    d_append_le32(message, offset, static_cast<uint32_t>(job.version));
    d_append_bytes(message, offset, job.merkle_root, 32U);
    d_append_le32(message, offset, job.time);
    d_append_le32(message, offset, job.bits);
    d_append_le64(message, offset, nonce64);
    d_append_le16(message, offset, static_cast<uint16_t>(job.n));
    d_append_byte(message, offset, which);
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
    d_append_le32(message, offset, static_cast<uint32_t>(job.version));
    d_append_bytes(message, offset, job.prev_hash, 32U);
    d_append_bytes(message, offset, job.merkle_root, 32U);
    d_append_le32(message, offset, job.time);
    d_append_le32(message, offset, job.bits);
    d_append_le64(message, offset, nonce64);
    d_append_le16(message, offset, static_cast<uint16_t>(job.n));
    d_append_bytes(message, offset, seed_a, 32U);
    d_append_bytes(message, offset, seed_b, 32U);
    return offset;
}

__device__ void d_matmul_seed_fast(
    const uint32_t seed_midstate[8],
    const CudaJobParams& job,
    uint64_t nonce64,
    uint8_t which,
    uint8_t out_internal[32])
{
    uint8_t message[128];
    const uint32_t len = d_build_matmul_seed_message(job, nonce64, which, message);
    d_sha256_bytes_from_midstate(seed_midstate, message, len, out_internal);
}

__device__ void d_derive_sigma_fast(
    const uint32_t header_midstate[8],
    const CudaJobParams& job,
    uint64_t nonce64,
    const uint8_t* seed_a,
    const uint8_t* seed_b,
    uint8_t* sigma_internal)
{
    uint8_t message[150];
    const uint32_t len = d_build_header_hash_message(job, nonce64, seed_a, seed_b, message);
    uint8_t header_hash[32];
    d_sha256_bytes_from_midstate(header_midstate, message, len, header_hash);
    d_sha256_bytes(header_hash, 32U, sigma_internal);
}

__device__ void d_init_scan_midstates(
    const CudaJobParams& job,
    uint32_t seed_midstate[8],
    uint32_t header_midstate[8])
{
    uint8_t prefix[150];
    d_build_matmul_seed_message(job, 0U, 0U, prefix);
    d_sha256_block0_midstate(prefix, seed_midstate);
    d_build_header_hash_message(job, 0U, job.prev_hash, job.prev_hash, prefix);
    d_sha256_block0_midstate(prefix, header_midstate);
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

constexpr uint32_t kReduceInterval = 4;

__device__ __forceinline__ uint64_t d_reduce64(uint64_t acc)
{
    uint64_t fold = (acc & kFieldModulus) + (acc >> 31);
    return (fold & kFieldModulus) + (fold >> 31);
}

__host__ __device__ __forceinline__ bool d_use_factored_path(uint32_t n, uint32_t bsz)
{
    if ((n % 32U) != 0U) {
        return false;
    }
    const uint32_t blocks_per_axis = n / bsz;
    return (blocks_per_axis % 2U) == 0U;
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

__device__ void d_fill_rect(
    uint32_t* out,
    uint32_t rows,
    uint32_t cols,
    const uint8_t* seed_internal,
    const uint32_t* seed_midstate)
{
    const uint32_t total = rows * cols;
    for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        out[idx] = seed_midstate
            ? d_from_oracle_midstate(seed_midstate, seed_internal, idx)
            : d_from_oracle_fast(seed_internal, idx);
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

__device__ void d_build_compress_vec(const uint8_t* sigma_internal, uint32_t bsz, uint32_t* compress_v)
{
    __shared__ uint8_t s_seed_internal[32];
    __shared__ uint32_t s_seed_midstate[16];

    if (threadIdx.x == 0) {
        uint8_t sigma_canonical[32];
        d_to_canonical(sigma_internal, sigma_canonical);

        uint8_t message[64];
        uint32_t offset = 0;
        static const char kTag[] = "matmul-compress-v1";
        d_append_bytes(message, offset, reinterpret_cast<const uint8_t*>(kTag), 18U);
        d_append_bytes(message, offset, sigma_canonical, 32U);
        uint8_t seedb[32];
        d_sha256_bytes(message, offset, seedb);

        for (int i = 0; i < 32; ++i) {
            s_seed_internal[i] = seedb[31 - i];
        }
        d_compute_seed_midstate(s_seed_internal, s_seed_midstate);
    }
    __syncthreads();

    const uint32_t len = bsz * bsz;
    for (uint32_t k = threadIdx.x; k < len; k += blockDim.x) {
        compress_v[k] = d_from_oracle_midstate(s_seed_midstate, s_seed_internal, k);
    }
}

__device__ void d_perturb_matrix(
    uint32_t* matrix,
    const uint32_t* base,
    const uint32_t* noise_left,
    const uint32_t* noise_right,
    uint32_t n,
    uint32_t r)
{
    const uint32_t total = n * n;
    for (uint32_t gid = threadIdx.x; gid < total; gid += blockDim.x) {
        const uint32_t row = gid / n;
        const uint32_t col = gid % n;
        uint64_t acc = 0;
        uint32_t pending = 0;
        for (uint32_t k = 0; k < r; ++k) {
            acc += static_cast<uint64_t>(noise_left[row * r + k]) * noise_right[k * n + col];
            if (++pending == kReduceInterval) {
                acc = d_reduce64(acc);
                pending = 0;
            }
        }
        matrix[gid] = d_add(base[gid], static_cast<uint32_t>(d_reduce64(acc)));
    }
}

__device__ void d_build_factored_rhs(
    const uint32_t* matrix_b,
    const uint32_t* compress_v,
    uint32_t* rhs,
    uint32_t n,
    uint32_t bsz,
    uint32_t blocks_per_axis)
{
    const uint32_t rhs_elems = bsz * n * blocks_per_axis;
    for (uint32_t gid = threadIdx.x; gid < rhs_elems; gid += blockDim.x) {
        const uint32_t m = gid % n;
        const uint32_t jx = gid / n;
        const uint32_t x = jx % bsz;
        const uint32_t j = jx / bsz;
        const uint32_t* b_row = matrix_b + static_cast<size_t>(m) * n + j * bsz;
        const uint32_t* w_row = compress_v + x * bsz;
        uint64_t acc = 0;
        uint32_t pending = 0;
        for (uint32_t y = 0; y < bsz; ++y) {
            acc += static_cast<uint64_t>(w_row[y]) * b_row[y];
            if (++pending == kReduceInterval) {
                acc = d_reduce64(acc);
                pending = 0;
            }
        }
        rhs[gid] = static_cast<uint32_t>(d_reduce64(acc));
    }
}

__device__ void d_compute_factored_words(
    const uint32_t* matrix_a,
    const uint32_t* rhs,
    uint32_t* output,
    uint32_t n,
    uint32_t bsz,
    uint32_t blocks_per_axis)
{
    const uint32_t tiles_per_axis = blocks_per_axis >> 1U;
    const uint32_t total_tiles = tiles_per_axis * tiles_per_axis;
    const uint32_t warps_per_block = blockDim.x >> 5U;
    __shared__ uint64_t s_lane_partials[8][32][4];

    for (uint32_t tile_base = 0; tile_base < total_tiles; tile_base += warps_per_block) {
        const uint32_t lane = threadIdx.x & 31U;
        const uint32_t warp_id = threadIdx.x >> 5U;
        const uint32_t tile_index = tile_base + warp_id;
        const bool active = tile_index < total_tiles;

        uint64_t acc00 = 0;
        uint64_t acc01 = 0;
        uint64_t acc10 = 0;
        uint64_t acc11 = 0;
        if (active) {
            const uint32_t j0 = (tile_index % tiles_per_axis) * 2U;
            const uint32_t i0 = (tile_index / tiles_per_axis) * 2U;
            uint32_t pending = 0;
            for (uint32_t x = 0; x < bsz; ++x) {
                const uint32_t* a_row0 = matrix_a + static_cast<size_t>(i0 * bsz + x) * n;
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
                    if (++pending == kReduceInterval) {
                        acc00 = d_reduce64(acc00);
                        acc01 = d_reduce64(acc01);
                        acc10 = d_reduce64(acc10);
                        acc11 = d_reduce64(acc11);
                        pending = 0;
                    }
                }
            }
            s_lane_partials[warp_id][lane][0] = d_reduce64(acc00);
            s_lane_partials[warp_id][lane][1] = d_reduce64(acc01);
            s_lane_partials[warp_id][lane][2] = d_reduce64(acc10);
            s_lane_partials[warp_id][lane][3] = d_reduce64(acc11);
        }
        __syncthreads();

        if (lane == 0 && active) {
            const uint32_t j0 = (tile_index % tiles_per_axis) * 2U;
            const uint32_t i0 = (tile_index / tiles_per_axis) * 2U;
            uint64_t sum00 = 0;
            uint64_t sum01 = 0;
            uint64_t sum10 = 0;
            uint64_t sum11 = 0;
            for (uint32_t i = 0; i < 32U; ++i) {
                sum00 += s_lane_partials[warp_id][i][0];
                sum01 += s_lane_partials[warp_id][i][1];
                sum10 += s_lane_partials[warp_id][i][2];
                sum11 += s_lane_partials[warp_id][i][3];
            }
            output[i0 * blocks_per_axis + j0] = static_cast<uint32_t>(d_reduce64(sum00));
            output[i0 * blocks_per_axis + j0 + 1U] = static_cast<uint32_t>(d_reduce64(sum01));
            output[(i0 + 1U) * blocks_per_axis + j0] = static_cast<uint32_t>(d_reduce64(sum10));
            output[(i0 + 1U) * blocks_per_axis + j0 + 1U] = static_cast<uint32_t>(d_reduce64(sum11));
        }
        __syncthreads();
    }
}

__device__ void d_copy_matrix(uint32_t* dst, const uint32_t* src, uint32_t count)
{
    for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__device__ bool d_solve_nonce(
    const CudaJobParams& job,
    uint32_t* A,
    uint32_t* B,
    const uint32_t* A_base,
    const uint32_t* B_base,
    uint64_t nonce64,
    uint32_t* C,
    uint32_t* E_L,
    uint32_t* E_R,
    uint32_t* F_L,
    uint32_t* F_R,
    uint32_t* compress_v,
    uint32_t* factored_rhs,
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
    __shared__ uint32_t s_scan_seed_midstate[8];
    __shared__ uint32_t s_scan_header_midstate[8];
    __shared__ uint32_t s_mat_a_midstate[16];
    __shared__ uint32_t s_mat_b_midstate[16];
    __shared__ uint32_t s_noise_midstate[16];
    __shared__ uint32_t s_tx_state[8];
    __shared__ uint32_t s_tx_buf[16];
    __shared__ uint32_t s_tx_count;

    const bool use_v2 = job.block_height >= 125000;
    const bool use_v3 = job.block_height >= btx::pow::kMatMulSeedV3Height;
    const bool use_factored = use_v2 ? false : d_use_factored_path(job.n, job.b);

    if (threadIdx.x == 0 && use_v2) {
        d_init_scan_midstates(job, s_scan_seed_midstate, s_scan_header_midstate);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (reuse_seed_a && reuse_seed_b) {
            for (int i = 0; i < 32; ++i) {
                s_seed_a[i] = reuse_seed_a[i];
                s_seed_b[i] = reuse_seed_b[i];
            }
        } else if (use_v2) {
            d_matmul_seed_fast(s_scan_seed_midstate, job, nonce64, 0, s_seed_a);
            d_matmul_seed_fast(s_scan_seed_midstate, job, nonce64, 1, s_seed_b);
        } else {
            for (int i = 0; i < 32; ++i) {
                s_seed_a[i] = job.seed_a[i];
                s_seed_b[i] = job.seed_b[i];
            }
        }
        if (use_v2) {
            d_derive_sigma_fast(
                s_scan_header_midstate, job, nonce64, s_seed_a, s_seed_b, s_sigma);
        } else {
            d_derive_sigma(job, nonce64, s_seed_a, s_seed_b, s_sigma);
        }
        s_sigma_pass = job.epsilon_bits == 0 ||
                       d_sigma_below_prehash(s_sigma, job.pre_hash_target);
    }
    __syncthreads();
    if (!s_sigma_pass) {
        return false;
    }

    if (use_v2 && A && B) {
        if (threadIdx.x == 0) {
            d_compute_seed_midstate(s_seed_a, s_mat_a_midstate);
            d_compute_seed_midstate(s_seed_b, s_mat_b_midstate);
        }
        __syncthreads();
        d_fill_rect(A, job.n, job.n, s_seed_a, s_mat_a_midstate);
        __syncthreads();
        d_fill_rect(B, job.n, job.n, s_seed_b, s_mat_b_midstate);
        __syncthreads();
    } else if (use_factored && A && B && A_base && B_base) {
        d_copy_matrix(A, A_base, job.n * job.n);
        __syncthreads();
        d_copy_matrix(B, B_base, job.n * job.n);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_derive_noise_seed("matmul_noise_EL_v1", s_sigma, s_seed_el);
        d_derive_noise_seed("matmul_noise_ER_v1", s_sigma, s_seed_er);
        d_derive_noise_seed("matmul_noise_FL_v1", s_sigma, s_seed_fl);
        d_derive_noise_seed("matmul_noise_FR_v1", s_sigma, s_seed_fr);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed_el, s_noise_midstate);
    }
    __syncthreads();
    d_fill_rect(E_L, job.n, job.r, s_seed_el, s_noise_midstate);
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed_er, s_noise_midstate);
    }
    __syncthreads();
    d_fill_rect(E_R, job.r, job.n, s_seed_er, s_noise_midstate);
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed_fl, s_noise_midstate);
    }
    __syncthreads();
    d_fill_rect(F_L, job.n, job.r, s_seed_fl, s_noise_midstate);
    __syncthreads();

    if (threadIdx.x == 0) {
        d_compute_seed_midstate(s_seed_fr, s_noise_midstate);
    }
    __syncthreads();
    d_fill_rect(F_R, job.r, job.n, s_seed_fr, s_noise_midstate);
    __syncthreads();

    const uint32_t bsz = job.b;
    const uint32_t N = job.n / bsz;
    const uint32_t word_count = N * N;

    d_zero_u32(C, word_count);
    __syncthreads();

    d_build_compress_vec(s_sigma, job.b, compress_v);
    __syncthreads();

    if (use_factored && A && B && factored_rhs) {
        d_perturb_matrix(A, A, E_L, E_R, job.n, job.r);
        __syncthreads();
        d_perturb_matrix(B, B, F_L, F_R, job.n, job.r);
        __syncthreads();
        d_build_factored_rhs(B, compress_v, factored_rhs, job.n, bsz, N);
        __syncthreads();
        d_compute_factored_words(A, factored_rhs, C, job.n, bsz, N);
        __syncthreads();
    } else {
        const uint32_t total_steps = N * N * N;

        if (threadIdx.x == 0 && use_v2 && !use_v3) {
            d_sha256_init_words(s_tx_state);
            s_tx_count = 0;
        }
        __syncthreads();

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

                if (use_v2 && !use_v3) {
                    s_tx_buf[s_tx_count++] = d_bswap32(C[word_idx]);
                    if (s_tx_count == 16) {
                        d_sha256_compress(s_tx_state, s_tx_buf);
                        s_tx_count = 0;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        if (use_v2 && !use_v3) {
            uint64_t bit_len = static_cast<uint64_t>(word_count * N) * 32;
            if (s_tx_count == 0) {
                uint32_t pad[16] = {};
                pad[0] = 0x80000000U;
                pad[14] = static_cast<uint32_t>(bit_len >> 32);
                pad[15] = static_cast<uint32_t>(bit_len);
                d_sha256_compress(s_tx_state, pad);
            } else {
                s_tx_buf[s_tx_count] = 0x80000000U;
                for (uint32_t i = s_tx_count + 1; i < 15; ++i) {
                    s_tx_buf[i] = 0;
                }
                if (s_tx_count < 14) {
                    s_tx_buf[14] = static_cast<uint32_t>(bit_len >> 32);
                    s_tx_buf[15] = static_cast<uint32_t>(bit_len);
                    d_sha256_compress(s_tx_state, s_tx_buf);
                } else {
                    d_sha256_compress(s_tx_state, s_tx_buf);
                    uint32_t pad2[16] = {};
                    pad2[14] = static_cast<uint32_t>(bit_len >> 32);
                    pad2[15] = static_cast<uint32_t>(bit_len);
                    d_sha256_compress(s_tx_state, pad2);
                }
            }

            uint32_t inner[8];
            for (int i = 0; i < 8; ++i) inner[i] = s_tx_state[i];
            d_sha256_init_words(s_tx_state);
            uint32_t w2[16] = {};
            w2[0] = inner[0]; w2[1] = inner[1]; w2[2] = inner[2]; w2[3] = inner[3];
            w2[4] = inner[4]; w2[5] = inner[5]; w2[6] = inner[6]; w2[7] = inner[7];
            w2[8] = 0x80000000U;
            w2[15] = 0x00000100U;
            d_sha256_compress(s_tx_state, w2);

            for (int i = 0; i < 8; ++i) {
                digest_out[i*4]   = static_cast<uint8_t>((s_tx_state[i] >> 24U) & 0xffU);
                digest_out[i*4+1] = static_cast<uint8_t>((s_tx_state[i] >> 16U) & 0xffU);
                digest_out[i*4+2] = static_cast<uint8_t>((s_tx_state[i] >> 8U) & 0xffU);
                digest_out[i*4+3] = static_cast<uint8_t>(s_tx_state[i] & 0xffU);
            }

            s_hit = d_uint256_le(digest_out, job.target);
        } else {
            uint8_t c_prime[32];
            d_hash_matrix_words(C, word_count, c_prime);
            d_finalize_product_digest(s_sigma, c_prime, job.n, bsz, digest_out);
            s_hit = d_uint256_le(digest_out, job.target);
        }
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
    const bool use_v2 = params.block_height >= 125000;
    const bool use_factored = use_v2 ? false : d_use_factored_path(n, bsz);
    const size_t blocks_per_axis = static_cast<size_t>(n / bsz);
    const size_t matrix_elems = use_v2 ? (2 * nn) : (use_factored ? (2 * nn) : 0);
    const size_t factored_rhs_elems = use_factored ? (static_cast<size_t>(bsz) * n * blocks_per_axis) : 0;
    const size_t legacy_scratch_elems = use_factored ? 0 : (scratch_elems * 5);
    const size_t per_nonce_elems =
        matrix_elems + nn + noise_elems + compress_elems + factored_rhs_elems + legacy_scratch_elems;

    // Per-nonce workspace layout:
    // [A nn][B nn] (V2 or factored non-V2) [C nn][noise...][compress][factored_rhs?][legacy scratch?]
    size_t off = idx * per_nonce_elems;

    uint32_t* A_local = (use_v2 || use_factored) ? (workspaces + off) : nullptr;
    if (use_v2 || use_factored) off += nn;
    uint32_t* B_local = (use_v2 || use_factored) ? (workspaces + off) : nullptr;
    if (use_v2 || use_factored) off += nn;

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
    uint32_t* factored_rhs = use_factored ? (workspaces + off) : nullptr;
    if (use_factored) off += factored_rhs_elems;
    uint32_t* ablk = use_factored ? nullptr : (workspaces + off);
    if (!use_factored) off += scratch_elems;
    uint32_t* bblk = use_factored ? nullptr : (workspaces + off);
    if (!use_factored) off += scratch_elems;
    uint32_t* noise_blk = use_factored ? nullptr : (workspaces + off);
    if (!use_factored) off += scratch_elems;
    uint32_t* prod = use_factored ? nullptr : (workspaces + off);
    if (!use_factored) off += scratch_elems;
    uint32_t* cblk = use_factored ? nullptr : (workspaces + off);

    __shared__ uint8_t s_v2_seed_a[32];
    __shared__ uint8_t s_v2_seed_b[32];
    __shared__ uint32_t s_scan_seed_midstate[8];
    __shared__ uint32_t s_scan_header_midstate[8];

    if (use_v2 && threadIdx.x == 0) {
        d_init_scan_midstates(params, s_scan_seed_midstate, s_scan_header_midstate);
        d_matmul_seed_fast(s_scan_seed_midstate, params, nonces[idx], 0, s_v2_seed_a);
        d_matmul_seed_fast(s_scan_seed_midstate, params, nonces[idx], 1, s_v2_seed_b);
    }
    __syncthreads();

    uint32_t* use_a = (use_v2 || use_factored) ? A_local : const_cast<uint32_t*>(A);
    uint32_t* use_b = (use_v2 || use_factored) ? B_local : const_cast<uint32_t*>(B);

    uint8_t digest[32];
    const uint8_t* reuse_a = use_v2 ? s_v2_seed_a : nullptr;
    const uint8_t* reuse_b = use_v2 ? s_v2_seed_b : nullptr;
    const bool hit = d_solve_nonce(
        params, use_a, use_b, A, B, nonces[idx], C, E_L, E_R, F_L, F_R, compress_v,
        factored_rhs, ablk, bblk, noise_blk, prod, cblk, reuse_a, reuse_b, digest);

    if (threadIdx.x == 0) {
        out_found[idx] = hit ? 1 : 0;
        for (int i = 0; i < 32; ++i) {
            out_digests[idx * 32 + i] = digest[i];
        }
    }
}

__global__ void init_scan_midstates_kernel(
    CudaJobParams params,
    uint32_t* seed_midstate,
    uint32_t* header_midstate)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        d_init_scan_midstates(params, seed_midstate, header_midstate);
    }
}

__global__ void sigma_gate_kernel(
    CudaJobParams params,
    uint64_t nonce_start,
    size_t nonce_count,
    const uint32_t* __restrict__ seed_midstate,
    const uint32_t* __restrict__ header_midstate,
    uint32_t* __restrict__ out_count,
    uint32_t* __restrict__ out_indices,
    uint8_t* __restrict__ out_sigma,
    uint8_t* __restrict__ out_seed_a,
    uint8_t* __restrict__ out_seed_b)
{
    const size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= nonce_count) {
        return;
    }

    uint8_t seed_a[32];
    uint8_t seed_b[32];
    uint8_t sigma[32];
    const uint64_t nonce = nonce_start + idx;

    if (params.block_height >= 125000) {
        d_matmul_seed_fast(seed_midstate, params, nonce, 0, seed_a);
        d_matmul_seed_fast(seed_midstate, params, nonce, 1, seed_b);
        d_derive_sigma_fast(
            header_midstate, params, nonce, seed_a, seed_b, sigma);
    } else {
        for (uint32_t i = 0; i < 32; ++i) {
            seed_a[i] = params.seed_a[i];
            seed_b[i] = params.seed_b[i];
        }
        d_derive_sigma(params, nonce, seed_a, seed_b, sigma);
    }

    if (params.epsilon_bits != 0 &&
        !d_sigma_below_prehash(sigma, params.pre_hash_target)) {
        return;
    }

    const uint32_t compact_index = atomicAdd(out_count, 1u);
    out_indices[compact_index] = static_cast<uint32_t>(idx);
    for (uint32_t i = 0; i < 32; ++i) {
        out_sigma[compact_index * 32 + i] = sigma[i];
        out_seed_a[compact_index * 32 + i] = seed_a[i];
        out_seed_b[compact_index * 32 + i] = seed_b[i];
    }
}

__global__ void scatter_passed_hits_kernel(
    const uint32_t* __restrict__ passed_indices,
    const int32_t* __restrict__ passed_results,
    const uint8_t* __restrict__ passed_digests,
    uint32_t passed_count,
    uint8_t* __restrict__ out_found,
    uint8_t* __restrict__ out_digests,
    size_t launch_batch)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= passed_count) {
        return;
    }
    if (passed_results[i] == 0) {
        return;
    }
    const uint32_t idx = passed_indices[i];
    if (idx >= launch_batch) {
        return;
    }
    out_found[idx] = 1;
    uint8_t* out = out_digests + size_t(idx) * 32;
    const uint8_t* in = passed_digests + size_t(i) * 32;
    for (int j = 0; j < 32; ++j) {
        out[j] = in[j];
    }
}

__global__ void batch_perturb_matrix_kernel(
    const uint32_t* __restrict__ base,
    const uint32_t* __restrict__ noise_left,
    const uint32_t* __restrict__ noise_right,
    uint32_t n,
    uint32_t r,
    size_t matrix_elements,
    size_t total_elements,
    uint32_t* __restrict__ output)
{
    const size_t gid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    const size_t batch_index = gid / matrix_elements;
    const uint32_t local_index = static_cast<uint32_t>(gid % matrix_elements);
    const uint32_t row = local_index / n;
    const uint32_t col = local_index % n;
    const uint32_t* batch_base = base + batch_index * matrix_elements;
    const uint32_t* left = noise_left + batch_index * n * r;
    const uint32_t* right = noise_right + batch_index * r * n;

    uint64_t acc = 0;
    uint32_t pending = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        if (k < r) {
            acc += static_cast<uint64_t>(left[row * r + k]) *
                   right[k * n + col];
            if (++pending == kReduceInterval) {
                acc = d_reduce64(acc);
                pending = 0;
            }
        }
    }
    output[gid] = d_add(
        batch_base[local_index],
        static_cast<uint32_t>(d_reduce64(acc)));
}

__global__ void batch_perturb_matrix_512x8_kernel(
    const uint32_t* __restrict__ base,
    const uint32_t* __restrict__ noise_left,
    const uint32_t* __restrict__ noise_right,
    size_t total_elements,
    uint32_t* __restrict__ output)
{
    const size_t gid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_elements) {
        return;
    }

    constexpr uint32_t kN = 512;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    constexpr size_t kNoiseElements = size_t{kN} * 8;
    const size_t batch_index = gid >> 18U;
    const uint32_t local_index =
        static_cast<uint32_t>(gid & (kMatrixElements - 1U));
    const uint32_t row = local_index >> 9U;
    const uint32_t col = local_index & (kN - 1U);
    const uint32_t* batch_base = base + batch_index * kMatrixElements;
    const uint32_t* left = noise_left + batch_index * kNoiseElements;
    const uint32_t* right = noise_right + batch_index * kNoiseElements;

    uint64_t acc = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        acc += static_cast<uint64_t>(left[(row << 3U) + k]) *
               right[k * kN + col];
        if ((k & 3U) == 3U) {
            acc = d_reduce64(acc);
        }
    }
    output[gid] = d_add(
        batch_base[local_index],
        static_cast<uint32_t>(d_reduce64(acc)));
}

__global__ void batch_compressed_words_kernel(
    const uint32_t* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const uint32_t* __restrict__ compress,
    uint32_t n,
    uint32_t bsz,
    uint32_t blocks_per_axis,
    uint32_t words_per_nonce,
    size_t matrix_elements,
    uint32_t* __restrict__ output)
{
    __shared__ uint32_t partials[256];

    const uint32_t global_word = blockIdx.x;
    const uint32_t batch_index = global_word / words_per_nonce;
    const uint32_t word_index = global_word % words_per_nonce;
    const uint32_t bi = word_index / blocks_per_axis;
    const uint32_t bj = word_index % blocks_per_axis;
    const uint32_t tid = threadIdx.x;
    const bool active = tid < bsz * bsz;
    const uint32_t x = active ? tid / bsz : 0;
    const uint32_t y = active ? tid % bsz : 0;

    const uint32_t* A = matrix_a + static_cast<size_t>(batch_index) * matrix_elements;
    const uint32_t* B = matrix_b + static_cast<size_t>(batch_index) * matrix_elements;
    const uint32_t* W = compress + static_cast<size_t>(batch_index) * bsz * bsz;

    uint32_t compressed_sum = 0;
    for (uint32_t ell = 0; ell < blocks_per_axis; ++ell) {
        uint64_t dot = 0;
        uint32_t pending = 0;
        if (active) {
            const uint32_t row = bi * bsz + x;
            const uint32_t col = bj * bsz + y;
            const uint32_t middle = ell * bsz;
            #pragma unroll
            for (uint32_t k = 0; k < 16; ++k) {
                if (k < bsz) {
                    dot += static_cast<uint64_t>(
                        A[static_cast<size_t>(row) * n + middle + k]) *
                        B[static_cast<size_t>(middle + k) * n + col];
                    if (++pending == kReduceInterval) {
                        dot = d_reduce64(dot);
                        pending = 0;
                    }
                }
            }
            partials[tid] = d_mul(
                static_cast<uint32_t>(d_reduce64(dot)), W[tid]);
        } else {
            partials[tid] = 0;
        }
        __syncthreads();

        for (uint32_t stride = 128; stride > 0; stride >>= 1U) {
            if (tid < stride) {
                partials[tid] = d_add(partials[tid], partials[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            compressed_sum = d_add(compressed_sum, partials[0]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[static_cast<size_t>(batch_index) * words_per_nonce + word_index] =
            compressed_sum;
    }
}

__global__ void batch_compressed_words_512x16_kernel(
    const uint32_t* __restrict__ matrix_a,
    const uint32_t* __restrict__ matrix_b,
    const uint32_t* __restrict__ compress,
    uint32_t* __restrict__ output)
{
    __shared__ uint32_t partials[256];

    constexpr uint32_t kN = 512;
    constexpr uint32_t kWordsPerNonce = 32 * 32;
    constexpr size_t kMatrixElements = size_t{kN} * kN;
    const uint32_t global_word = blockIdx.x;
    const uint32_t batch_index = global_word >> 10U;
    const uint32_t word_index = global_word & (kWordsPerNonce - 1U);
    const uint32_t bi = word_index >> 5U;
    const uint32_t bj = word_index & 31U;
    const uint32_t tid = threadIdx.x;
    const uint32_t row = (bi << 4U) + (tid >> 4U);
    const uint32_t col = (bj << 4U) + (tid & 15U);

    const uint32_t* A =
        matrix_a + static_cast<size_t>(batch_index) * kMatrixElements;
    const uint32_t* B =
        matrix_b + static_cast<size_t>(batch_index) * kMatrixElements;
    const uint32_t weight =
        compress[(static_cast<size_t>(batch_index) << 8U) + tid];

    uint32_t compressed_sum = 0;
    for (uint32_t ell = 0; ell < 32; ++ell) {
        const uint32_t middle = ell << 4U;
        uint64_t dot = 0;
        #pragma unroll
        for (uint32_t k = 0; k < 16; ++k) {
            dot += static_cast<uint64_t>(
                A[static_cast<size_t>(row) * kN + middle + k]) *
                B[static_cast<size_t>(middle + k) * kN + col];
            if ((k & 3U) == 3U) {
                dot = d_reduce64(dot);
            }
        }
        partials[tid] = d_mul(
            static_cast<uint32_t>(d_reduce64(dot)), weight);
        __syncthreads();

        for (uint32_t stride = 128; stride > 0; stride >>= 1U) {
            if (tid < stride) {
                partials[tid] = d_add(partials[tid], partials[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            compressed_sum = d_add(compressed_sum, partials[0]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[
            (static_cast<size_t>(batch_index) << 10U) + word_index] =
            compressed_sum;
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

struct PackedJobKey {
    uint8_t prev_hash[32];
    uint8_t merkle_root[32];
    uint32_t time = 0;
    uint32_t bits = 0;
    int32_t version = 0;
    uint32_t block_height = 0;
    uint32_t epsilon_bits = 0;
    int64_t parent_mtp = 0;
    uint8_t target[32];
    uint8_t pre_hash_target[32];
};

struct DeviceLaunchPool {
    int device = -1;
    uint64_t* d_nonces = nullptr;
    uint32_t* d_workspace = nullptr;
    uint8_t* d_digests = nullptr;
    uint8_t* d_found = nullptr;
    uint32_t* d_gate_count = nullptr;
    uint32_t* d_passed_indices = nullptr;
    uint8_t* d_passed_sigma = nullptr;
    uint8_t* d_passed_seed_a = nullptr;
    uint8_t* d_passed_seed_b = nullptr;
    uint32_t* d_seed_midstate = nullptr;
    uint32_t* d_header_midstate = nullptr;
    uint32_t h_gate_count = 0;
    cudaStream_t gate_stream = nullptr;
    cudaStream_t matmul_stream = nullptr;
    cudaEvent_t gate_done = nullptr;
    cudaEvent_t matmul_done = nullptr;
    bool matmul_in_flight = false;
    bool gate_prefetched = false;
    bool pending_next_gate = false;
    uint64_t pending_start_nonce = 0;
    size_t pending_batch = 0;
    size_t nonce_cap = 0;
    size_t ws_bytes = 0;
    size_t digest_cap = 0;
    uint32_t last_gate_passed = 0;
    bool use_batched_matmul = false;
    PackedJobKey job_key{};
    bool job_midstate_ready = false;
};

DeviceLaunchPool g_launch_pool[16];

struct BatchedMatMulPool {
    int device = -1;
    uint8_t* d_noise_seeds = nullptr;
    uint8_t* d_compress_seeds = nullptr;
    uint32_t* d_seed_midstates = nullptr;
    uint32_t* d_noise_midstates = nullptr;
    uint32_t* d_compress_midstates = nullptr;
    uint32_t* d_base_a = nullptr;
    uint32_t* d_base_b = nullptr;
    uint32_t* d_matrix_a = nullptr;
    uint32_t* d_matrix_b = nullptr;
    uint32_t* d_noise_el = nullptr;
    uint32_t* d_noise_er = nullptr;
    uint32_t* d_noise_fl = nullptr;
    uint32_t* d_noise_fr = nullptr;
    uint32_t* d_compress = nullptr;
    uint32_t* d_words = nullptr;
    uint8_t* d_digests = nullptr;
    int32_t* d_results = nullptr;
    uint8_t* d_block_target = nullptr;
    uint8_t* d_share_target = nullptr;
    cudaStream_t stream = nullptr;
    size_t capacity = 0;
    uint8_t cached_block_target[32]{};
    uint8_t cached_share_target[32]{};
    bool targets_ready = false;
};

BatchedMatMulPool g_batched_pool[16];

void FreeBatchedPool(BatchedMatMulPool& pool)
{
    if (pool.stream) {
        cudaStreamDestroy(pool.stream);
    }
    cudaFree(pool.d_noise_seeds);
    cudaFree(pool.d_compress_seeds);
    cudaFree(pool.d_seed_midstates);
    cudaFree(pool.d_noise_midstates);
    cudaFree(pool.d_compress_midstates);
    cudaFree(pool.d_base_a);
    cudaFree(pool.d_base_b);
    cudaFree(pool.d_matrix_a);
    cudaFree(pool.d_matrix_b);
    cudaFree(pool.d_noise_el);
    cudaFree(pool.d_noise_er);
    cudaFree(pool.d_noise_fl);
    cudaFree(pool.d_noise_fr);
    cudaFree(pool.d_compress);
    cudaFree(pool.d_words);
    cudaFree(pool.d_digests);
    cudaFree(pool.d_results);
    cudaFree(pool.d_block_target);
    cudaFree(pool.d_share_target);
    pool = {};
}

template <typename T>
bool CudaAlloc(T*& ptr, size_t count)
{
    return cudaMalloc(&ptr, count * sizeof(T)) == cudaSuccess;
}

bool EnsureBatchedPool(
    int device,
    size_t batch,
    uint32_t n,
    uint32_t r,
    uint32_t bsz)
{
    if (device < 0 || device >= 16) {
        return false;
    }
    BatchedMatMulPool& pool = g_batched_pool[device];
    if (pool.device == device && pool.capacity >= batch) {
        return true;
    }
    FreeBatchedPool(pool);
    pool.device = device;
    pool.capacity = batch;

    const size_t matrix = batch * static_cast<size_t>(n) * n;
    const size_t noise_left = batch * static_cast<size_t>(n) * r;
    const size_t noise_right = batch * static_cast<size_t>(r) * n;
    const size_t compress = batch * static_cast<size_t>(bsz) * bsz;
    const size_t words = batch * static_cast<size_t>(n / bsz) * (n / bsz);

    const bool ok =
        CudaAlloc(pool.d_noise_seeds, batch * 4 * 32) &&
        CudaAlloc(pool.d_compress_seeds, batch * 32) &&
        CudaAlloc(pool.d_seed_midstates, batch * 16 * 2) &&
        CudaAlloc(pool.d_noise_midstates, batch * 4 * 16) &&
        CudaAlloc(pool.d_compress_midstates, batch * 16) &&
        CudaAlloc(pool.d_base_a, matrix) &&
        CudaAlloc(pool.d_base_b, matrix) &&
        CudaAlloc(pool.d_matrix_a, matrix) &&
        CudaAlloc(pool.d_matrix_b, matrix) &&
        CudaAlloc(pool.d_noise_el, noise_left) &&
        CudaAlloc(pool.d_noise_er, noise_right) &&
        CudaAlloc(pool.d_noise_fl, noise_left) &&
        CudaAlloc(pool.d_noise_fr, noise_right) &&
        CudaAlloc(pool.d_compress, compress) &&
        CudaAlloc(pool.d_words, words) &&
        CudaAlloc(pool.d_digests, batch * 32) &&
        CudaAlloc(pool.d_results, batch) &&
        CudaAlloc(pool.d_block_target, 32) &&
        CudaAlloc(pool.d_share_target, 32);
    if (!ok) {
        FreeBatchedPool(pool);
    } else if (!pool.stream) {
        cudaStreamCreate(&pool.stream);
    }
    return ok;
}

void FreeLaunchPool(DeviceLaunchPool& pool)
{
    if (pool.gate_stream) {
        cudaStreamDestroy(pool.gate_stream);
    }
    if (pool.matmul_stream) {
        cudaStreamDestroy(pool.matmul_stream);
    }
    if (pool.gate_done) {
        cudaEventDestroy(pool.gate_done);
    }
    if (pool.matmul_done) {
        cudaEventDestroy(pool.matmul_done);
    }
    if (pool.d_nonces) cudaFree(pool.d_nonces);
    if (pool.d_workspace) cudaFree(pool.d_workspace);
    if (pool.d_digests) cudaFree(pool.d_digests);
    if (pool.d_found) cudaFree(pool.d_found);
    if (pool.d_gate_count) cudaFree(pool.d_gate_count);
    if (pool.d_passed_indices) cudaFree(pool.d_passed_indices);
    if (pool.d_passed_sigma) cudaFree(pool.d_passed_sigma);
    if (pool.d_passed_seed_a) cudaFree(pool.d_passed_seed_a);
    if (pool.d_passed_seed_b) cudaFree(pool.d_passed_seed_b);
    if (pool.d_seed_midstate) cudaFree(pool.d_seed_midstate);
    if (pool.d_header_midstate) cudaFree(pool.d_header_midstate);
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
        if (pool.d_gate_count) cudaFree(pool.d_gate_count);
        if (pool.d_passed_indices) cudaFree(pool.d_passed_indices);
        if (pool.d_passed_sigma) cudaFree(pool.d_passed_sigma);
        if (pool.d_passed_seed_a) cudaFree(pool.d_passed_seed_a);
        if (pool.d_passed_seed_b) cudaFree(pool.d_passed_seed_b);
        if (pool.d_seed_midstate) cudaFree(pool.d_seed_midstate);
        if (pool.d_header_midstate) cudaFree(pool.d_header_midstate);
        pool.d_digests = nullptr;
        pool.d_found = nullptr;
        pool.d_gate_count = nullptr;
        pool.d_passed_indices = nullptr;
        pool.d_passed_sigma = nullptr;
        pool.d_passed_seed_a = nullptr;
        pool.d_passed_seed_b = nullptr;
        pool.d_seed_midstate = nullptr;
        pool.d_header_midstate = nullptr;
        if (cudaMalloc(&pool.d_digests, digest_bytes) != cudaSuccess ||
            cudaMalloc(&pool.d_found, batch) != cudaSuccess ||
            cudaMalloc(&pool.d_gate_count, sizeof(uint32_t)) != cudaSuccess ||
            cudaMalloc(&pool.d_passed_indices, batch * sizeof(uint32_t)) != cudaSuccess ||
            cudaMalloc(&pool.d_passed_sigma, batch * 32) != cudaSuccess ||
            cudaMalloc(&pool.d_passed_seed_a, batch * 32) != cudaSuccess ||
            cudaMalloc(&pool.d_passed_seed_b, batch * 32) != cudaSuccess ||
            cudaMalloc(&pool.d_seed_midstate, 8 * sizeof(uint32_t)) != cudaSuccess ||
            cudaMalloc(&pool.d_header_midstate, 8 * sizeof(uint32_t)) != cudaSuccess) {
            FreeLaunchPool(pool);
            return false;
        }
        pool.digest_cap = batch;
    }
    if (pool.d_nonces && pool.d_workspace && pool.d_digests && pool.d_found &&
        pool.d_gate_count && pool.d_passed_indices && pool.d_passed_sigma &&
        pool.d_passed_seed_a && pool.d_passed_seed_b &&
        pool.d_seed_midstate && pool.d_header_midstate) {
        if (!pool.gate_stream) {
            cudaStreamCreate(&pool.gate_stream);
        }
        if (!pool.matmul_stream) {
            cudaStreamCreate(&pool.matmul_stream);
        }
        if (!pool.gate_done) {
            cudaEventCreateWithFlags(&pool.gate_done, cudaEventDisableTiming);
        }
        if (!pool.matmul_done) {
            cudaEventCreateWithFlags(&pool.matmul_done, cudaEventDisableTiming);
        }
    }
    return pool.d_nonces && pool.d_workspace && pool.d_digests && pool.d_found &&
           pool.d_gate_count && pool.d_passed_indices && pool.d_passed_sigma &&
           pool.d_passed_seed_a && pool.d_passed_seed_b &&
           pool.d_seed_midstate && pool.d_header_midstate &&
           pool.gate_stream && pool.matmul_stream &&
           pool.gate_done && pool.matmul_done;
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

bool ProcessPassedNoncesBatched(
    int device,
    const btx::pow::MatMulJob& job,
    const std::vector<uint8_t>& share_target,
    size_t batch,
    const uint8_t* d_seed_a,
    const uint8_t* d_seed_b,
    const uint8_t* d_sigma,
    cudaStream_t stream,
    bool sync_at_end,
    std::vector<btx::uint256>* digests_out,
    std::vector<uint8_t>* found_out)
{
    if (batch == 0) {
        return true;
    }
    if (!EnsureBatchedPool(device, batch, job.n, job.r, job.b)) {
        return false;
    }
    BatchedMatMulPool& pool = g_batched_pool[device];

    const std::vector<uint8_t>& block_target =
        job.block_target.size() == 32 ? job.block_target : share_target;
    if (share_target.size() != 32 || block_target.size() != 32) {
        return false;
    }

    const bool targets_changed =
        !pool.targets_ready ||
        std::memcmp(pool.cached_block_target, block_target.data(), 32) != 0 ||
        std::memcmp(pool.cached_share_target, share_target.data(), 32) != 0;
    if (targets_changed) {
        if (cudaMemcpy(
                pool.d_block_target, block_target.data(), 32,
                cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(
                pool.d_share_target, share_target.data(), 32,
                cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        std::memcpy(pool.cached_block_target, block_target.data(), 32);
        std::memcpy(pool.cached_share_target, share_target.data(), 32);
        pool.targets_ready = true;
    }

    const uint32_t n = job.n;
    const uint32_t r = job.r;
    const uint32_t bsz = job.b;
    const uint32_t blocks_per_axis = n / bsz;
    const uint32_t words_per_nonce = blocks_per_axis * blocks_per_axis;
    const size_t matrix_elements = static_cast<size_t>(n) * n;
    const size_t total_matrix_elements = batch * matrix_elements;
    const uint32_t noise_left_elements = n * r;
    const uint32_t noise_right_elements = r * n;
    const uint32_t compress_elements = bsz * bsz;

    const uint32_t batch_u32 = static_cast<uint32_t>(batch);

    gpasha::PrecomputeSeedMidstatesKernel_launch(
        d_seed_a, pool.d_seed_midstates, batch_u32, stream);
    gpasha::PrecomputeSeedMidstatesKernel_launch(
        d_seed_b, pool.d_seed_midstates + batch_u32 * 16, batch_u32, stream);
    gpasha::GenerateMatrixKernel_launch(
        d_seed_a, pool.d_seed_midstates, batch_u32, n, pool.d_base_a, stream);
    gpasha::GenerateMatrixKernel_launch(
        d_seed_b, pool.d_seed_midstates + batch_u32 * 16, batch_u32, n,
        pool.d_base_b, stream);
    gpasha::DeriveNoiseSeedsKernel_launch(
        d_sigma, pool.d_noise_seeds, pool.d_compress_seeds,
        pool.d_noise_midstates, pool.d_compress_midstates, batch_u32, stream);
    gpasha::GenerateAllNoiseKernel_launch(
        pool.d_noise_seeds, pool.d_noise_midstates, batch_u32,
        noise_left_elements, noise_right_elements,
        pool.d_noise_el, pool.d_noise_er, pool.d_noise_fl, pool.d_noise_fr, stream);
    gpasha::GenerateCompressKernel_launch(
        pool.d_compress_seeds, pool.d_compress_midstates, batch_u32,
        compress_elements, pool.d_compress, stream);

    constexpr uint32_t kThreads = 256;
    const uint32_t matrix_blocks = static_cast<uint32_t>(
        (total_matrix_elements + kThreads - 1) / kThreads);
    const bool use_512x16x8 = n == 512 && bsz == 16 && r == 8;
    if (use_512x16x8) {
        batch_perturb_matrix_512x8_kernel<<<matrix_blocks, kThreads, 0, stream>>>(
            pool.d_base_a, pool.d_noise_el, pool.d_noise_er,
            total_matrix_elements, pool.d_matrix_a);
        batch_perturb_matrix_512x8_kernel<<<matrix_blocks, kThreads, 0, stream>>>(
            pool.d_base_b, pool.d_noise_fl, pool.d_noise_fr,
            total_matrix_elements, pool.d_matrix_b);
    } else {
        batch_perturb_matrix_kernel<<<matrix_blocks, kThreads, 0, stream>>>(
            pool.d_base_a, pool.d_noise_el, pool.d_noise_er, n, r,
            matrix_elements, total_matrix_elements, pool.d_matrix_a);
        batch_perturb_matrix_kernel<<<matrix_blocks, kThreads, 0, stream>>>(
            pool.d_base_b, pool.d_noise_fl, pool.d_noise_fr, n, r,
            matrix_elements, total_matrix_elements, pool.d_matrix_b);
    }

    const uint32_t total_words = batch_u32 * words_per_nonce;
    if (use_512x16x8) {
        batch_compressed_words_512x16_kernel<<<total_words, kThreads, 0, stream>>>(
            pool.d_matrix_a, pool.d_matrix_b, pool.d_compress,
            pool.d_words);
    } else {
        batch_compressed_words_kernel<<<total_words, kThreads, 0, stream>>>(
            pool.d_matrix_a, pool.d_matrix_b, pool.d_compress,
            n, bsz, blocks_per_axis, words_per_nonce, matrix_elements,
            pool.d_words);
    }

    gpasha::HashTranscriptCompareKernel_launch(
        pool.d_words, d_sigma, words_per_nonce, n, bsz, batch_u32,
        pool.d_block_target, pool.d_share_target,
        pool.d_digests, pool.d_results, stream);

    if (sync_at_end) {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            err = cudaGetLastError();
        }
        if (err != cudaSuccess) {
            std::fprintf(
                stderr, "batched CUDA transcript failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
    } else if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    if (!digests_out || !found_out) {
        return true;
    }

    std::vector<int32_t> results(batch);
    if (cudaMemcpy(
            results.data(), pool.d_results, batch * sizeof(int32_t),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    digests_out->resize(batch);
    found_out->assign(batch, 0);
    for (size_t i = 0; i < batch; ++i) {
        if (results[i] == 0) {
            continue;
        }
        (*found_out)[i] = 1;
        if (cudaMemcpy(
                (*digests_out)[i].data(), pool.d_digests + i * 32, 32,
                cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
    }
    return true;
}

void FillPackedJobKey(
    const btx::pow::MatMulJob& job,
    const std::vector<uint8_t>& share_target,
    PackedJobKey& key)
{
    std::memset(&key, 0, sizeof(key));
    std::memcpy(key.prev_hash, job.prev_hash.data(), 32);
    std::memcpy(key.merkle_root, job.merkle_root.data(), 32);
    key.time = job.time;
    key.bits = job.bits;
    key.version = job.version;
    key.block_height = job.block_height;
    key.epsilon_bits = job.epsilon_bits;
    key.parent_mtp = job.has_parent_mtp ? job.parent_mtp : 0;
    const std::vector<uint8_t>& use_target =
        share_target.size() == 32 ? share_target : job.target;
    if (use_target.size() == 32) {
        std::memcpy(key.target, use_target.data(), 32);
        const std::vector<uint8_t>& prehash_base =
            job.block_target.size() == 32 ? job.block_target : use_target;
        const auto pre_hash = btx::pow::PreHashTargetShift(prehash_base, job.epsilon_bits);
        if (pre_hash.size() == 32) {
            std::memcpy(key.pre_hash_target, pre_hash.data(), 32);
        }
    }
}

bool LaunchGateForBatch(
    DeviceLaunchPool& pool,
    const CudaJobParams& h_params,
    const btx::pow::MatMulJob& job,
    const std::vector<uint8_t>& share_target,
    uint64_t start_nonce,
    size_t batch)
{
    if (cudaMemsetAsync(pool.d_gate_count, 0, sizeof(uint32_t), pool.gate_stream) !=
        cudaSuccess) {
        return false;
    }

    PackedJobKey job_key{};
    FillPackedJobKey(job, share_target, job_key);
    const bool job_changed =
        !pool.job_midstate_ready ||
        std::memcmp(&pool.job_key, &job_key, sizeof(job_key)) != 0;
    if (job_changed) {
        init_scan_midstates_kernel<<<1, 1, 0, pool.gate_stream>>>(
            h_params, pool.d_seed_midstate, pool.d_header_midstate);
        pool.job_key = job_key;
        pool.job_midstate_ready = true;
    }

    constexpr int kGateThreads = 256;
    const int gate_blocks =
        static_cast<int>((batch + kGateThreads - 1) / kGateThreads);
    sigma_gate_kernel<<<gate_blocks, kGateThreads, 0, pool.gate_stream>>>(
        h_params, start_nonce, batch,
        pool.d_seed_midstate, pool.d_header_midstate, pool.d_gate_count,
        pool.d_passed_indices, pool.d_passed_sigma,
        pool.d_passed_seed_a, pool.d_passed_seed_b);

    if (cudaMemcpyAsync(
            &pool.h_gate_count, pool.d_gate_count, sizeof(uint32_t),
            cudaMemcpyDeviceToHost, pool.gate_stream) != cudaSuccess) {
        return false;
    }
    return cudaEventRecord(pool.gate_done, pool.gate_stream) == cudaSuccess &&
           cudaGetLastError() == cudaSuccess;
}

bool WaitGateCount(DeviceLaunchPool& pool, uint32_t& passed)
{
    if (cudaEventSynchronize(pool.gate_done) != cudaSuccess) {
        return false;
    }
    passed = pool.h_gate_count;
    return cudaGetLastError() == cudaSuccess;
}

bool CollectScatteredHits(
    DeviceLaunchPool& pool,
    int device,
    size_t launch_batch,
    std::vector<btx::uint256>& out_digests,
    std::vector<bool>& out_found)
{
    out_digests.resize(launch_batch);
    out_found.assign(launch_batch, false);

    const uint32_t passed = pool.last_gate_passed;
    if (passed == 0) {
        return true;
    }
    if (device < 0 || device >= 16) {
        return false;
    }

    std::vector<uint32_t> indices(passed);
    if (cudaMemcpy(
            indices.data(), pool.d_passed_indices, passed * sizeof(uint32_t),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    if (pool.use_batched_matmul) {
        BatchedMatMulPool& bp = g_batched_pool[device];
        std::vector<int32_t> results(passed);
        if (cudaMemcpy(
                results.data(), bp.d_results, passed * sizeof(int32_t),
                cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        for (uint32_t i = 0; i < passed; ++i) {
            if (results[i] == 0) {
                continue;
            }
            const uint32_t idx = indices[i];
            if (idx >= launch_batch) {
                continue;
            }
            out_found[idx] = true;
            if (cudaMemcpy(
                    out_digests[idx].data(), bp.d_digests + static_cast<size_t>(i) * 32, 32,
                    cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    std::vector<uint8_t> found(passed);
    std::vector<uint8_t> digests(static_cast<size_t>(passed) * 32);
    if (cudaMemcpy(
            found.data(), pool.d_found, passed,
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(
            digests.data(), pool.d_digests, static_cast<size_t>(passed) * 32,
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    for (uint32_t i = 0; i < passed; ++i) {
        if (!found[i]) {
            continue;
        }
        const uint32_t idx = indices[i];
        if (idx >= launch_batch) {
            continue;
        }
        out_found[idx] = true;
        std::memcpy(out_digests[idx].data(), digests.data() + static_cast<size_t>(i) * 32, 32);
    }
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
    p.parent_mtp = job.parent_mtp;
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
    const bool use_factored = use_v2_seeds ? false : d_use_factored_path(n, bsz);
    const size_t blocks_per_axis = static_cast<size_t>(n / bsz);
    const size_t matrix_elems = use_v2_seeds ? (2 * nn) : (use_factored ? (2 * nn) : 0);
    const size_t factored_rhs_elems =
        use_factored ? (static_cast<size_t>(bsz) * n * blocks_per_axis) : 0;
    const size_t legacy_scratch_elems = use_factored ? 0 : (scratch * 5);
    return matrix_elems + nn + noise_elems + scratch + factored_rhs_elems + legacy_scratch_elems;
}

} // namespace

extern "C" void CudaSetPrefetchGate(
    int device,
    uint64_t next_start_nonce,
    size_t next_batch)
{
    if (device < 0 || device >= 16) {
        return;
    }
    DeviceLaunchPool& pool = g_launch_pool[device];
    pool.pending_next_gate = next_batch > 0;
    pool.pending_start_nonce = next_start_nonce;
    pool.pending_batch = next_batch;
}

extern "C" bool CudaCollectTranscriptBatch(
    int device,
    size_t launch_batch,
    std::vector<btx::uint256>& out_digests,
    std::vector<bool>& out_found)
{
    if (device < 0 || device >= 16) {
        return false;
    }
    DeviceLaunchPool& pool = g_launch_pool[device];
    if (!pool.matmul_in_flight) {
        return true;
    }
    if (cudaEventSynchronize(pool.matmul_done) != cudaSuccess) {
        return false;
    }
    pool.matmul_in_flight = false;
    return CollectScatteredHits(pool, device, launch_batch, out_digests, out_found);
}

extern "C" void CudaFinalizeTranscriptPipeline(int device)
{
    if (device < 0 || device >= 16) {
        return;
    }
    DeviceLaunchPool& pool = g_launch_pool[device];
    if (pool.matmul_in_flight && pool.matmul_done) {
        cudaEventSynchronize(pool.matmul_done);
        pool.matmul_in_flight = false;
    }
    if (pool.gate_prefetched && pool.gate_done) {
        cudaEventSynchronize(pool.gate_done);
        pool.gate_prefetched = false;
    }
    pool.pending_next_gate = false;
}

extern "C" bool LaunchMatMulTranscriptBatch(
    int device,
    const btx::pow::MatMulJob& job,
    uint64_t start_nonce,
    size_t batch_count,
    const std::vector<uint8_t>& target,
    std::vector<btx::uint256>& out_digests,
    std::vector<bool>& out_found)
{
    if (batch_count == 0) {
        return false;
    }
    if (job.block_height >= btx::pow::kMatMulSeedV3Height && !job.has_parent_mtp) {
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

    const size_t batch = batch_count;
    const size_t ws_uint32_per_nonce = WorkspaceUint32Count(job.n, job.r, job.b, use_v2);
    const size_t per_nonce_ws_bytes = ws_uint32_per_nonce * sizeof(uint32_t);
    const std::vector<uint8_t>& share_target =
        target.size() == 32 ? target : job.target;

    out_digests.resize(batch);
    out_found.assign(batch, false);

    const size_t launch_ws_bytes =
        use_v2 ? per_nonce_ws_bytes : (batch * per_nonce_ws_bytes);
    if (!EnsureLaunchPool(device, batch, launch_ws_bytes)) {
        return false;
    }
    DeviceLaunchPool& pool = g_launch_pool[device];

    if (!use_v2) {
        if (cudaMemset(pool.d_found, 0, batch) != cudaSuccess) {
            return false;
        }
        if (pool.nonce_cap < batch) {
            return false;
        }
        std::vector<uint64_t> host_nonces(batch);
        for (size_t i = 0; i < batch; ++i) {
            host_nonces[i] = start_nonce + i;
        }
        if (cudaMemcpy(pool.d_nonces, host_nonces.data(), batch * sizeof(uint64_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        matmul_nonce_kernel<<<static_cast<int>(batch), kBlockThreads>>>(
            h_params, d_A, d_B, pool.d_nonces, pool.d_workspace, pool.d_digests, pool.d_found);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            return false;
        }
        std::vector<uint8_t> h_found(batch, 0);
        std::vector<uint8_t> h_digests(batch * 32);
        if (cudaMemcpy(h_found.data(), pool.d_found, batch, cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        if (cudaMemcpy(h_digests.data(), pool.d_digests, batch * 32, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            return false;
        }
        for (size_t i = 0; i < batch; ++i) {
            out_found[i] = h_found[i] != 0;
            std::memcpy(out_digests[i].data(), h_digests.data() + i * 32, 32);
        }
        return true;
    }

    uint32_t passed = 0;
    if (pool.gate_prefetched) {
        pool.gate_prefetched = false;
        if (!WaitGateCount(pool, passed)) {
            return false;
        }
    } else {
        if (!LaunchGateForBatch(pool, h_params, job, share_target, start_nonce, batch)) {
            return false;
        }
        if (!WaitGateCount(pool, passed)) {
            return false;
        }
    }
    pool.last_gate_passed = passed;

    if (passed > 0) {
        // Batched transcript path: reuse gate seeds/sigma on device, parallelize
        // matrix perturbation + blocked compression across all passed nonces.
        pool.use_batched_matmul = true;
        if (!ProcessPassedNoncesBatched(
                device, job, share_target, passed,
                pool.d_passed_seed_a, pool.d_passed_seed_b, pool.d_passed_sigma,
                pool.matmul_stream, false, nullptr, nullptr)) {
            return false;
        }
        if (cudaEventRecord(pool.matmul_done, pool.matmul_stream) != cudaSuccess) {
            return false;
        }
        pool.matmul_in_flight = true;
    } else {
        pool.use_batched_matmul = false;
        pool.matmul_in_flight = false;
    }

    if (pool.pending_next_gate) {
        if (!LaunchGateForBatch(
                pool, h_params, job, share_target,
                pool.pending_start_nonce, pool.pending_batch)) {
            return false;
        }
        pool.gate_prefetched = true;
        pool.pending_next_gate = false;
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

    std::vector<btx::uint256> digests;
    std::vector<bool> found;
    if (!LaunchMatMulTranscriptBatch(device, job, nonce, 1, target, digests, found)) {
        return false;
    }
    if (!CudaCollectTranscriptBatch(device, 1, digests, found)) {
        return false;
    }
    CudaFinalizeTranscriptPipeline(device);

    btx::uint256 cpu_digest;
    btx::pow::VerifySolution(job, nonce, job.time, cpu_digest);
    const bool gpu_hit = !found.empty() && found[0];
    const bool cpu_hit = btx::pow::DigestMeetsTarget(cpu_digest, target);

    if (digests[0] != cpu_digest || cpu_hit != gpu_hit) {
        return false;
    }
    return true;
}

#endif // __CUDACC__