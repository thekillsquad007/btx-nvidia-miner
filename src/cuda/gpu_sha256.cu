#include "cuda/gpu_sha256.h"

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cstdint>

namespace gpasha {

__device__ __forceinline__ uint32_t d_rotr(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32 - n));
}

__device__ __constant__ uint32_t d_K[64];

namespace {
const uint32_t h_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

bool g_K_initialized = false;

void EnsureKInitialized()
{
    if (g_K_initialized) {
        return;
    }
    cudaMemcpyToSymbol(d_K, h_K, sizeof(h_K));
    g_K_initialized = true;
}
} // namespace

__device__ void sha256_single_block(const uint8_t* padded_block, uint8_t* hash_out)
{
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(padded_block[i * 4]) << 24) | (uint32_t(padded_block[i * 4 + 1]) << 16) |
               (uint32_t(padded_block[i * 4 + 2]) << 8) | uint32_t(padded_block[i * 4 + 3]);
    }
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = d_rotr(w[i - 15], 7) ^ d_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = d_rotr(w[i - 2], 17) ^ d_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + (d_rotr(e, 6) ^ d_rotr(e, 11) ^ d_rotr(e, 25)) +
                      ((e & f) ^ (~e & g)) + d_K[i] + w[i];
        uint32_t t2 = (d_rotr(a, 2) ^ d_rotr(a, 13) ^ d_rotr(a, 22)) +
                      ((a & b) ^ (a & c) ^ (b & c));
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    a += 0x6a09e667;
    b += 0xbb67ae85;
    c += 0x3c6ef372;
    d += 0xa54ff53a;
    e += 0x510e527f;
    f += 0x9b05688c;
    g += 0x1f83d9ab;
    h += 0x5be0cd19;
    hash_out[0] = a >> 24;
    hash_out[1] = a >> 16;
    hash_out[2] = a >> 8;
    hash_out[3] = a;
    hash_out[4] = b >> 24;
    hash_out[5] = b >> 16;
    hash_out[6] = b >> 8;
    hash_out[7] = b;
    hash_out[8] = c >> 24;
    hash_out[9] = c >> 16;
    hash_out[10] = c >> 8;
    hash_out[11] = c;
    hash_out[12] = d >> 24;
    hash_out[13] = d >> 16;
    hash_out[14] = d >> 8;
    hash_out[15] = d;
    hash_out[16] = e >> 24;
    hash_out[17] = e >> 16;
    hash_out[18] = e >> 8;
    hash_out[19] = e;
    hash_out[20] = f >> 24;
    hash_out[21] = f >> 16;
    hash_out[22] = f >> 8;
    hash_out[23] = f;
    hash_out[24] = g >> 24;
    hash_out[25] = g >> 16;
    hash_out[26] = g >> 8;
    hash_out[27] = g;
    hash_out[28] = h >> 24;
    hash_out[29] = h >> 16;
    hash_out[30] = h >> 8;
    hash_out[31] = h;
}

__device__ void d_WriteLE32(uint8_t* ptr, uint32_t val)
{
    ptr[0] = uint8_t(val);
    ptr[1] = uint8_t(val >> 8);
    ptr[2] = uint8_t(val >> 16);
    ptr[3] = uint8_t(val >> 24);
}

__device__ uint32_t d_ReadLE32(const uint8_t* ptr)
{
    return uint32_t(ptr[0]) | (uint32_t(ptr[1]) << 8) |
           (uint32_t(ptr[2]) << 16) | (uint32_t(ptr[3]) << 24);
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

__device__ void d_to_canonical(const uint8_t* internal, uint8_t* canonical)
{
    for (int i = 0; i < 32; ++i) {
        canonical[i] = internal[31 - i];
    }
}

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
            wt = d_sig1(w[(t - 2) & 15U]) + w[(t - 7) & 15U] +
                 d_sig0(w[(t - 15) & 15U]) + w[(t - 16) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + d_K[t] + wt;
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
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + d_K[t] + w[t];
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
            wt = d_sig1(w[(t - 2) & 15U]) + w[(t - 7) & 15U] +
                 d_sig0(w[(t - 15) & 15U]) + w[(t - 16) & 15U];
            w[t & 15U] = wt;
        }
        const uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + d_K[t] + wt;
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
    return d_bswap32(0x6a09e667U + a) & MODULUS;
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
    return d_bswap32(state[0]) & MODULUS;
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
    return d_bswap32(state[0]) % MODULUS;
}

__device__ __forceinline__ Element d_from_oracle_fast(const uint8_t* seed_internal, uint32_t index)
{
    for (uint32_t retry = 0; retry < 256; ++retry) {
        const uint32_t candidate = retry == 0
            ? d_candidate_from_seed_index(seed_internal, index, false, 0U)
            : d_candidate_from_seed_index(seed_internal, index, true, retry);
        if (candidate < MODULUS) {
            return candidate;
        }
    }
    return d_fallback_candidate(seed_internal, index);
}

__device__ __forceinline__ Element d_from_oracle_midstate(
    const uint32_t* midstate,
    const uint8_t* seed_internal,
    uint32_t index)
{
    const uint32_t candidate = d_candidate_from_midstate(midstate, index);
    return candidate < MODULUS ? candidate : d_from_oracle_fast(seed_internal, index);
}

__device__ void d_DeriveNoiseSeed(const uint8_t* sigma, uint32_t tag_idx, uint8_t* seed_out)
{
    const char* tags[5] = {
        "matmul_noise_EL_v1",
        "matmul_noise_ER_v1",
        "matmul_noise_FL_v1",
        "matmul_noise_FR_v1",
        "matmul-compress-v1"
    };
    const uint8_t tag_len = 18;
    const char* tag = tags[tag_idx];
    uint8_t block[64];
    for (int i = 0; i < 64; ++i) {
        block[i] = 0;
    }
    for (int i = 0; i < tag_len; ++i) {
        block[i] = tag[i];
    }
    for (int i = 0; i < 32; ++i) {
        block[tag_len + i] = sigma[31 - i];
    }
    const uint32_t total_len = tag_len + 32;
    d_sha256_pad(block, total_len);
    uint8_t hash[32];
    sha256_single_block(block, hash);
    for (int i = 0; i < 32; ++i) {
        seed_out[i] = hash[31 - i];
    }
}

__global__ void DeriveNoiseSeedsKernel(
    const uint8_t* sigma_batch,
    uint8_t* noise_seeds,
    uint8_t* compress_seeds,
    uint32_t batch_size)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t nonce_idx = idx / 5;
    const uint32_t seed_idx = idx % 5;
    if (nonce_idx >= batch_size) {
        return;
    }
    const uint8_t* sigma = sigma_batch + size_t(nonce_idx) * 32;
    uint8_t seed_out[32];
    d_DeriveNoiseSeed(sigma, seed_idx, seed_out);
    if (seed_idx < 4) {
        uint8_t* out = noise_seeds + (size_t(nonce_idx) * 4 + seed_idx) * 32;
        for (int i = 0; i < 32; ++i) {
            out[i] = seed_out[i];
        }
    } else {
        uint8_t* out = compress_seeds + size_t(nonce_idx) * 32;
        for (int i = 0; i < 32; ++i) {
            out[i] = seed_out[i];
        }
    }
}

__global__ void PrecomputeSeedMidstatesKernel(
    const uint8_t* seeds,
    uint32_t* seed_midstates,
    uint32_t seed_count)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seed_count) {
        return;
    }
    d_compute_seed_midstate(seeds + size_t(idx) * 32, seed_midstates + size_t(idx) * 16);
}

__global__ void GenerateNoiseKernel(
    const uint8_t* noise_seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t num_elements,
    uint32_t seed_index,
    Element* output)
{
    const size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t nonce_idx = static_cast<uint32_t>(gid / num_elements);
    const uint32_t elem_idx = static_cast<uint32_t>(gid % num_elements);
    if (nonce_idx >= batch_size) {
        return;
    }
    const size_t seed_slot = size_t(nonce_idx) * 4 + seed_index;
    const uint8_t* seed = noise_seeds + seed_slot * 32;
    const Element val = seed_midstates
        ? d_from_oracle_midstate(seed_midstates + seed_slot * 16, seed, elem_idx)
        : d_from_oracle_fast(seed, elem_idx);
    output[size_t(nonce_idx) * num_elements + elem_idx] = val;
}

__global__ void GenerateCompressKernel(
    const uint8_t* compress_seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t num_elements,
    Element* output)
{
    const size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t nonce_idx = static_cast<uint32_t>(gid / num_elements);
    const uint32_t elem_idx = static_cast<uint32_t>(gid % num_elements);
    if (nonce_idx >= batch_size) {
        return;
    }
    const uint8_t* seed = compress_seeds + size_t(nonce_idx) * 32;
    const Element val = seed_midstates
        ? d_from_oracle_midstate(seed_midstates + size_t(nonce_idx) * 16, seed, elem_idx)
        : d_from_oracle_fast(seed, elem_idx);
    output[size_t(nonce_idx) * num_elements + elem_idx] = val;
}

__device__ void sha256_blocks_midstate(uint32_t s[8], const uint8_t* data, uint32_t block_count);
__device__ void sha256_finalize_from_midstate(uint32_t s[8], uint8_t hash_out[32]);

__device__ void d_sha256_message(const uint8_t* msg, uint32_t len, uint8_t out[32])
{
    uint32_t s[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const uint32_t full_blocks = len / 64;
    if (full_blocks > 0) {
        sha256_blocks_midstate(s, msg, full_blocks);
    }
    const uint32_t rem = len % 64;
    uint8_t pad[64];
    for (int i = 0; i < 64; ++i) {
        pad[i] = 0;
    }
    for (uint32_t i = 0; i < rem; ++i) {
        pad[i] = msg[full_blocks * 64 + i];
    }
    pad[rem] = 0x80;
    const uint64_t total_bits = uint64_t(len) * 8u;
    if (rem < 56) {
        pad[62] = uint8_t((total_bits >> 8) & 0xFF);
        pad[63] = uint8_t(total_bits & 0xFF);
        sha256_blocks_midstate(s, pad, 1);
    } else {
        sha256_blocks_midstate(s, pad, 1);
        for (int i = 0; i < 64; ++i) {
            pad[i] = 0;
        }
        pad[62] = uint8_t((total_bits >> 8) & 0xFF);
        pad[63] = uint8_t(total_bits & 0xFF);
        sha256_blocks_midstate(s, pad, 1);
    }
    sha256_finalize_from_midstate(s, out);
}

__device__ void d_hash_matrix_words(const Element* words, uint32_t count, uint8_t c_prime_data[32])
{
    uint32_t s[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t block[64];
    uint32_t pos = 0;
    const uint64_t total_bits = uint64_t(count) * 4u * 8u;

    for (uint32_t w = 0; w < count; ++w) {
        d_WriteLE32(block + pos, words[w]);
        pos += 4;
        if (pos == 64) {
            sha256_blocks_midstate(s, block, 1);
            pos = 0;
        }
    }

    uint8_t pad[64];
    for (int i = 0; i < 64; ++i) {
        pad[i] = 0;
    }
    if (pos > 0) {
        for (uint32_t i = 0; i < pos; ++i) {
            pad[i] = block[i];
        }
    }
    pad[pos] = 0x80;
    if (pos < 56) {
        pad[62] = uint8_t((total_bits >> 8) & 0xFF);
        pad[63] = uint8_t(total_bits & 0xFF);
        sha256_blocks_midstate(s, pad, 1);
    } else {
        sha256_blocks_midstate(s, pad, 1);
        for (int i = 0; i < 64; ++i) {
            pad[i] = 0;
        }
        pad[62] = uint8_t((total_bits >> 8) & 0xFF);
        pad[63] = uint8_t(total_bits & 0xFF);
        sha256_blocks_midstate(s, pad, 1);
    }

    uint8_t inner[32];
    sha256_finalize_from_midstate(s, inner);
    d_sha256_message(inner, 32, c_prime_data);
}

__device__ void d_finalize_product_digest(
    const uint8_t* sigma_data,
    const uint8_t* c_prime_data,
    uint32_t n,
    uint32_t b,
    uint8_t digest_data[32])
{
    uint8_t msg[96];
    uint32_t pos = 0;
    const char tag[] = "matmul-product-digest-v3";
    const uint8_t tag_len = sizeof(tag) - 1;
    for (int i = 0; i < tag_len; ++i) {
        msg[pos++] = tag[i];
    }
    for (int i = 0; i < 32; ++i) {
        msg[pos++] = sigma_data[i];
    }
    for (int i = 0; i < 32; ++i) {
        msg[pos++] = c_prime_data[i];
    }
    d_WriteLE32(msg + pos, n);
    pos += 4;
    d_WriteLE32(msg + pos, b);
    pos += 4;

    uint8_t inner[32];
    d_sha256_message(msg, pos, inner);
    d_sha256_message(inner, 32, digest_data);
}

__global__ void HashTranscriptKernel(
    const Element* compressed_words,
    const uint8_t* sigma_batch,
    uint32_t words_per_nonce,
    uint32_t n,
    uint32_t b,
    uint32_t batch_size,
    uint8_t* digest_batch)
{
    const uint32_t nonce_idx = blockIdx.x;
    if (nonce_idx >= batch_size) {
        return;
    }
    const Element* words = compressed_words + size_t(nonce_idx) * words_per_nonce;
    const uint8_t* sigma_data = sigma_batch + size_t(nonce_idx) * 32;

    uint8_t c_prime_data[32];
    d_hash_matrix_words(words, words_per_nonce, c_prime_data);

    uint8_t* out = digest_batch + size_t(nonce_idx) * 32;
    d_finalize_product_digest(sigma_data, c_prime_data, n, b, out);
}

__global__ void CompareDigestsKernel(
    const uint8_t* digest_batch,
    const uint8_t* block_target,
    const uint8_t* share_target,
    uint32_t batch_size,
    int32_t* results)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) {
        return;
    }
    const uint8_t* digest = digest_batch + size_t(idx) * 32;
    bool le_block = true;
    bool le_share = true;
    for (int i = 31; i >= 0; --i) {
        if (digest[i] < block_target[i]) {
            break;
        }
        if (digest[i] > block_target[i]) {
            le_block = false;
            break;
        }
    }
    for (int i = 31; i >= 0; --i) {
        if (digest[i] < share_target[i]) {
            break;
        }
        if (digest[i] > share_target[i]) {
            le_share = false;
            break;
        }
    }
    results[idx] = 0;
    if (le_block) {
        results[idx] = 2;
    } else if (le_share) {
        results[idx] = 1;
    }
}

__device__ void sha256_blocks_midstate(uint32_t s[8], const uint8_t* data, uint32_t block_count)
{
    for (uint32_t blk = 0; blk < block_count; ++blk) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(data[blk * 64 + i * 4]) << 24) |
                   (uint32_t(data[blk * 64 + i * 4 + 1]) << 16) |
                   (uint32_t(data[blk * 64 + i * 4 + 2]) << 8) |
                   uint32_t(data[blk * 64 + i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = d_rotr(w[i - 15], 7) ^ d_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = d_rotr(w[i - 2], 17) ^ d_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = h + (d_rotr(e, 6) ^ d_rotr(e, 11) ^ d_rotr(e, 25)) +
                          ((e & f) ^ (~e & g)) + d_K[i] + w[i];
            uint32_t t2 = (d_rotr(a, 2) ^ d_rotr(a, 13) ^ d_rotr(a, 22)) +
                          ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        s[0] += a;
        s[1] += b;
        s[2] += c;
        s[3] += d;
        s[4] += e;
        s[5] += f;
        s[6] += g;
        s[7] += h;
    }
}

__device__ void sha256_finalize_from_midstate(uint32_t s[8], uint8_t hash_out[32])
{
    for (int i = 0; i < 8; ++i) {
        hash_out[i * 4] = uint8_t(s[i] >> 24);
        hash_out[i * 4 + 1] = uint8_t(s[i] >> 16);
        hash_out[i * 4 + 2] = uint8_t(s[i] >> 8);
        hash_out[i * 4 + 3] = uint8_t(s[i]);
    }
}

__global__ void GenerateMatrixKernel(
    const uint8_t* seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t n,
    Element* output)
{
    const size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t total_elements = batch_size * n * n;
    if (gid >= size_t(total_elements)) {
        return;
    }
    const uint32_t nonce_idx = static_cast<uint32_t>(gid / (n * n));
    const uint32_t elem_idx = static_cast<uint32_t>(gid % (n * n));
    const uint8_t* seed = seeds + size_t(nonce_idx) * 32;
    const Element val = seed_midstates
        ? d_from_oracle_midstate(seed_midstates + size_t(nonce_idx) * 16, seed, elem_idx)
        : d_from_oracle_fast(seed, elem_idx);
    output[size_t(nonce_idx) * n * n + elem_idx] = val;
}

void DeriveNoiseSeedsKernel_launch(
    const uint8_t* sigma_batch,
    uint8_t* noise_seeds,
    uint8_t* compress_seeds,
    uint32_t batch_size,
    cudaStream_t stream)
{
    EnsureKInitialized();
    const uint32_t seed_threads = 256;
    const uint32_t seed_total = batch_size * 5;
    const uint32_t seed_blocks = (seed_total + seed_threads - 1) / seed_threads;
    DeriveNoiseSeedsKernel<<<seed_blocks, seed_threads, 0, stream>>>(
        sigma_batch, noise_seeds, compress_seeds, batch_size);
}

void PrecomputeSeedMidstatesKernel_launch(
    const uint8_t* seeds,
    uint32_t* seed_midstates,
    uint32_t seed_count,
    cudaStream_t stream)
{
    EnsureKInitialized();
    const uint32_t threads = 256;
    const uint32_t blocks = (seed_count + threads - 1) / threads;
    PrecomputeSeedMidstatesKernel<<<blocks, threads, 0, stream>>>(
        seeds, seed_midstates, seed_count);
}

void GenerateNoiseKernel_launch(
    const uint8_t* noise_seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t num_elements,
    uint32_t seed_index,
    Element* output,
    cudaStream_t stream)
{
    EnsureKInitialized();
    const uint32_t total_el = batch_size * num_elements;
    const uint32_t gen_blocks = (total_el + 255) / 256;
    GenerateNoiseKernel<<<gen_blocks, 256, 0, stream>>>(
        noise_seeds, seed_midstates, batch_size, num_elements, seed_index, output);
}

void GenerateCompressKernel_launch(
    const uint8_t* compress_seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t num_elements,
    Element* output,
    cudaStream_t stream)
{
    EnsureKInitialized();
    const uint32_t total_el = batch_size * num_elements;
    const uint32_t gen_blocks = (total_el + 255) / 256;
    GenerateCompressKernel<<<gen_blocks, 256, 0, stream>>>(
        compress_seeds, seed_midstates, batch_size, num_elements, output);
}

void HashTranscriptKernel_launch(
    const Element* compressed_words,
    const uint8_t* sigma_batch,
    uint32_t words_per_nonce,
    uint32_t n,
    uint32_t b,
    uint32_t batch_size,
    uint8_t* digest_batch,
    cudaStream_t stream)
{
    HashTranscriptKernel<<<batch_size, 1, 0, stream>>>(
        compressed_words, sigma_batch, words_per_nonce, n, b, batch_size, digest_batch);
}

void CompareDigestsKernel_launch(
    const uint8_t* digest_batch,
    const uint8_t* block_target,
    const uint8_t* share_target,
    uint32_t batch_size,
    int32_t* results,
    cudaStream_t stream)
{
    const uint32_t cmp_blocks = (batch_size + 255) / 256;
    CompareDigestsKernel<<<cmp_blocks, 256, 0, stream>>>(
        digest_batch, block_target, share_target, batch_size, results);
}

void GenerateMatrixKernel_launch(
    const uint8_t* seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t n,
    Element* output,
    cudaStream_t stream)
{
    EnsureKInitialized();
    const uint32_t total_elements = batch_size * n * n;
    const uint32_t threads = 256;
    const uint32_t blocks = (total_elements + threads - 1) / threads;
    GenerateMatrixKernel<<<blocks, threads, 0, stream>>>(
        seeds, seed_midstates, batch_size, n, output);
}

} // namespace gpasha

#endif // __CUDACC__