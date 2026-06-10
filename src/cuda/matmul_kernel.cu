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

    uint32_t m[64];
    for (int i = 0; i < 16; ++i) {
        m[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        m[i] = d_sig1(m[i - 2]) + m[i - 7] + d_sig0(m[i - 15]) + m[i - 16];
    }

    uint32_t a = ctx->state[0];
    uint32_t b = ctx->state[1];
    uint32_t c = ctx->state[2];
    uint32_t d = ctx->state[3];
    uint32_t e = ctx->state[4];
    uint32_t f = ctx->state[5];
    uint32_t g = ctx->state[6];
    uint32_t h = ctx->state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + d_ep1(e) + d_ch(e, f, g) + K[i] + m[i];
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
        if ((candidate & kFieldModulus) == candidate) {
            return candidate & kFieldModulus;
        }
        if (candidate < kFieldModulus) {
            return candidate;
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

__device__ void d_derive_sigma(const CudaJobParams& job, uint64_t nonce64, uint8_t* sigma_internal)
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
    d_sha256_update(&h, job.seed_a, 32);
    d_sha256_update(&h, job.seed_b, 32);

    uint8_t first[32];
    d_sha256_final(&h, first);

    Sha256State h2;
    d_sha256_init(&h2);
    d_sha256_update(&h2, first, 32);
    d_sha256_final(&h2, sigma_internal);
}

__device__ void d_fill_rect(uint32_t* out, uint32_t rows, uint32_t cols, const uint8_t* seed_internal)
{
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

__device__ bool d_below_target(const uint8_t digest[32], const uint8_t target[32])
{
    for (int i = 7; i >= 0; --i) {
        const uint32_t d = static_cast<uint32_t>(digest[i * 4]) |
                           (static_cast<uint32_t>(digest[i * 4 + 1]) << 8) |
                           (static_cast<uint32_t>(digest[i * 4 + 2]) << 16) |
                           (static_cast<uint32_t>(digest[i * 4 + 3]) << 24);
        const uint32_t t = static_cast<uint32_t>(target[i * 4]) |
                           (static_cast<uint32_t>(target[i * 4 + 1]) << 8) |
                           (static_cast<uint32_t>(target[i * 4 + 2]) << 16) |
                           (static_cast<uint32_t>(target[i * 4 + 3]) << 24);
        if (d < t) {
            return true;
        }
        if (d > t) {
            return false;
        }
    }
    return true;
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
    const uint32_t* A,
    const uint32_t* B,
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
    uint8_t* digest_out)
{
    __shared__ uint8_t s_sigma[32];
    __shared__ uint8_t s_seed_el[32];
    __shared__ uint8_t s_seed_er[32];
    __shared__ uint8_t s_seed_fl[32];
    __shared__ uint8_t s_seed_fr[32];
    __shared__ Sha256State s_hasher;
    __shared__ bool s_hit;
    __shared__ bool s_sigma_pass;

    if (threadIdx.x == 0) {
        d_derive_sigma(job, nonce64, s_sigma);
        s_sigma_pass = job.epsilon_bits == 0 ||
                       d_below_target(s_sigma, job.pre_hash_target);
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

    const uint32_t nn = job.n * job.n;
    d_zero_u32(C, nn);
    __syncthreads();

    d_build_compress_vec(s_sigma, job.b, compress_v);
    __syncthreads();

    const uint32_t bsz = job.b;
    const uint32_t N = job.n / bsz;
    const uint32_t total_steps = N * N * N;

    if (threadIdx.x == 0) {
        d_sha256_init(&s_hasher);
    }
    __syncthreads();

    // Flat step loop: all threads share the same (bi,bj,ell) each iteration.
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

        d_mat_get(C, job.n, row0, col0, cblk, bsz);
        __syncthreads();
        d_add_block(cblk, prod, bsz);
        __syncthreads();
        d_write_c_block(C, job.n, row0, col0, cblk, bsz);
        __syncthreads();

        if (threadIdx.x == 0) {
            const uint32_t comp = d_dot(cblk, compress_v, bsz * bsz);
            uint8_t le[4] = {
                static_cast<uint8_t>(comp & 0xff),
                static_cast<uint8_t>((comp >> 8) & 0xff),
                static_cast<uint8_t>((comp >> 16) & 0xff),
                static_cast<uint8_t>((comp >> 24) & 0xff),
            };
            d_sha256_update(&s_hasher, le, 4);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint8_t inner[32];
        d_sha256_final(&s_hasher, inner);

        Sha256State outer;
        d_sha256_init(&outer);
        d_sha256_update(&outer, inner, 32);
        d_sha256_final(&outer, digest_out);
        s_hit = d_below_target(digest_out, job.target);
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

    // Per-nonce workspace layout:
    // [C nn][E_L n*r][E_R r*n][F_L n*r][F_R r*n][compress b*b][ablk][bblk][noise][prod][cblk]
    size_t off = idx * (nn + noise_elems + compress_elems + scratch_elems * 5);

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

    uint8_t digest[32];
    const bool hit = d_solve_nonce(
        params, A, B, nonces[idx], C, E_L, E_R, F_L, F_R, compress_v,
        ablk, bblk, noise_blk, prod, cblk, digest);

    if (threadIdx.x == 0) {
        if (hit) {
            out_found[idx] = 1;
        }
        for (int i = 0; i < 32; ++i) {
            out_digests[idx * 32 + i] = digest[i];
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

size_t WorkspaceUint32Count(uint32_t n, uint32_t r, uint32_t bsz)
{
    const size_t nn = static_cast<size_t>(n) * n;
    const size_t noise_elems = 2 * (static_cast<size_t>(n) * r + static_cast<size_t>(r) * n);
    const size_t scratch = static_cast<size_t>(bsz) * bsz;
    return nn + noise_elems + scratch + scratch * 5;
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

    uint32_t* d_A = nullptr;
    uint32_t* d_B = nullptr;
    if (!EnsureMatricesOnDevice(device, job, &d_A, &d_B)) {
        return false;
    }

    CudaJobParams h_params = MakeCudaJobParams(job, target);

    const size_t batch = nonces.size();
    const size_t ws_uint32_per_nonce = WorkspaceUint32Count(job.n, job.r, job.b);
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
    if (cudaMemcpy(h_digests.data(), pool.d_digests, batch * 32, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    for (size_t i = 0; i < batch; ++i) {
        std::memcpy(out_digests[i].data(), h_digests.data() + i * 32, 32);
        out_found[i] = h_found[i] != 0;
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
    const bool cpu_hit = btx::pow::VerifySolution(job, nonce, job.time, cpu_digest);
    const bool gpu_hit = !found.empty() && found[0];

    if (cpu_hit != gpu_hit) {
        return false;
    }
    if (cpu_hit && digests[0] != cpu_digest) {
        return false;
    }
    return true;
}

#endif // __CUDACC__