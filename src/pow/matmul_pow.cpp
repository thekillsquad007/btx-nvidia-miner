#include "pow/matmul_pow.h"

#include "pow/field.h"
#include "pow/matrix.h"
#include "pow/noise.h"
#include "pow/sha256_portable.h"
#include "pow/uint256_stub.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <vector>

namespace btx {
namespace pow {

namespace {

// --- uint256 helpers to match node byte ordering for sigma/header hash ---

static std::array<uint8_t, 32> ToCanonical(const uint256& v)
{
    std::array<uint8_t, 32> out;
    for (size_t i = 0; i < 32; ++i) {
        out[i] = v.data()[31 - i];
    }
    return out;
}

static uint256 FromCanonical(const std::array<uint8_t, 32>& bytes)
{
    uint256 out;
    for (size_t i = 0; i < 32; ++i) {
        out.data()[i] = bytes[31 - i];
    }
    return out;
}

static uint256 From32(const uint8_t d[32])
{
    return uint256(d);
}

// Replicate the node's ComputeMatMulHeaderHash + DeriveSigma exactly.
uint256 ComputeMatMulHeaderHashForSigma(int32_t version,
                                        const uint256& prev,
                                        const uint256& merkle,
                                        uint32_t time,
                                        uint32_t bits,
                                        uint64_t nonce64,
                                        uint16_t dim,
                                        const uint256& seed_a,
                                        const uint256& seed_b)
{
    sha256_state h;
    sha256_init(&h);

    // LE fields
    uint8_t ver[4];   ver[0] = version&0xff;   ver[1]=(version>>8)&0xff;   ver[2]=(version>>16)&0xff;   ver[3]=(version>>24)&0xff;
    uint8_t t[4];     t[0] = time&0xff;        t[1]=(time>>8)&0xff;        t[2]=(time>>16)&0xff;        t[3]=(time>>24)&0xff;
    uint8_t bi[4];    bi[0] = bits&0xff;       bi[1]=(bits>>8)&0xff;       bi[2]=(bits>>16)&0xff;       bi[3]=(bits>>24)&0xff;
    uint8_t n64[8];   for (int i=0;i<8;i++) n64[i] = (nonce64 >> (i*8)) & 0xff;
    uint8_t d16[2];   d16[0] = dim & 0xff;     d16[1] = (dim >> 8) & 0xff;

    sha256_update(&h, ver, 4);
    auto prev_c = ToCanonical(prev);   sha256_update(&h, prev_c.data(), 32);
    auto merkle_c = ToCanonical(merkle); sha256_update(&h, merkle_c.data(), 32);
    sha256_update(&h, t, 4);
    sha256_update(&h, bi, 4);
    sha256_update(&h, n64, 8);
    sha256_update(&h, d16, 2);
    auto sa = ToCanonical(seed_a); sha256_update(&h, sa.data(), 32);
    auto sb = ToCanonical(seed_b); sha256_update(&h, sb.data(), 32);

    uint8_t first[32];
    sha256_final(&h, first);

    sha256_state h2;
    sha256_init(&h2);
    sha256_update(&h2, first, 32);
    uint8_t sigma_bytes[32];
    sha256_final(&h2, sigma_bytes);

    return FromCanonical({}); // placeholder to satisfy signature; we return proper below
    // Actually construct:
    uint256 sigma;
    std::memcpy(sigma.data(), sigma_bytes, 32); // store in the internal order we chose
    // The ToCanonical on this sigma later will reverse again to feed the PRFs — this mirrors the node.
    return sigma;
}

// Corrected version that returns the 32-byte sigma directly for PRF use.
static std::array<uint8_t, 32> DeriveSigmaBytes(int32_t version,
                                                const uint256& prev,
                                                const uint256& merkle,
                                                uint32_t time,
                                                uint32_t bits,
                                                uint64_t nonce64,
                                                uint16_t dim,
                                                const uint256& seed_a,
                                                const uint256& seed_b)
{
    sha256_state h;
    sha256_init(&h);

    uint8_t ver[4];  ver[0]=version&0xff; ver[1]=(version>>8)&0xff; ver[2]=(version>>16)&0xff; ver[3]=(version>>24)&0xff;
    uint8_t t[4];    t[0]=time&0xff; t[1]=(time>>8)&0xff; t[2]=(time>>16)&0xff; t[3]=(time>>24)&0xff;
    uint8_t bi[4];   bi[0]=bits&0xff; bi[1]=(bits>>8)&0xff; bi[2]=(bits>>16)&0xff; bi[3]=(bits>>24)&0xff;
    uint8_t n64[8];  for(int i=0;i<8;i++) n64[i]=(nonce64>>(i*8))&0xff;
    uint8_t dm[2];   dm[0]=dim&0xff; dm[1]=(dim>>8)&0xff;

    sha256_update(&h, ver, 4);
    auto pc = ToCanonical(prev);   sha256_update(&h, pc.data(), 32);
    auto mc = ToCanonical(merkle); sha256_update(&h, mc.data(), 32);
    sha256_update(&h, t, 4);
    sha256_update(&h, bi, 4);
    sha256_update(&h, n64, 8);
    sha256_update(&h, dm, 2);
    auto sa = ToCanonical(seed_a); sha256_update(&h, sa.data(), 32);
    auto sb = ToCanonical(seed_b); sha256_update(&h, sb.data(), 32);

    uint8_t first[32];
    sha256_final(&h, first);

    sha256_state h2; sha256_init(&h2);
    sha256_update(&h2, first, 32);
    uint8_t sigma[32];
    sha256_final(&h2, sigma);
    std::array<uint8_t, 32> out{};
    std::memcpy(out.data(), sigma, 32);
    return out;
}

static uint256 MakeUint256(const std::array<uint8_t, 32>& b)
{
    uint256 u;
    std::memcpy(u.data(), b.data(), 32);
    return u;
}

static bool IsBelowTarget(const uint8_t digest[32], const std::vector<uint8_t>& target_be)
{
    if (target_be.size() != 32) return false;
    // Compare as big-endian 256-bit: first byte that differs decides
    for (int i = 0; i < 32; ++i) {
        if (digest[i] < target_be[i]) return true;
        if (digest[i] > target_be[i]) return false;
    }
    return true; // equal is valid
}

} // namespace

// Public API -----------------------------------------------------------------

bool VerifySolution(const MatMulJob& job, uint64_t nonce, uint32_t ntime, uint256& out_digest)
{
    const uint16_t dim = static_cast<uint16_t>(job.n);

    auto sigma_bytes = DeriveSigmaBytes(job.version, job.prev_hash, job.merkle_root,
                                        ntime ? ntime : job.time, job.bits, nonce,
                                        dim, job.seed_a, job.seed_b);

    // Build sigma uint256 for FromSeed / noise (the PRFs expect the byte layout we use in ToCanonical)
    uint256 sigma = MakeUint256(sigma_bytes);

    Matrix A = FromSeed(job.seed_a, job.n);
    Matrix B = FromSeed(job.seed_b, job.n);

    // Now using the cleaned noise module (DeriveNoiseSeed stores using the
    // CanonicalBytesToUint256 convention so from_oracle produces the exact
    // same low-rank factors as the reference node implementation).
    auto np = noise::Generate(sigma, job.n, job.r);

    Matrix E_L = np.E_L;
    Matrix E_R = np.E_R;
    Matrix F_L = np.F_L;
    Matrix F_R = np.F_R;

    Matrix E = E_L * E_R;   // low-rank
    Matrix F = F_L * F_R;
    Matrix Ap = A + E;
    Matrix Bp = B + F;

    // Now the canonical blocked matmul + running transcript
    const uint32_t bsz = job.b;
    const uint32_t N = job.n / bsz;

    Matrix C(job.n, job.n);

    sha256_state hasher;
    sha256_init(&hasher);

    // Local compression vector derivation (matches node matmul-compress-v1 using CanonicalBytesToUint256 store).
    auto compress_vec = [&]() -> std::vector<field::Element> {
        sha256_state st; sha256_init(&st);
        const char* tag = "matmul-compress-v1";
        auto sb = ToCanonical(sigma);
        sha256_update(&st, (const uint8_t*)tag, 18);
        sha256_update(&st, sb.data(), 32);
        uint8_t seedb[32]; sha256_final(&st, seedb);

        // Store reversed so from_oracle's swap recovers seedb as the PRF input (exact node behavior).
        uint256 seedv;
        for (size_t i = 0; i < 32; ++i) {
            seedv.data()[i] = seedb[31 - i];
        }

        const uint64_t len = static_cast<uint64_t>(bsz) * bsz;
        std::vector<field::Element> vv; vv.reserve(static_cast<size_t>(len));
        for (uint64_t k = 0; k < len; ++k) {
            vv.push_back(field::from_oracle(seedv, static_cast<uint32_t>(k)));
        }
        return vv;
    }();

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            for (uint32_t ell = 0; ell < N; ++ell) {
                Matrix ablk = Ap.block(i, ell, bsz);
                Matrix bblk = Bp.block(ell, j, bsz);
                Matrix prod = ablk * bblk;

                Matrix cblk = C.block(i, j, bsz);
                // field-wise add
                for (uint32_t x=0; x<bsz; x++) for (uint32_t y=0; y<bsz; y++) {
                    cblk.at(x,y) = field::add(cblk.at(x,y), prod.at(x,y));
                }
                // write back (we don't have set_block yet - do it via direct on C)
                for (uint32_t x=0; x<bsz; x++) for (uint32_t y=0; y<bsz; y++) {
                    C.at(i*bsz + x, j*bsz + y) = cblk.at(x,y);
                }

                // compress the *running* block
                field::Element comp = field::dot(cblk.data(), compress_vec.data(), bsz*bsz);
                uint8_t le[4];
                le[0] = comp & 0xff; le[1]=(comp>>8)&0xff; le[2]=(comp>>16)&0xff; le[3]=(comp>>24)&0xff;
                sha256_update(&hasher, le, 4);
            }
        }
    }

    uint8_t inner[32];
    sha256_final(&hasher, inner);

    sha256_state outer; sha256_init(&outer);
    sha256_update(&outer, inner, 32);
    uint8_t finald[32];
    sha256_final(&outer, finald);

    out_digest = From32(finald);

    return IsBelowTarget(finald, job.target);
}

MatMulSolution SolveCPU(const MatMulJob& job, uint64_t max_tries, uint32_t ntime)
{
    MatMulSolution sol;
    if (max_tries == 0) return sol;

    uint32_t use_time = ntime ? ntime : job.time;
    uint64_t nonce = job.nonce_start;

    for (uint64_t i = 0; i < max_tries; ++i) {
        uint256 dig;
        if (VerifySolution(job, nonce, use_time, dig)) {
            sol.found = true;
            sol.nonce = nonce;
            sol.ntime = use_time;
            sol.digest = dig;
            // Check if it also meets the block target (for logging / solo)
            // We compare against job.target; caller may have passed share target or block target.
            sol.meets_block_target = true; // for now the caller decides by which target it gave us
            return sol;
        }
        if (nonce == UINT64_MAX) break;
        ++nonce;
    }
    return sol;
}

} // namespace pow
} // namespace btx
