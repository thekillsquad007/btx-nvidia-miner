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

// SHA-256 outputs MSB-first bytes; uint256 / pool targets store MSB at data()[31].
static uint256 CanonicalBytesToUint256(const uint8_t raw[32])
{
    uint256 out;
    for (int i = 0; i < 32; ++i) {
        out.data()[i] = raw[31 - i];
    }
    return out;
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
    // Node hashes header uint256 fields via .data() (internal layout), not ToCanonical.
    sha256_update(&h, prev.data(), 32);
    sha256_update(&h, merkle.data(), 32);
    sha256_update(&h, t, 4);
    sha256_update(&h, bi, 4);
    sha256_update(&h, n64, 8);
    sha256_update(&h, dm, 2);
    sha256_update(&h, seed_a.data(), 32);
    sha256_update(&h, seed_b.data(), 32);

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

static uint32_t ReadLE32(const uint8_t* p)
{
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

// Match btxchain/btx UintToArith256 comparison (arith_uint256::operator<=).
static bool IsBelowTarget(const uint8_t digest[32], const std::vector<uint8_t>& target)
{
    if (target.size() != 32) return false;
    for (int i = 7; i >= 0; --i) {
        const uint32_t d = ReadLE32(digest + i * 4);
        const uint32_t t = ReadLE32(target.data() + i * 4);
        if (d < t) return true;
        if (d > t) return false;
    }
    return true;
}

static bool IsZero256(const uint8_t value[32])
{
    for (int i = 0; i < 32; ++i) {
        if (value[i] != 0) return false;
    }
    return true;
}

static void WriteLE32(uint8_t* p, uint32_t v)
{
    p[0] = static_cast<uint8_t>(v & 0xff);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xff);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xff);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xff);
}

static void WriteLE64(uint8_t* p, uint64_t v)
{
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
    }
}

static void WriteLE16(uint8_t* p, uint16_t v)
{
    p[0] = static_cast<uint8_t>(v & 0xff);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xff);
}

static void WriteCompactSize(sha256_state& hasher, uint64_t val)
{
    if (val < 253) {
        const uint8_t c = static_cast<uint8_t>(val);
        sha256_update(&hasher, &c, 1);
    } else if (val < 0x10000) {
        const uint8_t buf[3] = {
            0xFD,
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8),
        };
        sha256_update(&hasher, buf, 3);
    } else if (val < 0x100000000ULL) {
        const uint8_t buf[5] = {
            0xFE,
            static_cast<uint8_t>(val),
            static_cast<uint8_t>(val >> 8),
            static_cast<uint8_t>(val >> 16),
            static_cast<uint8_t>(val >> 24),
        };
        sha256_update(&hasher, buf, 5);
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
        sha256_update(&hasher, buf, 9);
    }
}

static void SetMaxArith256(uint8_t out[32])
{
    for (int i = 0; i < 8; ++i) {
        WriteLE32(out + i * 4, 0xffffffffU);
    }
}

static bool GetBit256(const uint8_t value[32], unsigned bit)
{
    return ((value[bit / 8] >> (bit % 8)) & 1U) != 0;
}

static bool WouldOverflowLeftShift256(const uint8_t value[32], unsigned shift_bits)
{
    if (shift_bits >= 256) {
        return !IsZero256(value);
    }
    for (unsigned bit = 256U - shift_bits; bit < 256U; ++bit) {
        if (GetBit256(value, bit)) {
            return true;
        }
    }
    return false;
}

static void ShiftLeft256(const uint8_t in[32], uint8_t out[32], unsigned shift_bits)
{
    std::memset(out, 0, 32);
    if (shift_bits >= 256) {
        return;
    }
    for (unsigned out_bit = 0; out_bit < 256U; ++out_bit) {
        const int in_bit = static_cast<int>(out_bit) - static_cast<int>(shift_bits);
        if (in_bit >= 0 && GetBit256(in, static_cast<unsigned>(in_bit))) {
            out[out_bit / 8] |= static_cast<uint8_t>(1U << (out_bit % 8));
        }
    }
}

static std::vector<uint8_t> SaturatingShiftTargetLeft(const std::vector<uint8_t>& target,
                                                      uint32_t shift_bits)
{
    if (target.size() != 32 || shift_bits == 0) {
        return target;
    }
    std::vector<uint8_t> out(32);
    if (IsZero256(target.data())) {
        std::memcpy(out.data(), target.data(), 32);
        return out;
    }
    if (shift_bits >= 256 || WouldOverflowLeftShift256(target.data(), shift_bits)) {
        SetMaxArith256(out.data());
        return out;
    }
    ShiftLeft256(target.data(), out.data(), shift_bits);
    return out;
}

} // namespace

// Public API -----------------------------------------------------------------

std::vector<uint8_t> PreHashTargetShift(const std::vector<uint8_t>& target, uint32_t epsilon_bits)
{
    return SaturatingShiftTargetLeft(target, epsilon_bits);
}

uint256 DeterministicMatMulSeedV2(
    const uint256& prev_hash,
    int32_t height,
    int32_t version,
    const uint256& merkle_root,
    uint32_t time,
    uint32_t bits,
    uint64_t nonce64,
    uint16_t matmul_dim,
    uint8_t which)
{
    sha256_state hasher;
    sha256_init(&hasher);

    static const char kTag[] = "BTX_MATMUL_SEED_V2";
    WriteCompactSize(hasher, sizeof(kTag) - 1);
    sha256_update(&hasher, reinterpret_cast<const uint8_t*>(kTag), sizeof(kTag) - 1);
    sha256_update(&hasher, prev_hash.data(), 32);

    uint8_t le[8];
    WriteLE32(le, static_cast<uint32_t>(height));
    sha256_update(&hasher, le, 4);
    WriteLE32(le, static_cast<uint32_t>(version));
    sha256_update(&hasher, le, 4);
    sha256_update(&hasher, merkle_root.data(), 32);
    WriteLE32(le, time);
    sha256_update(&hasher, le, 4);
    WriteLE32(le, bits);
    sha256_update(&hasher, le, 4);
    WriteLE64(le, nonce64);
    sha256_update(&hasher, le, 8);
    WriteLE16(le, matmul_dim);
    sha256_update(&hasher, le, 2);
    sha256_update(&hasher, &which, 1);

    uint8_t digest[32];
    sha256_final(&hasher, digest);
    return From32(digest);
}

bool SigmaBelowPreHashTarget(const uint8_t sigma[32], const std::vector<uint8_t>& target_arith)
{
    if (target_arith.size() != 32) return false;
    for (int i = 0; i < 32; ++i) {
        const uint8_t s = sigma[i];
        const uint8_t t = target_arith[31 - i];
        if (s < t) return true;
        if (s > t) return false;
    }
    return true;
}

bool DigestMeetsTarget(const uint256& digest, const std::vector<uint8_t>& target)
{
    if (target.size() != 32) return false;
    for (int i = 31; i >= 0; --i) {
        const uint8_t d = digest.data()[i];
        const uint8_t t = target[i];
        if (d < t) return true;
        if (d > t) return false;
    }
    return true;
}

bool VerifySolution(const MatMulJob& job, uint64_t nonce, uint32_t ntime, uint256& out_digest)
{
    const uint16_t dim = static_cast<uint16_t>(job.n);
    const uint32_t use_time = ntime ? ntime : job.time;

    uint256 seed_a = job.seed_a;
    uint256 seed_b = job.seed_b;
    if (job.block_height >= kMatMulSeedV2Height) {
        seed_a = DeterministicMatMulSeedV2(
            job.prev_hash, static_cast<int32_t>(job.block_height), job.version,
            job.merkle_root, use_time, job.bits, nonce, dim, 0);
        seed_b = DeterministicMatMulSeedV2(
            job.prev_hash, static_cast<int32_t>(job.block_height), job.version,
            job.merkle_root, use_time, job.bits, nonce, dim, 1);
    }

    auto sigma_bytes = DeriveSigmaBytes(job.version, job.prev_hash, job.merkle_root,
                                        use_time, job.bits, nonce,
                                        dim, seed_a, seed_b);

    if (job.epsilon_bits > 0 && job.block_target.size() == 32) {
        const auto pre_hash_target = PreHashTargetShift(job.block_target, job.epsilon_bits);
        if (!SigmaBelowPreHashTarget(sigma_bytes.data(), pre_hash_target)) {
            return false;
        }
    }

    // Build sigma uint256 for FromSeed / noise (the PRFs expect the byte layout we use in ToCanonical)
    uint256 sigma = MakeUint256(sigma_bytes);

    Matrix A = FromSeed(seed_a, job.n);
    Matrix B = FromSeed(seed_b, job.n);

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

    out_digest = CanonicalBytesToUint256(finald);

    return DigestMeetsTarget(out_digest, job.target);
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
