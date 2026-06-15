#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "pow/matmul_pow.h"
#include "pow/uint256_stub.h"

extern "C" bool CudaVerifyAgainstCpu(
    const btx::pow::MatMulJob& job,
    uint64_t nonce,
    const std::vector<uint8_t>& target);

static btx::pow::MatMulJob MakeSmallJob()
{
    btx::pow::MatMulJob job;
    job.n = 16;
    job.b = 8;
    job.r = 2;
    job.version = 1;
    job.time = 1740000000u;
    job.bits = 0x1d00ffff;
    job.target.assign(32, 0xff);
    uint256_from_hex(job.seed_a, "1111111111111111111111111111111111111111111111111111111111111111");
    uint256_from_hex(job.seed_b, "2222222222222222222222222222222222222222222222222222222222222222");
    uint256_from_hex(job.prev_hash, "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    uint256_from_hex(job.merkle_root, "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899");
    return job;
}

static btx::pow::MatMulJob MakeV2Job()
{
    auto job = MakeSmallJob();
    job.n = 512;
    job.b = 16;
    job.r = 8;
    job.block_height = btx::pow::kMatMulSeedV2Height;
    job.version = 536870912;
    job.time = 1775000000u;
    job.bits = 0x1d0b8746;
    job.target.assign(32, 0xff);
    uint256_from_hex(job.prev_hash, "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20");
    uint256_from_hex(job.merkle_root, "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899");
    return job;
}

static btx::pow::MatMulJob MakeV3Job()
{
    auto job = MakeV2Job();
    job.block_height = btx::pow::kMatMulSeedV3Height;
    job.time = 1781098511u;
    job.bits = 0x1d14bd00U;
    job.parent_mtp = 1780000000LL;
    job.has_parent_mtp = true;
    uint256_from_hex(job.prev_hash,
        "e41768fe0c8ed2d40b967c981e3af7cddf6fc495f844563836756fa76a0d2ec9");
    uint256_from_hex(job.merkle_root,
        "fe14530b149adfa21a45f7d2666f3c2dbef7296333398ba208ab77ea6b44a6e2");
    return job;
}

static void VerifyNonceRange(const btx::pow::MatMulJob& job, uint64_t start, uint64_t count)
{
    for (uint64_t nonce = start; nonce < start + count; ++nonce) {
        assert(CudaVerifyAgainstCpu(job, nonce, job.target));
    }
}

int main()
{
    auto v2 = MakeV2Job();
    VerifyNonceRange(v2, 1000000, 64);

    auto v3 = MakeV3Job();
    VerifyNonceRange(v3, 2000000, 64);

    std::cout << "CUDA PoW matches CPU reference (legacy + v2 + v3 sample nonces)." << std::endl;
    return 0;
}