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

int main()
{
    auto job = MakeSmallJob();

    btx::uint256 cpu_digest;
    for (uint64_t nonce = 0; nonce < 8; ++nonce) {
        const bool cpu_hit = btx::pow::VerifySolution(job, nonce, job.time, cpu_digest);
        assert(CudaVerifyAgainstCpu(job, nonce, job.target));
        (void)cpu_hit;
    }

    std::cout << "CUDA PoW matches CPU reference for sample nonces." << std::endl;
    return 0;
}