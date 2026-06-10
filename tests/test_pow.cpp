#include <cassert>
#include <iostream>
#include <string>

#include "pow/matmul_pow.h"
#include "stratum/stratum_protocol.h"

static void test_target_from_hex()
{
    std::vector<uint8_t> t;
    assert(btx::stratum::TargetFromHex("00ff", t));
    assert(t.size() == 32);
    assert(t[0] == 0xff);
    assert(t[1] == 0x00);
}

static void test_subscribe_not_confused_with_notify()
{
    const std::string sub = R"({"id":1,"result":[[["mining.notify","6a2894a7000017f4"]],"000017f4",4],"error":null})";
    assert(!btx::stratum::IsMiningNotifyLine(sub));
    std::string en1;
    int en2 = 0;
    assert(btx::stratum::ParseSubscribeResult(sub, en1, en2));
}

static void test_parse_live_notify()
{
    const std::string line = R"({"id":null,"method":"mining.notify","params":["0000000000002c88",536870912,"e41768fe0c8ed2d40b967c981e3af7cddf6fc495f844563836756fa76a0d2ec9","fe14530b149adfa21a45f7d2666f3c2dbef7296333398ba208ab77ea6b44a6e2",1781098511,"1d14bd00","000052f400000000000000000000000000000000000000000000000000000000",true,{"block_height":126655,"epsilon_bits":18,"matmul_b":16,"matmul_n":512,"matmul_r":8,"nonce64_start":26336739459072,"seed_a":"a6f74b5acf03e8b9955f1e8503045f868223e6a21add4bd1d287df4828f232c6","seed_b":"a257d5579fb72a5bbee627db164336b8aea91095af751ddbae91de568da04597"}]})";

    assert(btx::stratum::IsMiningNotifyLine(line));
    btx::stratum::StratumJob job;
    assert(btx::stratum::ParseNotifyLine(line, job));
    assert(job.seed_b.size() == 64);
    btx::pow::MatMulJob pjob;
    assert(btx::stratum::StratumJobToPowJob(job, pjob));
}

static void test_parse_notify()
{
    const std::string line = R"({"id":null,"method":"mining.notify","params":["job42",536870912,"0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20","aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",1740000000,"1d00ffff","0000000000000000000000000000000000000000000000000000000000ffff",false,{"seed_a":"1111111111111111111111111111111111111111111111111111111111111111","seed_b":"2222222222222222222222222222222222222222222222222222222222222222","block_height":12345,"matmul_n":512,"matmul_b":16,"matmul_r":8,"epsilon_bits":18,"nonce64_start":9876543210}}]})";

    btx::stratum::StratumJob job;
    assert(btx::stratum::ParseNotifyLine(line, job));
    assert(job.job_id == "job42");
    assert(job.version == 536870912);
    assert(job.prev_hash.size() == 64);
    assert(job.merkle_root.size() == 64);
    assert(job.time == 1740000000u);
    assert(job.bits == "1d00ffff");
    assert(job.target.size() == 64);
    assert(job.clean_jobs == false);
    assert(job.seed_a.size() == 64);
    assert(job.block_height == 12345u);
    assert(job.nonce64_start == 9876543210ull);

    btx::pow::MatMulJob pjob;
    assert(btx::stratum::StratumJobToPowJob(job, pjob));
    assert(pjob.n == 512);
    assert(pjob.target.size() == 32);
    assert(pjob.target[0] == 0xff);
    assert(pjob.target[1] == 0x00);
}

static void test_parse_subscribe()
{
    const std::string line = R"({"id":1,"result":[[["mining.notify","6a2894a7000017f4"]],"000017f4",4],"error":null})";
    std::string en1;
    int en2 = 0;
    assert(btx::stratum::ParseSubscribeResult(line, en1, en2));
    assert(en1 == "000017f4");
    assert(en2 == 4);
}

static void test_parse_pool_url()
{
    std::string host;
    uint16_t port = 0;
    assert(btx::stratum::ParsePoolUrl("stratum+tcp://stratum.minebtx.com:3333", host, port));
    assert(host == "stratum.minebtx.com");
    assert(port == 3333);
}

static void test_live_job_no_false_positives()
{
    const std::string line = R"({"id":null,"method":"mining.notify","params":["0000000000002c88",536870912,"e41768fe0c8ed2d40b967c981e3af7cddf6fc495f844563836756fa76a0d2ec9","fe14530b149adfa21a45f7d2666f3c2dbef7296333398ba208ab77ea6b44a6e2",1781098511,"1d14bd00","000052f400000000000000000000000000000000000000000000000000000000",true,{"block_height":126655,"epsilon_bits":18,"matmul_b":16,"matmul_n":512,"matmul_r":8,"nonce64_start":26336739459072,"seed_a":"a6f74b5acf03e8b9955f1e8503045f868223e6a21add4bd1d287df4828f232c6","seed_b":"a257d5579fb72a5bbee627db164336b8aea91095af751ddbae91de568da04597"}]})";

    btx::stratum::StratumJob job;
    assert(btx::stratum::ParseNotifyLine(line, job));
    btx::pow::MatMulJob pjob;
    assert(btx::stratum::StratumJobToPowJob(job, pjob));

    int hits = 0;
    for (int i = 0; i < 256; ++i) {
        uint256 d;
        if (btx::pow::VerifySolution(pjob, pjob.nonce_start + static_cast<uint64_t>(i), pjob.time, d) &&
            btx::pow::DigestMeetsTarget(d, pjob.target)) {
            ++hits;
        }
    }
    assert(hits == 0);
}

static void test_pow_smoke()
{
    btx::pow::MatMulJob job;
    job.n = 16;
    job.b = 8;
    job.r = 2;
    job.bits = 0x1d00ffff;
    job.target.assign(32, 0xff);

    auto sol = btx::pow::SolveCPU(job, 32, 0);
    assert(sol.found);
}

int main()
{
    test_target_from_hex();
    test_subscribe_not_confused_with_notify();
    test_parse_live_notify();
    test_parse_notify();
    test_parse_subscribe();
    test_parse_pool_url();
    test_live_job_no_false_positives();
    test_pow_smoke();
    std::cout << "All pow/stratum tests passed." << std::endl;
    return 0;
}