#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "pow/matmul_pow.h"
#include "stratum/stratum_protocol.h"

static std::string DigestHex(const uint256& digest)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 31; i >= 0; --i) {
        ss << std::setw(2) << static_cast<unsigned>(digest.data()[i]);
    }
    return ss.str();
}

static int test_oracle_digests()
{
    const std::string line = R"({"id":null,"method":"mining.notify","params":["0000000000002c88",536870912,"e41768fe0c8ed2d40b967c981e3af7cddf6fc495f844563836756fa76a0d2ec9","fe14530b149adfa21a45f7d2666f3c2dbef7296333398ba208ab77ea6b44a6e2",1781098511,"1d14bd00","000052f400000000000000000000000000000000000000000000000000000000",true,{"block_height":126655,"epsilon_bits":0,"matmul_b":16,"matmul_n":512,"matmul_r":8,"nonce64_start":26336739459072,"seed_a":"a6f74b5acf03e8b9955f1e8503045f868223e6a21add4bd1d287df4828f232c6","seed_b":"a257d5579fb72a5bbee627db164336b8aea91095af751ddbae91de568da04597"}]})";

    btx::stratum::StratumJob job;
    if (!btx::stratum::ParseNotifyLine(line, job)) {
        std::cerr << "failed to parse notify line" << std::endl;
        return 1;
    }
    btx::pow::MatMulJob pjob;
    if (!btx::stratum::StratumJobToPowJob(job, pjob)) {
        std::cerr << "failed to convert stratum job" << std::endl;
        return 1;
    }
    pjob.epsilon_bits = 0;

    struct Case {
        uint64_t nonce;
        const char* digest_hex;
    };
    const Case cases[] = {
        {26336739459072ULL, "c0bdb016d1a95736824a11b868844dd6daefe1abd9e7ffbc8629fabd0800a83f"},
        {26336739459073ULL, "a483712032e8e67ea93920df33a60f908136891a8754ec200cccd2e8052b7667"},
        {26336739459074ULL, "82a5648d474df97cc0d260ccc9b8989c445c86dc8a649772676ed615974374ab"},
        {26336739459075ULL, "0069e86f8697da76fee6b2ad423e4d5e88c9b0f6142f5a77a7a04638b8fef87c"},
        {26336739459076ULL, "2fb406227b17ff78af04644352f4223877d1fe56af358731e51fa0eb7d3652fa"},
    };

    for (const auto& c : cases) {
        uint256 digest;
        const bool ok = btx::pow::VerifySolution(pjob, c.nonce, pjob.time, digest);
        const std::string got = DigestHex(digest);
        if (got != c.digest_hex) {
            std::cerr << "nonce " << c.nonce << " digest mismatch\n"
                      << " expected=" << c.digest_hex << "\n"
                      << "      got=" << got << "\n"
                      << " verify_ok=" << ok << std::endl;
            return 1;
        }
    }
    return 0;
}

int main()
{
    if (test_oracle_digests() != 0) {
        return 1;
    }
    std::cout << "Oracle digest vectors match btx-gbt-solve." << std::endl;
    return 0;
}