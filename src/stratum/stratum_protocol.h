#pragma once

#include "pow/matmul_pow.h"
#include "stratum/stratum_client.h"

#include <cstdint>
#include <string>
#include <vector>

namespace btx {
namespace stratum {

// Parse a 64-char (or shorter, left-padded) big-endian hex target into 32 bytes.
bool TargetFromHex(const std::string& hex, std::vector<uint8_t>& out);
bool BlockTargetFromBits(const std::string& bits_hex, std::vector<uint8_t>& out);

// True only for pool push messages, not subscribe responses that mention "mining.notify".
bool IsMiningNotifyLine(const std::string& line);

// Parse mining.notify JSON line per RFC-0001 / dexbtx reference:
// [job_id, version, prevhash, merkleroot, time, bits, share_target, clean_jobs, matmul_meta]
bool ParseNotifyLine(const std::string& line, StratumJob& out);

// Convert a parsed stratum job into the PoW job the solver consumes.
bool StratumJobToPowJob(const StratumJob& job, pow::MatMulJob& out);

// Parse stratum+tcp://host:port or host:port.
bool ParsePoolUrl(const std::string& url, std::string& host, uint16_t& port);

// Parse mining.subscribe result: extranonce1 and extranonce2_size.
bool ParseSubscribeResult(const std::string& line, std::string& extranonce1, int& extranonce2_size);

// Parse mining.submit / other RPC responses with an id field.
bool ParseRpcResult(const std::string& line, uint64_t id, bool& ok, std::string& error);

struct CanonicalNameAssignment {
    std::string gpu_uuid;
    std::string canonical_name;
    std::string operator_label;
};

// Parse pool push: mining.set_canonical_name
bool ParseSetCanonicalNameLine(const std::string& line, std::vector<CanonicalNameAssignment>& out);

} // namespace stratum
} // namespace btx