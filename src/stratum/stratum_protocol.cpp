#include "stratum/stratum_protocol.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

namespace btx {
namespace stratum {

namespace {

struct JsonParser {
    const char* p = nullptr;
    const char* end = nullptr;

    explicit JsonParser(const std::string& s)
        : p(s.c_str()), end(s.c_str() + s.size()) {}

    void skip_ws()
    {
        while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;
    }

    bool consume(char c)
    {
        skip_ws();
        if (p < end && *p == c) { ++p; return true; }
        return false;
    }

    bool parse_string(std::string& out)
    {
        skip_ws();
        if (p >= end || *p != '"') return false;
        ++p;
        out.clear();
        while (p < end) {
            char c = *p++;
            if (c == '"') return true;
            if (c == '\\' && p < end) {
                char esc = *p++;
                if (esc == 'n') out.push_back('\n');
                else if (esc == 't') out.push_back('\t');
                else if (esc == 'r') out.push_back('\r');
                else out.push_back(esc);
                continue;
            }
            out.push_back(c);
        }
        return false;
    }

    bool parse_number(uint64_t& out)
    {
        skip_ws();
        if (p >= end || (!std::isdigit(static_cast<unsigned char>(*p)) && *p != '-')) return false;
        char* nend = nullptr;
        out = std::strtoull(p, &nend, 10);
        if (nend == p) return false;
        p = nend;
        return true;
    }

    bool parse_bool(bool& out)
    {
        skip_ws();
        if (p + 4 <= end && std::strncmp(p, "true", 4) == 0) {
            p += 4; out = true; return true;
        }
        if (p + 5 <= end && std::strncmp(p, "false", 5) == 0) {
            p += 5; out = false; return true;
        }
        return false;
    }

    bool skip_value()
    {
        skip_ws();
        if (p >= end) return false;
        if (*p == '"') { std::string tmp; return parse_string(tmp); }
        if (*p == '{') return skip_object();
        if (*p == '[') return skip_array();
        if (*p == 't' || *p == 'f') { bool b; return parse_bool(b); }
        if (*p == 'n') {
            if (p + 4 <= end && std::strncmp(p, "null", 4) == 0) { p += 4; return true; }
            return false;
        }
        uint64_t n; return parse_number(n);
    }

    bool skip_object()
    {
        if (!consume('{')) return false;
        skip_ws();
        if (consume('}')) return true;
        while (true) {
            std::string key;
            if (!parse_string(key)) return false;
            if (!consume(':')) return false;
            if (!skip_value()) return false;
            skip_ws();
            if (consume('}')) return true;
            if (!consume(',')) return false;
        }
    }

    bool skip_array()
    {
        if (!consume('[')) return false;
        skip_ws();
        if (consume(']')) return true;
        while (true) {
            if (!skip_value()) return false;
            skip_ws();
            if (consume(']')) return true;
            if (!consume(',')) return false;
        }
    }

    bool parse_array(std::vector<std::string>& elements_as_json)
    {
        if (!consume('[')) return false;
        skip_ws();
        if (consume(']')) return true;
        while (true) {
            const char* start = p;
            if (!skip_value()) return false;
            elements_as_json.emplace_back(start, p);
            skip_ws();
            if (consume(']')) return true;
            if (!consume(',')) return false;
        }
    }

    bool parse_object_field(const std::string& key, std::string& value_out)
    {
        if (!consume('{')) return false;
        skip_ws();
        if (consume('}')) return true;
        while (true) {
            std::string k;
            if (!parse_string(k)) return false;
            if (!consume(':')) return false;
            const char* start = p;
            if (!skip_value()) return false;
            if (k == key) value_out.assign(start, p);
            skip_ws();
            if (consume('}')) return true;
            if (!consume(',')) return false;
        }
    }

    bool parse_string_field(const std::string& fragment, const std::string& key, std::string& out)
    {
        JsonParser sub(fragment);
        return sub.parse_object_field(key, out);
    }

    bool parse_uint_field(const std::string& fragment, const std::string& key, uint64_t& out)
    {
        std::string raw;
        if (!parse_string_field(fragment, key, raw)) return false;
        JsonParser sub(raw);
        return sub.parse_number(out);
    }

    bool parse_element_string(const std::string& fragment, std::string& out)
    {
        JsonParser sub(fragment);
        return sub.parse_string(out);
    }

    bool parse_element_uint(const std::string& fragment, uint64_t& out)
    {
        JsonParser sub(fragment);
        return sub.parse_number(out);
    }

    bool parse_element_bool(const std::string& fragment, bool& out)
    {
        JsonParser sub(fragment);
        return sub.parse_bool(out);
    }
};

bool ExtractMethodAndParams(const std::string& line, std::string& method, std::vector<std::string>& params)
{
    JsonParser parser(line);
    if (!parser.consume('{')) return false;
    method.clear();
    params.clear();

    while (true) {
        std::string key;
        if (!parser.parse_string(key)) return false;
        if (!parser.consume(':')) return false;

        if (key == "method") {
            if (!parser.parse_string(method)) return false;
        } else if (key == "params") {
            if (!parser.parse_array(params)) return false;
        } else {
            if (!parser.skip_value()) return false;
        }

        parser.skip_ws();
        if (parser.consume('}')) return !method.empty();
        if (!parser.consume(',')) return false;
    }
}

bool ParseMatmulMeta(const std::string& fragment, StratumJob& job)
{
    auto get_string = [&](const std::string& key, std::string& out) -> bool {
        std::string raw;
        JsonParser parser(fragment);
        if (!parser.parse_object_field(key, raw)) return false;
        JsonParser sub(raw);
        return sub.parse_string(out);
    };
    auto get_uint = [&](const std::string& key, uint64_t& out) -> bool {
        std::string raw;
        JsonParser parser(fragment);
        if (!parser.parse_object_field(key, raw)) return false;
        JsonParser sub(raw);
        return sub.parse_number(out);
    };

    uint64_t n = 0;
    get_string("seed_a", job.seed_a);
    get_string("seed_b", job.seed_b);
    if (get_uint("block_height", n)) job.block_height = static_cast<uint32_t>(n);
    if (get_uint("matmul_n", n)) job.matmul_n = static_cast<uint32_t>(n);
    if (get_uint("matmul_b", n)) job.matmul_b = static_cast<uint32_t>(n);
    if (get_uint("matmul_r", n)) job.matmul_r = static_cast<uint32_t>(n);
    if (get_uint("epsilon_bits", n)) job.epsilon_bits = static_cast<int>(n);
    if (get_uint("nonce64_start", n)) job.nonce64_start = n;
    return true;
}

} // namespace

bool BlockTargetFromBits(const std::string& bits_hex, std::vector<uint8_t>& out)
{
    if (bits_hex.empty()) return false;
    unsigned int compact = 0;
    if (std::sscanf(bits_hex.c_str(), "%x", &compact) != 1) return false;

    const int n_size = static_cast<int>(compact >> 24);
    uint32_t n_word = compact & 0x007fffffU;
    if (n_word == 0) return false;

    uint8_t raw[32]{};
    if (n_size <= 3) {
        n_word >>= 8 * (3 - n_size);
        raw[0] = static_cast<uint8_t>(n_word & 0xff);
        raw[1] = static_cast<uint8_t>((n_word >> 8) & 0xff);
        raw[2] = static_cast<uint8_t>((n_word >> 16) & 0xff);
    } else if (n_size > 34) {
        return false;
    } else {
        const int shift_bytes = n_size - 3;
        if (shift_bytes < 0 || shift_bytes > 29) return false;
        raw[shift_bytes + 0] = static_cast<uint8_t>(n_word & 0xff);
        raw[shift_bytes + 1] = static_cast<uint8_t>((n_word >> 8) & 0xff);
        raw[shift_bytes + 2] = static_cast<uint8_t>((n_word >> 16) & 0xff);
    }

    uint256 target(raw);
    out.assign(target.data(), target.data() + 32);
    return true;
}

bool TargetFromHex(const std::string& hex, std::vector<uint8_t>& out)
{
    std::string h = hex;
    if (h.size() > 2 && (h.rfind("0x", 0) == 0 || h.rfind("0X", 0) == 0)) {
        h = h.substr(2);
    }
    h.erase(std::remove_if(h.begin(), h.end(), [](unsigned char c) {
        return std::isspace(c);
    }), h.end());

    if (h.empty()) return false;
    if (h.size() < 64) h = std::string(64 - h.size(), '0') + h;
    if (h.size() > 64) h = h.substr(h.size() - 64);

    uint256 target;
    uint256_from_hex(target, h);
    out.assign(target.data(), target.data() + 32);
    return true;
}

bool IsMiningNotifyLine(const std::string& line)
{
    return line.find("\"method\":\"mining.notify\"") != std::string::npos ||
           line.find("\"method\": \"mining.notify\"") != std::string::npos;
}

bool ParseNotifyLine(const std::string& line, StratumJob& out)
{
    std::string method;
    std::vector<std::string> params;
    if (!ExtractMethodAndParams(line, method, params)) return false;
    if (method != "mining.notify") return false;
    if (line.find("\"id\":null") == std::string::npos &&
        line.find("\"id\": null") == std::string::npos) {
        // Real pool pushes use "id":null; RPC-shaped lines are not work jobs.
        return false;
    }
    if (params.size() < 7) return false;

    JsonParser elem(params[0]);
    if (!elem.parse_string(out.job_id)) return false;

    uint64_t version = 0;
    JsonParser v(params[1]);
    if (!v.parse_number(version)) return false;
    out.version = static_cast<int>(version);

    JsonParser prev(params[2]);
    if (!prev.parse_string(out.prev_hash)) return false;

    JsonParser merkle(params[3]);
    if (!merkle.parse_string(out.merkle_root)) return false;

    uint64_t time_val = 0;
    JsonParser t(params[4]);
    if (!t.parse_number(time_val)) return false;
    out.time = static_cast<uint32_t>(time_val);

    JsonParser bits(params[5]);
    if (!bits.parse_string(out.bits)) return false;

    JsonParser target(params[6]);
    if (!target.parse_string(out.target)) return false;
    out.share_target = out.target;

    if (params.size() > 7) {
        JsonParser clean(params[7]);
        clean.parse_bool(out.clean_jobs);
    }

    if (params.size() > 8) {
        ParseMatmulMeta(params[8], out);
    }

    if (out.matmul_n == 0) out.matmul_n = 512;
    if (out.matmul_b == 0) out.matmul_b = 16;
    if (out.matmul_r == 0) out.matmul_r = 8;
    if (out.epsilon_bits == 0) out.epsilon_bits = 18;

    return !out.job_id.empty();
}

bool StratumJobToPowJob(const StratumJob& job, pow::MatMulJob& out)
{
    if (job.seed_a.empty() || job.seed_b.empty()) return false;
    if (job.target.empty()) return false;
    if (job.matmul_n < 512 || job.matmul_b == 0 || job.matmul_r == 0) return false;
    if (job.matmul_n % job.matmul_b != 0) return false;
    if (job.matmul_r > job.matmul_n) return false;

    out.n = job.matmul_n;
    out.b = job.matmul_b;
    out.r = job.matmul_r;
    out.version = job.version;
    out.time = job.time;
    out.nonce_start = job.nonce64_start;
    out.block_height = job.block_height;
    out.epsilon_bits = job.epsilon_bits > 0 ? static_cast<uint32_t>(job.epsilon_bits) : 0;

    if (!job.bits.empty()) {
        unsigned int b = 0;
        std::sscanf(job.bits.c_str(), "%x", &b);
        out.bits = b;
    }

    uint256_from_hex(out.seed_a, job.seed_a);
    uint256_from_hex(out.seed_b, job.seed_b);
    uint256_from_hex(out.prev_hash, job.prev_hash);
    uint256_from_hex(out.merkle_root, job.merkle_root);

    if (!TargetFromHex(job.target, out.target)) return false;
    return out.target.size() == 32;
}

bool ParsePoolUrl(const std::string& url, std::string& host, uint16_t& port)
{
    std::string u = url;
    const std::string prefix = "stratum+tcp://";
    if (u.rfind(prefix, 0) == 0) u = u.substr(prefix.size());
    else if (u.rfind("stratum://", 0) == 0) u = u.substr(10);

    auto colon = u.rfind(':');
    if (colon == std::string::npos || colon == 0) return false;

    host = u.substr(0, colon);
    int p = std::atoi(u.substr(colon + 1).c_str());
    if (p <= 0 || p > 65535) return false;
    port = static_cast<uint16_t>(p);
    return !host.empty();
}

bool ParseSubscribeResult(const std::string& line, std::string& extranonce1, int& extranonce2_size)
{
    if (line.find("\"id\":1") == std::string::npos &&
        line.find("\"id\": 1") == std::string::npos) {
        return false;
    }
    if (line.find("\"error\":null") == std::string::npos &&
        line.find("\"error\": null") == std::string::npos) {
        return false;
    }

    // result is [[["mining.notify", sid], ...], extranonce1, extranonce2_size]
    size_t result_pos = line.find("\"result\":");
    if (result_pos == std::string::npos) return false;

    std::vector<std::string> params;
    JsonParser parser(line.substr(result_pos + 9));
    parser.skip_ws();
    if (parser.parse_array(params) && params.size() >= 3) {
        JsonParser en1(params[1]);
        uint64_t en2 = 4;
        JsonParser en2p(params[2]);
        if (en1.parse_string(extranonce1) && en2p.parse_number(en2)) {
            extranonce2_size = static_cast<int>(en2);
            return true;
        }
    }

    // Fallback for nested subscription details arrays from live pools.
    size_t anchor = line.find("]],\"", result_pos);
    if (anchor == std::string::npos) anchor = line.find("],\"", result_pos);
    if (anchor == std::string::npos) return false;

    size_t start = line.find('"', anchor + 2);
    if (start == std::string::npos) return false;
    ++start;
    size_t end = line.find('"', start);
    if (end == std::string::npos) return false;
    extranonce1 = line.substr(start, end - start);

    size_t comma = line.find(',', end);
    if (comma == std::string::npos) return false;
    extranonce2_size = std::atoi(line.c_str() + comma + 1);
    if (extranonce2_size <= 0) extranonce2_size = 4;
    return !extranonce1.empty();
}

bool ParseRpcResult(const std::string& line, uint64_t id, bool& ok, std::string& error)
{
    char id_pat[64];
    std::snprintf(id_pat, sizeof(id_pat), "\"id\":%llu", static_cast<unsigned long long>(id));
    std::string id_pat_spaced = std::string("\"id\": ") + std::to_string(id);
    if (line.find(id_pat) == std::string::npos &&
        line.find(id_pat_spaced) == std::string::npos) {
        return false;
    }

    ok = false;
    error.clear();

    const bool err_null = (line.find("\"error\":null") != std::string::npos ||
                           line.find("\"error\": null") != std::string::npos);
    const bool result_true = (line.find("\"result\":true") != std::string::npos ||
                              line.find("\"result\": true") != std::string::npos ||
                              line.find("\"result\":True") != std::string::npos);
    const bool result_false = (line.find("\"result\":false") != std::string::npos ||
                               line.find("\"result\": false") != std::string::npos ||
                               line.find("\"result\":False") != std::string::npos);

    if (err_null && result_true) {
        ok = true;
        return true;
    }
    if (result_false) {
        ok = false;
        return true;
    }
    if (err_null && result_false) {
        ok = false;
        return true;
    }

    size_t err_pos = line.find("\"error\":");
    if (err_pos != std::string::npos && line.find("\"error\":null", err_pos) == std::string::npos) {
        error = line.substr(err_pos, std::min<size_t>(120, line.size() - err_pos));
    }
    return true;
}

} // namespace stratum
} // namespace btx