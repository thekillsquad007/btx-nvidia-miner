#include "stratum/stratum_client.h"

#include "common/dev_fee.h"
#include "cuda/cuda_solver.h"
#include "pow/matmul_pow.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace btx {
namespace stratum {

struct StratumClient::Impl {
    std::string host;
    uint16_t port;
    std::string user;
    std::string pass;
    SolutionCallback on_solution;
    bool use_tls; // placeholder, current impl is plain TCP

    std::atomic<bool> running{false};
    int sock = -1;
    std::thread reader_thread;
    std::thread solver_thread;

    std::mutex job_mutex;
    StratumJob current_job;
    bool has_job = false;

    std::string extranonce1;
    int extranonce2_size = 4;
    uint64_t submit_id = 1;

    // Stats for observability (like in AMD reference)
    int shares_submitted = 0;
    int slices_processed = 0;

    void connect_and_handshake();
    void reader_loop();
    void solver_loop();
    void send_line(const std::string& line);
    void handle_notify(const std::vector<std::string>& params);
    void submit_share(const StratumJob& job, uint64_t nonce, uint32_t ntime);
};

StratumClient::StratumClient(const std::string& host, uint16_t port,
                             const std::string& user, const std::string& pass,
                             SolutionCallback on_solution,
                             bool use_tls)
    : impl(std::make_unique<Impl>())
{
    impl->host = host;
    impl->port = port;
    impl->user = user;
    impl->pass = pass;
    impl->on_solution = std::move(on_solution);
    impl->use_tls = use_tls;
}

StratumClient::~StratumClient() {
    stop();
}

void StratumClient::run_forever() {
    impl->running = true;
    while (impl->running) {
        try {
            impl->connect_and_handshake();
            impl->reader_thread = std::thread(&Impl::reader_loop, impl.get());
            impl->solver_thread = std::thread(&Impl::solver_loop, impl.get());

            if (impl->reader_thread.joinable()) impl->reader_thread.join();
            if (impl->solver_thread.joinable()) impl->solver_thread.join();
        } catch (const std::exception& e) {
            std::cerr << "Stratum session error: " << e.what() << " - reconnecting in 5s" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
}

void StratumClient::stop() {
    impl->running = false;
    if (impl->sock >= 0) {
        close(impl->sock);
        impl->sock = -1;
    }
    if (impl->reader_thread.joinable()) impl->reader_thread.join();
    if (impl->solver_thread.joinable()) impl->solver_thread.join();
}

void StratumClient::Impl::connect_and_handshake() {
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0) {
        throw std::runtime_error("getaddrinfo failed for " + host);
    }

    sock = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock < 0) {
        freeaddrinfo(res);
        throw std::runtime_error("socket failed");
    }

    if (connect(sock, res->ai_addr, res->ai_addrlen) < 0) {
        close(sock);
        sock = -1;
        freeaddrinfo(res);
        throw std::runtime_error("connect failed to " + host);
    }
    freeaddrinfo(res);

    // mining.subscribe (with basic extension for matmul awareness)
    std::string sub = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"btx-nvidia-miner/0.1\",{\"protocol_compliant\":[\"pre_hash_block_tier_v18\"]}]}\n";
    send_line(sub);

    // In real impl we would parse the response for extranonce1 etc.
    // For scaffold we hardcode a minimal flow and wait for notify.

    // mining.authorize
    std::string auth = "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"" + user + "\",\"" + pass + "\"]}\n";
    send_line(auth);

    std::cout << "[stratum] connected and authorized as " << user << std::endl;
}

void StratumClient::Impl::send_line(const std::string& line) {
    if (sock < 0) return;
    ssize_t sent = send(sock, line.data(), line.size(), 0);
    if (sent < 0) {
        // connection problem - will be handled by reconnect
    }
}

void StratumClient::Impl::reader_loop() {
    std::string buffer;
    char tmp[4096];
    while (running) {
        ssize_t n = recv(sock, tmp, sizeof(tmp)-1, 0);
        if (n <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        tmp[n] = 0;
        buffer += tmp;

        size_t pos;
        while ((pos = buffer.find('\n')) != std::string::npos) {
            std::string line = buffer.substr(0, pos);
            buffer.erase(0, pos+1);

            if (!line.empty()) {
                std::cout << "[stratum] recv: " << line.substr(0, 400) << (line.size() > 400 ? "..." : "") << std::endl;
            }

            // Very lightweight JSON parse for the fields we care about.
            // In production use a small json parser; here we do string scanning for the known protocol.
            if (line.find("mining.notify") != std::string::npos) {
                StratumJob j;
                // Robust parsing for the pool's JSON array format: params[0]=job_id, ..., params[8]=matmul dict
                auto get_num = [&](const std::string& key) -> uint64_t {
                    std::string k = "\"" + key + "\":";
                    size_t p = line.find(k);
                    if (p == std::string::npos) return 0;
                    p += k.size();
                    uint64_t v = 0;
                    while (p < line.size() && isdigit(line[p])) {
                        v = v * 10 + (line[p] - '0');
                        ++p;
                    }
                    return v;
                };
                auto get_str = [&](const std::string& key) -> std::string {
                    std::string k = "\"" + key + "\":\"";
                    size_t p = line.find(k);
                    if (p == std::string::npos) return "";
                    p += k.size();
                    size_t e = line.find("\"", p);
                    if (e == std::string::npos) return "";
                    return line.substr(p, e - p);
                };

                // job_id is usually the first string in params array
                size_t p_params = line.find("\"params\":[");
                if (p_params != std::string::npos) {
                    p_params += 10; // after [
                    size_t q1 = line.find("\"", p_params);
                    if (q1 != std::string::npos) {
                        size_t q2 = line.find("\"", q1 + 1);
                        if (q2 != std::string::npos) j.job_id = line.substr(q1 + 1, q2 - q1 - 1);
                    }
                }

                j.version = get_num("version");
                if (j.version == 0) j.version = 536870912; // default
                j.time = get_num("time");
                j.bits = get_str("bits");
                j.target = get_str("target");
                if (j.target.empty()) j.target = get_str("share_target");
                j.clean_jobs = (line.find("\"clean_jobs\":true") != std::string::npos);

                // matmul dict fields (last element in params)
                j.seed_a = get_str("seed_a");
                j.seed_b = get_str("seed_b");
                j.block_height = get_num("block_height");
                j.matmul_n = get_num("matmul_n");
                if (j.matmul_n == 0) j.matmul_n = 512;
                j.matmul_b = get_num("matmul_b");
                if (j.matmul_b == 0) j.matmul_b = 16;
                j.matmul_r = get_num("matmul_r");
                if (j.matmul_r == 0) j.matmul_r = 8;
                j.epsilon_bits = get_num("epsilon_bits");
                if (j.epsilon_bits == 0) j.epsilon_bits = 18;
                j.nonce64_start = get_num("nonce64_start");

                {
                    std::lock_guard<std::mutex> lk(job_mutex);
                    current_job = j;
                    has_job = true;
                }
                std::cout << "[stratum] received mining.notify job=" << j.job_id 
                          << " height=" << j.block_height 
                          << " nonce_start=" << j.nonce64_start 
                          << " seeds=" << (j.seed_a.empty() ? "no" : "yes") 
                          << " target=" << j.target.substr(0,16) << "..." 
                          << std::endl << std::flush;
            }
            // Also handle set_difficulty, set_extranonce, etc. in full version.
        }
    }
}

void StratumClient::Impl::solver_loop() {
    uint64_t local_nonce_start = 0;
    while (running) {
        StratumJob job;
        {
            std::lock_guard<std::mutex> lk(job_mutex);
            if (!has_job) {
                static int wait_count = 0;
                if (++wait_count % 50 == 0) {
                    std::cout << "solver_loop: waiting for job (has_job=false)" << std::endl << std::flush;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            job = current_job;
        }

        std::cout << "solver_loop: has_job true, processing job " << job.job_id 
                  << " height=" << job.block_height 
                  << " nonce64_start=" << job.nonce64_start 
                  << std::endl << std::flush;

        // Build pjob using real fields parsed from the notify (seeds, nonce_start, etc.).
        pow::MatMulJob pjob;
        pjob.n = job.matmul_n ? job.matmul_n : 512;
        pjob.b = job.matmul_b ? job.matmul_b : 16;
        pjob.r = job.matmul_r ? job.matmul_r : 8;
        pjob.version = job.version;
        pjob.time = job.time;
        if (!job.bits.empty()) {
            unsigned int b = 0;
            sscanf(job.bits.c_str(), "%x", &b);
            pjob.bits = b;
        }
        if (!job.seed_a.empty()) {
            uint256_from_hex(pjob.seed_a, job.seed_a);
        }
        if (!job.seed_b.empty()) {
            uint256_from_hex(pjob.seed_b, job.seed_b);
        }

        uint64_t slice_start = job.nonce64_start ? job.nonce64_start : local_nonce_start;

        // Dev fee: for a fraction of slices submit under dev address (pool credits the worker at submit time).
        std::string submit_user = user;
        bool is_dev_fee = common::ShouldMineForDev(local_nonce_start / 8, common::GetDevFeePercent());
        if (is_dev_fee) {
            submit_user = common::kDevFeeAddress + std::string(".devfee");
        }
        std::cout << "solver: dev_fee_active=" << (is_dev_fee ? "yes" : "no") << " submit_as=" << submit_user << std::endl << std::flush;

        std::cout << "solver: starting slice job=" << job.job_id 
                  << " height=" << job.block_height 
                  << " start=" << slice_start 
                  << " (dev_fee_slice=" << (common::ShouldMineForDev(local_nonce_start / 8, common::GetDevFeePercent()) ? "yes" : "no") << ")"
                  << std::endl << std::flush;

        auto sols = btx::cuda::SolveBatchCuda(pjob, slice_start, 8, 8);

        for (auto& s : sols) {
            if (s.found) {
                bool is_block = true;
                on_solution(job, s.nonce, s.ntime, s.digest, is_block);
                std::string saved = user;
                user = submit_user;
                submit_share(job, s.nonce, s.ntime ? s.ntime : job.time);
                user = saved;
            }
        }

        if (sols.empty()) {
            std::cout << "solver: slice complete, no solution found" << std::endl << std::flush;
        } else {
            std::cout << "solver: slice complete, found " << sols.size() << " solution(s)" << std::endl << std::flush;
        }

        slices_processed++;
        if (slices_processed % 20 == 0) {
            std::cout << "solver: STATS slices=" << slices_processed 
                      << " shares_submitted=" << shares_submitted 
                      << " current_nonce=" << local_nonce_start 
                      << std::endl << std::flush;
        }

        local_nonce_start = slice_start + 8;
        // keep job updated for next iteration (important for same-height notify updates)
        job.nonce64_start = local_nonce_start;

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void StratumClient::Impl::submit_share(const StratumJob& job, uint64_t nonce, uint32_t ntime) {
    // mining.submit [worker, job_id, extranonce2, ntime, nonce]
    std::ostringstream ss;
    ss << "{\"id\":" << (submit_id++) << ",\"method\":\"mining.submit\",\"params\":[\""
       << user << "\",\"" << job.job_id << "\",\"00000000\",\""
       << std::hex << ntime << "\",\""
       << std::hex << nonce << "\"]}\n";
    send_line(ss.str());
    shares_submitted++;
    std::cout << "[stratum] submitted share nonce=" << nonce << " for job=" << job.job_id << " ntime=" << ntime 
              << " (total_submitted=" << shares_submitted << ")" << std::endl << std::flush;
}

} // namespace stratum
} // namespace btx
