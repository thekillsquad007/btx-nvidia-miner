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

            // Very lightweight JSON parse for the fields we care about.
            // In production use a small json parser; here we do string scanning for the known protocol.
            if (line.find("\"method\":\"mining.notify\"") != std::string::npos) {
                // Extract params array roughly.
                // For a real robust client we would parse the 9-element array.
                // Here we simulate by looking for known keys in the line (the pool sends them).
                StratumJob j;
                // Minimal extraction for the demo (in real code parse the JSON array properly).
                // We assume the notify format from the dexbtx reference we studied.
                j.job_id = "job-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
                // In a full impl we would parse version, prev, merkle, time, bits, share_target, clean, then the matmul object.
                // For this step we just mark that we have a job and the solver loop will use a constructed job from the last known or defaults.
                {
                    std::lock_guard<std::mutex> lk(job_mutex);
                    current_job = j;
                    has_job = true;
                }
                std::cout << "[stratum] received mining.notify (job " << j.job_id << ")" << std::endl;
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
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            job = current_job;
        }

        // Build a pow::MatMulJob from the stratum job (map the fields).
        pow::MatMulJob pjob;
        pjob.n = job.matmul_n;
        pjob.b = job.matmul_b;
        pjob.r = job.matmul_r;
        // seed_a / seed_b are hex in the notify; in real code hex decode to the uint256.
        // For scaffold we use zero seeds (the real client will fill from the notify).
        // The solver will be fed proper values once the JSON parsing is complete.

        // Dev fee for pool: for ~1% of slices, mine under the dev address so the hashrate
        // contributes to the dev on the PPLNS pool. We keep the same pool, just change
        // the authorized "user" for that slice (the pool credits the worker name).
        std::string effective_user = user;
        if (common::ShouldMineForDev(local_nonce_start / 10000, common::GetDevFeePercent())) {
            effective_user = common::kDevFeeAddress + std::string(".devfee");
        }

        // Use CUDA batch if available, else CPU reference.
        // (The cuda solver already falls back gracefully.)
        auto sols = btx::cuda::SolveBatchCuda(pjob, local_nonce_start, 10000, 64);

        for (auto& s : sols) {
            if (s.found) {
                bool is_block = true; // In real code compare against the block target vs share target.
                on_solution(job, s.nonce, s.ntime, s.digest, is_block);
                // Also submit via stratum
                submit_share(job, s.nonce, s.ntime);
            }
        }

        local_nonce_start += 10000;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
    std::cout << "[stratum] submitted share nonce=" << nonce << std::endl;
}

} // namespace stratum
} // namespace btx
