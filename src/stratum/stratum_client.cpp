#include "stratum/stratum_client.h"

#include "common/dev_fee.h"
#include "cuda/cuda_solver.h"
#include "stratum/stratum_protocol.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
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
    bool use_tls;
    StratumConfig config;

    std::atomic<bool> running{false};
    int sock = -1;
    std::thread reader_thread;
    std::thread solver_thread;

    std::mutex job_mutex;
    StratumJob current_job;
    bool has_job = false;

    std::string extranonce1;
    int extranonce2_size = 4;
    uint64_t submit_id = 3;

    int shares_submitted = 0;
    int shares_accepted = 0;
    int shares_rejected = 0;
    int slices_processed = 0;
    uint64_t local_nonce_cursor = 0;
    std::atomic<bool> slice_in_progress{false};

    void connect_and_handshake();
    void reader_loop();
    void solver_loop();
    bool recv_line(std::string& line, int timeout_ms);
    void send_line(const std::string& line);
    void handle_notify(const StratumJob& incoming);
    void submit_share(const StratumJob& job, uint64_t nonce, uint32_t ntime);
    void log(const std::string& msg) const;
};

StratumClient::StratumClient(const std::string& host, uint16_t port,
                             const std::string& user, const std::string& pass,
                             SolutionCallback on_solution,
                             bool use_tls,
                             StratumConfig config)
    : impl(std::make_unique<Impl>())
{
    impl->host = host;
    impl->port = port;
    impl->user = user;
    impl->pass = pass;
    impl->on_solution = std::move(on_solution);
    impl->use_tls = use_tls;
    impl->config = config;
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
            std::cerr << "[stratum] session error: " << e.what() << " — reconnecting in 5s" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
}

void StratumClient::stop() {
    impl->running = false;
    if (impl->sock >= 0) {
        shutdown(impl->sock, SHUT_RDWR);
        close(impl->sock);
        impl->sock = -1;
    }
    if (impl->reader_thread.joinable()) impl->reader_thread.join();
    if (impl->solver_thread.joinable()) impl->solver_thread.join();
}

void StratumClient::Impl::log(const std::string& msg) const
{
    if (config.verbose) {
        std::cout << msg << std::endl;
    }
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

    struct timeval tv{};
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::string sub = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"btx-nvidia-miner/0.2\",{\"protocol_compliant\":[\"pre_hash_block_tier_v18\"]}]}\n";
    send_line(sub);

    std::string line;
    bool got_sub = false;
    for (int attempt = 0; attempt < 60 && !got_sub; ++attempt) {
        if (!recv_line(line, 5000)) continue;

        if (line.find("mining.notify") != std::string::npos) {
            StratumJob j;
            if (ParseNotifyLine(line, j)) {
                handle_notify(j);
            }
            continue;
        }

        if (ParseSubscribeResult(line, extranonce1, extranonce2_size)) {
            got_sub = true;
            log("[stratum] subscribed extranonce1=" + extranonce1 + " en2_size=" + std::to_string(extranonce2_size));
        }
    }
    if (!got_sub) {
        throw std::runtime_error("timed out waiting for mining.subscribe response");
    }

    std::string auth = "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"" + user + "\",\"" + pass + "\"]}\n";
    send_line(auth);

    bool authorized = false;
    for (int attempt = 0; attempt < 60 && !authorized; ++attempt) {
        if (!recv_line(line, 5000)) continue;

        if (line.find("mining.notify") != std::string::npos) {
            StratumJob j;
            if (ParseNotifyLine(line, j)) {
                handle_notify(j);
            }
            continue;
        }

        bool ok = false;
        std::string err;
        if (ParseRpcResult(line, 2, ok, err)) {
            if (!ok) throw std::runtime_error("mining.authorize rejected: " + err);
            authorized = true;
        }
    }
    if (!authorized) {
        throw std::runtime_error("timed out waiting for mining.authorize response");
    }

    std::cout << "[stratum] connected to " << host << ":" << port << " as " << user << std::endl;
}

void StratumClient::Impl::send_line(const std::string& line) {
    if (sock < 0) return;
    ssize_t sent = send(sock, line.data(), line.size(), 0);
    if (sent < 0) {
        throw std::runtime_error("send failed");
    }
}

bool StratumClient::Impl::recv_line(std::string& line, int timeout_ms)
{
    line.clear();
    char ch = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (running) {
        if (std::chrono::steady_clock::now() > deadline) return false;

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(sock, &fds);
        timeval tv{};
        auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        tv.tv_sec = remain.count() / 1000;
        tv.tv_usec = (remain.count() % 1000) * 1000;

        int sel = select(sock + 1, &fds, nullptr, nullptr, &tv);
        if (sel < 0) return false;
        if (sel == 0) return false;

        ssize_t n = recv(sock, &ch, 1, 0);
        if (n <= 0) return false;
        if (ch == '\n') return !line.empty();
        if (ch != '\r') line.push_back(ch);
    }
    return false;
}

void StratumClient::Impl::reader_loop() {
    std::string line;
    while (running) {
        if (!recv_line(line, 30000)) {
            if (!running) break;
            throw std::runtime_error("pool closed connection or read timeout");
        }

        log("[stratum] recv: " + line.substr(0, 300) + (line.size() > 300 ? "..." : ""));

        if (line.find("mining.notify") != std::string::npos) {
            StratumJob j;
            if (ParseNotifyLine(line, j)) {
                handle_notify(j);
            } else {
                std::cerr << "[stratum] failed to parse mining.notify" << std::endl;
            }
            continue;
        }

        bool ok = false;
        std::string err;
        if (ParseRpcResult(line, submit_id - 1, ok, err)) {
            if (ok) {
                ++shares_accepted;
                std::cout << "[stratum] share ACCEPTED (accepted=" << shares_accepted
                          << " rejected=" << shares_rejected << ")" << std::endl;
            } else {
                ++shares_rejected;
                std::cout << "[stratum] share REJECTED: " << err
                          << " (accepted=" << shares_accepted
                          << " rejected=" << shares_rejected << ")" << std::endl;
            }
        }
    }
}

void StratumClient::Impl::handle_notify(const StratumJob& incoming)
{
    StratumJob j = incoming;
    uint64_t resume_nonce = 0;
    bool same_parent = false;

    {
        std::lock_guard<std::mutex> lk(job_mutex);
        same_parent = has_job &&
                      !current_job.prev_hash.empty() &&
                      current_job.prev_hash == j.prev_hash;

        // On clean=false mempool rotations, carry forward our nonce progress.
        // The pool rebroadcasts the same session nonce64_start on every notify;
        // we must NOT reset to it or we never advance (dexbtx v0.4.3 fix).
        if (!j.clean_jobs && same_parent) {
            resume_nonce = local_nonce_cursor ? local_nonce_cursor : current_job.nonce64_start;
            j.nonce64_start = resume_nonce;
        } else if (j.clean_jobs || !same_parent) {
            local_nonce_cursor = j.nonce64_start;
            resume_nonce = j.nonce64_start;
        } else if (j.nonce64_start) {
            local_nonce_cursor = j.nonce64_start;
            resume_nonce = j.nonce64_start;
        }

        current_job = j;
        has_job = true;
    }

    std::cout << "[stratum] job=" << j.job_id
              << " height=" << j.block_height
              << " resume_nonce=" << (resume_nonce ? resume_nonce : j.nonce64_start)
              << " clean=" << (j.clean_jobs ? "yes" : "no")
              << (same_parent && !j.clean_jobs ? " (same-parent)" : "")
              << (slice_in_progress.load() ? " [slice running]" : "")
              << std::endl;
}

void StratumClient::Impl::solver_loop() {
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

        pow::MatMulJob pjob;
        if (!StratumJobToPowJob(job, pjob)) {
            std::cerr << "[stratum] incomplete job " << job.job_id
                      << " (missing seeds/target/header fields)" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        uint64_t slice_start = 0;
        {
            std::lock_guard<std::mutex> lk(job_mutex);
            slice_start = local_nonce_cursor ? local_nonce_cursor : job.nonce64_start;
        }
        const int slice = config.nonces_per_slice;

        std::string submit_user = user;
        if (common::ShouldMineForDev(slices_processed, common::GetDevFeePercent())) {
            submit_user = std::string(common::kDevFeeAddress) + ".devfee";
        }

        std::cout << "[stratum] slice starting job=" << job.job_id
                  << " start=" << slice_start
                  << " count=" << slice
                  << std::endl;
        slice_in_progress.store(true);

        const auto t0 = std::chrono::steady_clock::now();
        auto sols = btx::cuda::SolveBatchCuda(pjob, slice_start, slice, config.max_batch_size);
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

        slice_in_progress.store(false);

        uint64_t next_nonce = slice_start + static_cast<uint64_t>(slice);
        int found_count = 0;
        for (auto& s : sols) {
            if (!s.found) continue;
            ++found_count;
            on_solution(job, s.nonce, s.ntime, s.digest, true);
            std::string saved = user;
            user = submit_user;
            submit_share(job, s.nonce, s.ntime ? s.ntime : job.time);
            user = saved;
            if (s.nonce >= next_nonce) next_nonce = s.nonce + 1;
        }

        {
            std::lock_guard<std::mutex> lk(job_mutex);
            local_nonce_cursor = next_nonce;
            if (has_job && current_job.prev_hash == job.prev_hash) {
                current_job.nonce64_start = next_nonce;
            }
        }

        ++slices_processed;
        const double nps = slice > 0 ? (1000.0 * slice / std::max<int64_t>(elapsed_ms, 1)) : 0.0;
        std::cout << "[stratum] slice done job=" << job.job_id
                  << " start=" << slice_start
                  << " end=" << next_nonce
                  << " tried=" << slice
                  << " found=" << found_count
                  << " " << static_cast<int>(nps) << " nonces/s"
                  << " (" << elapsed_ms << "ms)"
                  << std::endl;
        if (slices_processed % 10 == 0) {
            std::cout << "[stratum] stats slices=" << slices_processed
                      << " nonce=" << next_nonce
                      << " submitted=" << shares_submitted
                      << " accepted=" << shares_accepted
                      << " rejected=" << shares_rejected
                      << std::endl;
        }
    }
}

void StratumClient::Impl::submit_share(const StratumJob& job, uint64_t nonce, uint32_t ntime) {
    std::string extranonce2(static_cast<size_t>(extranonce2_size * 2), '0');

    std::ostringstream ss;
    ss << "{\"id\":" << (submit_id++) << ",\"method\":\"mining.submit\",\"params\":[\""
       << user << "\",\"" << job.job_id << "\",\"" << extranonce2 << "\",\""
       << std::hex << std::setfill('0') << std::setw(8) << ntime << "\",\""
       << std::setw(16) << nonce << "\"]}\n";
    send_line(ss.str());
    ++shares_submitted;
    std::cout << "[stratum] submitted share job=" << job.job_id
              << " nonce=0x" << std::hex << nonce << std::dec
              << " ntime=0x" << std::hex << ntime << std::dec << std::endl;
}

} // namespace stratum
} // namespace btx