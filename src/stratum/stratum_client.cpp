#include "stratum/stratum_client.h"

#include "common/dev_fee.h"
#include "cuda/cuda_solver.h"
#include "cuda/hashrate.h"
#include "stratum/stratum_protocol.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace btx {
namespace stratum {

namespace {
void LogLine(const std::string& msg)
{
    std::cout << msg << std::endl << std::flush;
}

std::string DigestHex(const uint256& digest)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 31; i >= 0; --i) {
        ss << std::setw(2) << static_cast<unsigned>(digest.data()[i]);
    }
    return ss.str();
}
} // namespace

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

    std::mutex handshake_mutex;
    std::condition_variable handshake_cv;
    bool subscribed = false;
    bool authorized = false;
    std::string auth_error;

    std::string extranonce1;
    int extranonce2_size = 4;
    uint64_t submit_id = 3;
    std::deque<uint64_t> pending_submit_ids;

    int shares_submitted = 0;
    int shares_accepted = 0;
    int shares_rejected = 0;
    int slices_processed = 0;
    uint64_t local_nonce_cursor = 0;
    std::atomic<bool> slice_in_progress{false};

    void reset_session_state();
    bool connect_socket();
    void close_socket();
    void join_session_threads();
    void begin_session();
    void reader_loop();
    void solver_loop();
    void dispatch_line(const std::string& line);
    bool recv_line_blocking(std::string& line);
    void send_line(const std::string& line);
    void handle_notify(const StratumJob& incoming);
    void submit_share(const StratumJob& job, uint64_t nonce, uint32_t ntime, const uint256& digest);
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
            impl->reset_session_state();
            if (!impl->connect_socket()) {
                throw std::runtime_error("failed to open socket");
            }
            impl->begin_session();

            if (impl->reader_thread.joinable()) impl->reader_thread.join();
            if (impl->solver_thread.joinable()) impl->solver_thread.join();
        } catch (const std::exception& e) {
            std::cerr << "[stratum] session error: " << e.what() << " — reconnecting in 5s" << std::endl;
            impl->close_socket();
            impl->join_session_threads();
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
}

void StratumClient::stop() {
    impl->running = false;
    impl->handshake_cv.notify_all();
    impl->close_socket();
    impl->join_session_threads();
}

void StratumClient::Impl::close_socket()
{
    if (sock >= 0) {
        shutdown(sock, SHUT_RDWR);
        close(sock);
        sock = -1;
    }
}

void StratumClient::Impl::join_session_threads()
{
    handshake_cv.notify_all();
    if (reader_thread.joinable()) reader_thread.join();
    if (solver_thread.joinable()) solver_thread.join();
}

void StratumClient::Impl::reset_session_state()
{
    subscribed = false;
    authorized = false;
    auth_error.clear();
    has_job = false;
    local_nonce_cursor = 0;
    slice_in_progress.store(false);
    slices_processed = 0;
}

void StratumClient::Impl::log(const std::string& msg) const
{
    if (config.verbose) {
        LogLine(msg);
    }
}

bool StratumClient::Impl::connect_socket()
{
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
    return true;
}

void StratumClient::Impl::begin_session()
{
    // Reader owns the socket for the full session.
    reader_thread = std::thread(&Impl::reader_loop, this);

    std::string sub = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"btx-nvidia-miner/0.2.14\",{\"protocol_compliant\":[\"pre_hash_block_tier_v18\"]}]}\n";
    send_line(sub);
    LogLine("[stratum] sent mining.subscribe");

    {
        std::unique_lock<std::mutex> lk(handshake_mutex);
        if (!handshake_cv.wait_for(lk, std::chrono::seconds(30), [&] { return subscribed || !running; })) {
            close_socket();
            join_session_threads();
            throw std::runtime_error("timed out waiting for mining.subscribe response");
        }
        if (!running) {
            close_socket();
            join_session_threads();
            return;
        }
        if (!subscribed) {
            close_socket();
            join_session_threads();
            throw std::runtime_error("disconnected before subscribe completed");
        }
    }

    LogLine("[stratum] subscribed extranonce1=" + extranonce1 + " en2_size=" + std::to_string(extranonce2_size));

    // Start mining immediately after subscribe — don't block on authorize.
    solver_thread = std::thread(&Impl::solver_loop, this);
    LogLine("[stratum] solver started");

    std::string auth = "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"" + user + "\",\"" + pass + "\"]}\n";
    send_line(auth);
    LogLine("[stratum] sent mining.authorize");

    {
        std::unique_lock<std::mutex> lk(handshake_mutex);
        handshake_cv.wait_for(lk, std::chrono::seconds(30), [&] { return authorized || !auth_error.empty() || !running; });
        if (!auth_error.empty()) {
            close_socket();
            join_session_threads();
            throw std::runtime_error("mining.authorize rejected: " + auth_error);
        }
        if (!authorized) {
            LogLine("[stratum] warning: authorize still pending — mining continues, shares may fail until authorized");
        }
    }

    LogLine("[stratum] connected to " + host + ":" + std::to_string(port) + " as " + user);
}

void StratumClient::Impl::send_line(const std::string& line) {
    if (sock < 0) return;
    ssize_t sent = send(sock, line.data(), line.size(), 0);
    if (sent < 0) {
        throw std::runtime_error("send failed");
    }
}

bool StratumClient::Impl::recv_line_blocking(std::string& line)
{
    line.clear();
    char ch = 0;
    while (running) {
        ssize_t n = recv(sock, &ch, 1, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        if (ch == '\n') return !line.empty();
        if (ch != '\r') line.push_back(ch);
    }
    return false;
}

void StratumClient::Impl::dispatch_line(const std::string& line)
{
    log("[stratum] recv: " + line.substr(0, 300) + (line.size() > 300 ? "..." : ""));

    // Subscribe response embeds the string "mining.notify" in capabilities —
    // must handle RPC responses before notify detection.
    if (!subscribed && ParseSubscribeResult(line, extranonce1, extranonce2_size)) {
        std::lock_guard<std::mutex> lk(handshake_mutex);
        subscribed = true;
        handshake_cv.notify_all();
        return;
    }

    bool ok = false;
    std::string err;
    if (!authorized && ParseRpcResult(line, 2, ok, err)) {
        std::lock_guard<std::mutex> lk(handshake_mutex);
        if (ok) {
            authorized = true;
        } else {
            auth_error = err.empty() ? "authorize returned false" : err;
        }
        handshake_cv.notify_all();
        return;
    }

    if (IsMiningNotifyLine(line)) {
        StratumJob j;
        if (ParseNotifyLine(line, j)) {
            handle_notify(j);
        } else {
            std::cerr << "[stratum] failed to parse mining.notify: "
                      << line.substr(0, 200) << std::endl;
        }
        return;
    }

    for (auto it = pending_submit_ids.begin(); it != pending_submit_ids.end(); ++it) {
        if (!ParseRpcResult(line, *it, ok, err)) {
            continue;
        }
        pending_submit_ids.erase(it);
        if (ok) {
            ++shares_accepted;
            LogLine("[stratum] share ACCEPTED (accepted=" + std::to_string(shares_accepted) +
                    " rejected=" + std::to_string(shares_rejected) + ")");
        } else {
            ++shares_rejected;
            LogLine("[stratum] share REJECTED: " + err +
                    " (accepted=" + std::to_string(shares_accepted) +
                    " rejected=" + std::to_string(shares_rejected) + ")");
        }
        break;
    }
}

void StratumClient::Impl::reader_loop() {
    std::string line;
    try {
        while (running) {
            if (!recv_line_blocking(line)) {
                break;
            }
            dispatch_line(line);
        }
    } catch (const std::exception& e) {
        std::cerr << "[stratum] reader ended: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[stratum] reader ended unexpectedly" << std::endl;
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

    LogLine("[stratum] job=" + j.job_id +
            " height=" + std::to_string(j.block_height) +
            " resume_nonce=" + std::to_string(resume_nonce ? resume_nonce : j.nonce64_start) +
            " clean=" + (j.clean_jobs ? "yes" : "no") +
            (same_parent && !j.clean_jobs ? " (same-parent)" : "") +
            (slice_in_progress.load() ? " [slice running]" : ""));
}

void StratumClient::Impl::solver_loop() {
    LogLine("[stratum] solver loop ready");
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

        uint64_t slice_start = 0;
        {
            std::lock_guard<std::mutex> lk(job_mutex);
            slice_start = local_nonce_cursor ? local_nonce_cursor : job.nonce64_start;
        }
        const int slice = config.nonces_per_slice;
        const int chunk = config.job_chunk_size > 0 ? config.job_chunk_size : 32;

        std::string submit_user = user;
        if (common::ShouldMineForDev(slices_processed, common::GetDevFeePercent())) {
            submit_user = std::string(common::kDevFeeAddress) + ".devfee";
        }

        LogLine("[stratum] slice starting job=" + job.job_id +
                " start=" + std::to_string(slice_start) +
                " count=" + std::to_string(slice));
        slice_in_progress.store(true);

        const auto t0 = std::chrono::steady_clock::now();
        uint64_t cursor = slice_start;
        const uint64_t slice_end = slice_start + static_cast<uint64_t>(slice);
        int found_count = 0;

        while (cursor < slice_end) {
            const int this_chunk = static_cast<int>(
                std::min<uint64_t>(static_cast<uint64_t>(chunk), slice_end - cursor));

            StratumJob snap;
            {
                std::lock_guard<std::mutex> lk(job_mutex);
                snap = current_job;
            }

            pow::MatMulJob pjob;
            if (!StratumJobToPowJob(snap, pjob)) {
                std::cerr << "[stratum] incomplete job " << snap.job_id
                          << " (missing seeds/target/header fields)" << std::endl;
                break;
            }

            std::vector<uint8_t> block_target;
            const bool have_block_target = BlockTargetFromBits(snap.bits, block_target);

            auto sols = btx::cuda::SolveBatchCuda(pjob, cursor, this_chunk, config.max_batch_size);
            cursor += static_cast<uint64_t>(this_chunk);

            for (auto& s : sols) {
                if (!s.found) continue;

                const uint32_t submit_ntime = s.ntime ? s.ntime : snap.time;
                uint256 verify_digest;
                if (!pow::VerifySolution(pjob, s.nonce, submit_ntime, verify_digest) ||
                    !pow::DigestMeetsTarget(verify_digest, pjob.target)) {
                    continue;
                }

                ++found_count;
                const bool is_block = have_block_target &&
                                      pow::DigestMeetsTarget(verify_digest, block_target);
                on_solution(snap, s.nonce, submit_ntime, verify_digest, is_block);
                std::string saved = user;
                user = submit_user;
                submit_share(snap, s.nonce, submit_ntime, verify_digest);
                user = saved;
            }

            {
                std::lock_guard<std::mutex> lk(job_mutex);
                local_nonce_cursor = cursor;
                if (has_job && current_job.prev_hash == snap.prev_hash) {
                    current_job.nonce64_start = cursor;
                }
            }
        }

        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        slice_in_progress.store(false);

        const uint64_t next_nonce = cursor;

        ++slices_processed;
        const double nps = slice > 0 ? (1000.0 * slice / std::max<int64_t>(elapsed_ms, 1)) : 0.0;
        LogLine("[stratum] slice done job=" + job.job_id +
                " start=" + std::to_string(slice_start) +
                " end=" + std::to_string(next_nonce) +
                " tried=" + std::to_string(slice) +
                " found=" + std::to_string(found_count) +
                " " + std::to_string(static_cast<int>(nps)) + " nonces/s" +
                " (" + std::to_string(elapsed_ms) + "ms)");
        LogLine("[stratum] " + btx::cuda::FormatGpuHashrateLog());
        if (slices_processed % 10 == 0) {
            LogLine("[stratum] stats slices=" + std::to_string(slices_processed) +
                    " nonce=" + std::to_string(next_nonce) +
                    " submitted=" + std::to_string(shares_submitted) +
                    " accepted=" + std::to_string(shares_accepted) +
                    " rejected=" + std::to_string(shares_rejected));
        }
    }
}

void StratumClient::Impl::submit_share(
    const StratumJob& job, uint64_t nonce, uint32_t ntime, const uint256& digest)
{
    std::string extranonce2(static_cast<size_t>(extranonce2_size * 2), '0');

    const uint64_t rpc_id = submit_id++;
    pending_submit_ids.push_back(rpc_id);

    std::ostringstream ss;
    ss << "{\"id\":" << rpc_id << ",\"method\":\"mining.submit\",\"params\":[\""
       << user << "\",\"" << job.job_id << "\",\"" << extranonce2 << "\",\""
       << std::hex << std::setfill('0') << std::setw(8) << ntime << "\",\""
       << std::setw(16) << nonce << "\"]}\n";
    send_line(ss.str());
    ++shares_submitted;
    std::ostringstream slog;
    slog << "[stratum] submitted share job=" << job.job_id
         << " height=" << std::dec << job.block_height
         << " nonce=0x" << std::hex << nonce
         << " ntime=0x" << ntime
         << " digest=" << DigestHex(digest);
    LogLine(slog.str());
}

} // namespace stratum
} // namespace btx