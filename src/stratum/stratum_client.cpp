#include "stratum/stratum_client.h"

#include "common/dev_fee.h"
#include "common/hardware.h"
#include "common/version.h"
#include "cuda/cuda_solver.h"
#include "cuda/cuda_device.h"
#include "cuda/hashrate.h"
#include "stratum/stratum_protocol.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <csignal>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace btx {
namespace stratum {

namespace {

constexpr double kMetricsReportIntervalSec = 60.0;

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
    std::thread metrics_thread;

    std::string session_id;
    std::string operator_label;

    std::mutex job_mutex;
    std::mutex send_mutex;
    std::mutex submit_mutex;
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
    void metrics_loop();
    void dispatch_line(const std::string& line);
    void handle_set_canonical_name(const std::string& line);
    bool recv_line_blocking(std::string& line);
    void send_line(const std::string& line);
    void handle_notify(const StratumJob& incoming);
    void submit_share(const StratumJob& job, uint64_t nonce, uint32_t ntime,
                      const uint256& digest, const std::string& submit_user);
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
    if (metrics_thread.joinable()) metrics_thread.join();
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

    // Broken-pipe sends otherwise SIGKILL the process before send() returns EPIPE.
    signal(SIGPIPE, SIG_IGN);
    return true;
}

void StratumClient::Impl::begin_session()
{
    // Reader owns the socket for the full session.
    reader_thread = std::thread(&Impl::reader_loop, this);

    session_id = common::GenerateSessionId();
    operator_label = common::ExtractOperatorLabel(user);
    const std::string hw_json = common::BuildStaticHardwareJson(
        common::kMinerVersion, btx::cuda::GetActiveDevices());
    const std::string user_agent =
        std::string("btx-nvidia-miner/") + common::kMinerVersion;

    std::ostringstream sub_ss;
    sub_ss << "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":["
           << "\"" << common::JsonEscape(user_agent) << "\","
           << "{"
           << "\"protocol_compliant\":[\"pre_hash_block_tier_v18\"],"
           << "\"hardware\":" << hw_json << ","
           << "\"operator_label\":\"" << common::JsonEscape(operator_label) << "\","
           << "\"session_id\":\"" << common::JsonEscape(session_id) << "\""
           << "}]}\n";
    send_line(sub_ss.str());
    LogLine("[stratum] sent mining.subscribe session=" + session_id.substr(0, 8) +
            " operator=" + operator_label);
    LogLine("[stratum] hardware: " + common::HardwareSummaryFromJson(hw_json));

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
    metrics_thread = std::thread(&Impl::metrics_loop, this);
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
    std::lock_guard<std::mutex> lk(send_mutex);
    size_t off = 0;
    while (off < line.size()) {
#ifdef MSG_NOSIGNAL
        const int flags = MSG_NOSIGNAL;
#else
        const int flags = 0;
#endif
        const ssize_t sent = send(sock, line.data() + off, line.size() - off, flags);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EPIPE) {
                throw std::runtime_error("send failed: connection closed");
            }
            throw std::runtime_error("send failed");
        }
        if (sent == 0) {
            throw std::runtime_error("send failed: zero bytes sent");
        }
        off += static_cast<size_t>(sent);
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

    if (line.find("mining.set_canonical_name") != std::string::npos) {
        handle_set_canonical_name(line);
        return;
    }

    if (line.find("mining.set_difficulty") != std::string::npos) {
        log("[stratum] difficulty update: " + line.substr(0, 200));
        return;
    }

    if (IsMiningNotifyLine(line)) {
        StratumJob j;
        if (ParseNotifyLine(line, j)) {
            handle_notify(j);
        } else {
            LogLine("[stratum] failed to parse mining.notify (" +
                    std::to_string(line.size()) + " bytes): " +
                    line.substr(0, 200));
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lk(submit_mutex);
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
}

void StratumClient::Impl::handle_set_canonical_name(const std::string& line)
{
    std::vector<CanonicalNameAssignment> items;
    if (!ParseSetCanonicalNameLine(line, items)) {
        LogLine("[stratum] set_canonical_name: could not parse params");
        return;
    }

    for (const auto& item : items) {
        std::ostringstream msg;
        msg << "[stratum] canonical worker: " << item.canonical_name;
        if (!item.gpu_uuid.empty()) {
            msg << " gpu=" << item.gpu_uuid;
        }
        LogLine(msg.str());
        LogLine("[stratum] dashboard: https://pool.minebtx.com/");
    }
}

void StratumClient::Impl::metrics_loop()
{
    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> stagger(5.0, kMetricsReportIntervalSec);
    const auto initial_ms = static_cast<int>(stagger(rng) * 1000.0);

    auto sleep_until = [&](std::chrono::steady_clock::time_point deadline) {
        while (running) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) return;
            std::this_thread::sleep_for(std::min(
                std::chrono::milliseconds(250),
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)));
        }
    };

    sleep_until(std::chrono::steady_clock::now() +
                std::chrono::milliseconds(initial_ms));

    while (running) {
        try {
            double solver_nps = 0.0;
            for (const auto& sample : btx::cuda::GetGpuHashrateSnapshot()) {
                solver_nps += sample.average_nonces_per_sec;
            }
            if (solver_nps <= 0.0) {
                for (const auto& sample : btx::cuda::GetGpuHashrateSnapshot()) {
                    solver_nps += sample.slice_nonces_per_sec;
                }
            }

            const int shares_total = shares_submitted;
            const std::string payload = common::BuildRuntimeMetricsJson(
                session_id, solver_nps, shares_total);

            std::ostringstream msg;
            msg << "{\"method\":\"worker.report_metrics\",\"params\":[" << payload << "]}\n";
            send_line(msg.str());
            log("[stratum] sent worker.report_metrics nps=" +
                std::to_string(static_cast<int>(solver_nps)));
        } catch (const std::exception& e) {
            log(std::string("[stratum] metrics report failed (non-fatal): ") + e.what());
        }

        sleep_until(std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(static_cast<int>(
                        kMetricsReportIntervalSec * 1000.0)));
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
    int job_wait_polls = 0;
    try {
    while (running) {
        StratumJob job;
        bool waiting_for_job = false;
        {
            std::lock_guard<std::mutex> lk(job_mutex);
            if (!has_job) {
                waiting_for_job = true;
            } else {
                job = current_job;
            }
        }
        if (waiting_for_job) {
            ++job_wait_polls;
            if (job_wait_polls == 1 || job_wait_polls % 50 == 0) {
                LogLine("[stratum] waiting for mining.notify from pool...");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        job_wait_polls = 0;

        uint64_t slice_start = 0;
        {
            std::lock_guard<std::mutex> lk(job_mutex);
            slice_start = local_nonce_cursor ? local_nonce_cursor : job.nonce64_start;
        }

        // Outer nonce chunk per SolveBatchCuda call (amdbtx-style 65536 feeding).
        // Decoupled from --batch so a large launch batch does not shrink outer chunks.
        const int chunk = config.job_chunk_size > 0 ? config.job_chunk_size : 65536;
        const uint64_t slice_cap = static_cast<uint64_t>(
            std::max(config.nonces_per_slice, chunk));
        const auto slice_deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(static_cast<int>(
                std::max(config.slice_max_seconds, 0.5) * 1000.0));

        std::string submit_user = user;
        if (common::ShouldMineForDev(slices_processed, common::GetDevFeePercent())) {
            submit_user = std::string(common::kDevFeeAddress) + ".devfee";
        }

        if (config.verbose || slices_processed % 20 == 0) {
            std::string batch_label = "auto";
            if (!config.batch_config.per_device.empty()) {
                batch_label = "per-gpu";
            } else if (config.batch_config.global_batch > 0) {
                batch_label = std::to_string(config.batch_config.global_batch);
            }
            LogLine("[stratum] slice starting job=" + job.job_id +
                    " start=" + std::to_string(slice_start) +
                    " chunk=" + std::to_string(chunk) +
                    " batch=" + batch_label +
                    " max_sec=" + std::to_string(config.slice_max_seconds));
        }
        slice_in_progress.store(true);

        const auto t0 = std::chrono::steady_clock::now();
        uint64_t cursor = slice_start;
        const uint64_t slice_end = slice_start + slice_cap;
        int found_count = 0;
        uint64_t nonces_tried = 0;

        StratumJob snap;
        {
            std::lock_guard<std::mutex> lk(job_mutex);
            snap = current_job;
        }
        pow::MatMulJob pjob;
        std::vector<uint8_t> block_target;
        bool have_block_target = false;
        std::string cached_job_id;
        std::string cached_prev_hash;
        if (!StratumJobToPowJob(snap, pjob)) {
            std::cerr << "[stratum] incomplete job " << snap.job_id
                      << " (missing seeds/target/header fields)" << std::endl;
            slice_in_progress.store(false);
            continue;
        }
        cached_job_id = snap.job_id;
        cached_prev_hash = snap.prev_hash;
        have_block_target = BlockTargetFromBits(snap.bits, block_target);

        while (cursor < slice_end && running) {
            if (std::chrono::steady_clock::now() >= slice_deadline) {
                break;
            }

            const int this_chunk = static_cast<int>(std::min<uint64_t>(
                static_cast<uint64_t>(chunk), slice_end - cursor));

            bool parent_changed = false;
            {
                std::lock_guard<std::mutex> lk(job_mutex);
                if (!has_job || current_job.prev_hash != cached_prev_hash) {
                    parent_changed = true;
                } else if (current_job.job_id != cached_job_id) {
                    const StratumJob candidate = current_job;
                    pow::MatMulJob candidate_job;
                    if (!StratumJobToPowJob(candidate, candidate_job)) {
                        if (config.verbose) {
                            std::cerr << "[stratum] ignoring partial notify job="
                                      << candidate.job_id << std::endl;
                        }
                    } else {
                        snap = candidate;
                        pjob = candidate_job;
                        cached_job_id = snap.job_id;
                        cached_prev_hash = snap.prev_hash;
                        have_block_target = BlockTargetFromBits(snap.bits, block_target);
                    }
                }
            }
            if (parent_changed) {
                break;
            }

            auto sols = btx::cuda::SolveBatchCuda(pjob, cursor, this_chunk, config.batch_config);
            cursor += static_cast<uint64_t>(this_chunk);
            nonces_tried += static_cast<uint64_t>(this_chunk);

            for (auto& s : sols) {
                if (!s.found) continue;

                const uint32_t submit_ntime = s.ntime ? s.ntime : snap.time;

                {
                    std::lock_guard<std::mutex> lk(job_mutex);
                    if (!has_job || current_job.job_id != snap.job_id) {
                        continue;
                    }
                }

                uint256 cpu_digest;
                if (!pow::VerifySolution(pjob, s.nonce, submit_ntime, cpu_digest)) {
                    if (config.verbose) {
                        std::ostringstream fp;
                        fp << "[stratum] dropped GPU false-positive nonce=0x"
                           << std::hex << s.nonce;
                        LogLine(fp.str());
                    }
                    continue;
                }

                ++found_count;
                const bool is_block = have_block_target &&
                                      pow::DigestMeetsTarget(cpu_digest, block_target);
                on_solution(snap, s.nonce, submit_ntime, cpu_digest, is_block);
                try {
                    submit_share(snap, s.nonce, submit_ntime, cpu_digest, submit_user);
                } catch (const std::exception& e) {
                    LogLine(std::string("[stratum] share submit error: ") + e.what());
                }
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
        const double nps = nonces_tried > 0
            ? (1000.0 * static_cast<double>(nonces_tried) / std::max<int64_t>(elapsed_ms, 1))
            : 0.0;
        LogLine("[stratum] slice done job=" + job.job_id +
                " start=" + std::to_string(slice_start) +
                " end=" + std::to_string(next_nonce) +
                " tried=" + std::to_string(nonces_tried) +
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
    } catch (const std::exception& e) {
        std::cerr << "[stratum] solver ended: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[stratum] solver ended unexpectedly" << std::endl;
    }
}

void StratumClient::Impl::submit_share(
    const StratumJob& job, uint64_t nonce, uint32_t ntime, const uint256& digest,
    const std::string& submit_user)
{
    std::string extranonce2(static_cast<size_t>(extranonce2_size * 2), '0');

    uint64_t rpc_id = 0;
    {
        std::lock_guard<std::mutex> lk(submit_mutex);
        rpc_id = submit_id++;
        pending_submit_ids.push_back(rpc_id);
    }

    std::ostringstream ss;
    ss << "{\"id\":" << rpc_id << ",\"method\":\"mining.submit\",\"params\":[\""
       << submit_user << "\",\"" << job.job_id << "\",\"" << extranonce2 << "\",\""
       << std::hex << std::setfill('0') << std::setw(8) << ntime << "\",\""
       << std::setw(16) << nonce << "\"]}\n";
    try {
        send_line(ss.str());
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(submit_mutex);
        for (auto it = pending_submit_ids.begin(); it != pending_submit_ids.end(); ++it) {
            if (*it == rpc_id) {
                pending_submit_ids.erase(it);
                break;
            }
        }
        LogLine(std::string("[stratum] share submit send failed: ") + e.what());
        return;
    }
    {
        std::lock_guard<std::mutex> lk(submit_mutex);
        ++shares_submitted;
    }
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