#pragma once

#include <string>
#include <vector>

namespace btx {
namespace common {

std::string JsonEscape(const std::string& s);
std::string GenerateSessionId();
std::string ExtractOperatorLabel(const std::string& user);

// One-shot fingerprint for mining.subscribe extension["hardware"].
std::string BuildStaticHardwareJson(const std::string& miner_version,
                                    const std::vector<int>& active_gpu_indices);

// Periodic worker.report_metrics params[0] payload.
std::string BuildRuntimeMetricsJson(const std::string& session_id,
                                    double solver_nps,
                                    int shares_session_total);

std::string HardwareSummaryFromJson(const std::string& hardware_json);

} // namespace common
} // namespace btx