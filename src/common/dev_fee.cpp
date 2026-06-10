#include "common/dev_fee.h"

#include <algorithm>
#include <cmath>

namespace btx {
namespace common {

float GetDevFeePercent(float override)
{
    if (override >= 0.0f) {
        return std::min(5.0f, std::max(0.0f, override));
    }
    const char* env = std::getenv("BTX_DEV_FEE_PCT");
    if (env) {
        float v = std::strtof(env, nullptr);
        return std::min(5.0f, std::max(0.0f, v));
    }
    return kDefaultDevFeePercent;
}

bool ShouldMineForDev(uint64_t slice_index, float fee_percent)
{
    if (fee_percent <= 0.0f) return false;
    if (fee_percent >= 100.0f) return true;
    // Simple deterministic cycle: e.g. for 1% we mine 1 out of every 100 slices for dev.
    const int cycle = 100;
    const int dev_slices = std::max(1, static_cast<int>(std::round(fee_percent)));
    return (slice_index % cycle) < static_cast<uint64_t>(dev_slices);
}

uint64_t ComputeDevReward(uint64_t full_value, float fee_percent)
{
    if (fee_percent <= 0.0f) return 0;
    double frac = std::min(0.05, std::max(0.0, static_cast<double>(fee_percent) / 100.0));
    return static_cast<uint64_t>(std::llround(static_cast<double>(full_value) * frac));
}

} // namespace common
} // namespace btx
