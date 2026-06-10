#pragma once

#include <cstdint>
#include <string>

namespace btx {
namespace common {

// Dev fee address (hard-coded as requested).
// 1% of block rewards / hashrate will go here for both solo and pool modes.
inline constexpr const char* kDevFeeAddress = "btx1z0069dewdztkwnrxx97lt9c5paynh0nynegqxq2kgykh0ct8xaggq0953gx";

// Default dev fee percentage (0.0 - 5.0 recommended). Can be overridden by CLI/env.
inline constexpr float kDefaultDevFeePercent = 1.0f;

// Helper to get the effective dev fee percent (clamped).
float GetDevFeePercent(float override = -1.0f);

// Returns true if the current work slice should be credited to dev (for pool time-slicing).
// period is the total slices in a "cycle", e.g. 100 for 1%.
bool ShouldMineForDev(uint64_t slice_index, float fee_percent);

// For solo: the reward split (user gets 1 - fee, dev gets fee).
// Returns the dev portion in satoshis given the full block reward (or coinbase value).
uint64_t ComputeDevReward(uint64_t full_value, float fee_percent);

} // namespace common
} // namespace btx
