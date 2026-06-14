#pragma once

#include <string>
#include <vector>

namespace btx {
namespace common {

struct UpdateInfo {
    bool update_available = false;
    std::string current_version;
    std::string latest_version;
    std::string release_tag;
    std::string download_url;
    std::string release_notes;
};

// Query GitHub releases/latest for thekillsquad007/btx-nvidia-miner.
UpdateInfo CheckForUpdate();

// Download release tarball and atomically replace the running binary.
// Returns true if installed; on success the caller should re-exec.
bool InstallUpdate(const UpdateInfo& info, std::string& error_out);

std::string GetExecutablePath();

// Re-run the current binary with the same argv (after update).
[[noreturn]] void ReexecCurrentProcess(const std::vector<std::string>& argv);

} // namespace common
} // namespace btx