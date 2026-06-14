#include "common/updater.h"
#include "common/version.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <unistd.h>
#include <vector>

namespace btx {
namespace common {
namespace {

constexpr const char* kRepoOwner = "thekillsquad007";
constexpr const char* kRepoName = "btx-nvidia-miner";
constexpr const char* kReleaseAsset = "btx-miner-linux-x86_64.tar.gz";

std::string ShellQuote(const std::string& value)
{
    std::ostringstream ss;
    ss << '\'';
    for (char ch : value) {
        if (ch == '\'') {
            ss << "'\\''";
        } else {
            ss << ch;
        }
    }
    ss << '\'';
    return ss.str();
}

std::string RunCommand(const std::string& cmd)
{
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return output;
    }
    while (true) {
        const size_t n = std::fread(buffer.data(), 1, buffer.size(), pipe);
        if (n == 0) {
            break;
        }
        output.append(buffer.data(), n);
    }
    pclose(pipe);
    return output;
}

bool RunCommandChecked(const std::string& cmd, std::string& error_out)
{
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        error_out = "command failed (" + std::to_string(rc) + "): " + cmd;
        return false;
    }
    return true;
}

std::string ExtractJsonString(const std::string& json, const std::string& key)
{
    const std::string needle = "\"" + key + "\":\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return {};
    }
    const size_t start = pos + needle.size();
    const size_t end = json.find('"', start);
    if (end == std::string::npos || end <= start) {
        return {};
    }
    return json.substr(start, end - start);
}

int ParseVersionPart(const std::string& text, size_t& pos)
{
    while (pos < text.size() && !std::isdigit(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    int value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        value = value * 10 + (text[pos] - '0');
        ++pos;
    }
    return value;
}

bool VersionLess(const std::string& left, const std::string& right)
{
    size_t lp = 0;
    size_t rp = 0;
    for (int part = 0; part < 3; ++part) {
        const int l = ParseVersionPart(left, lp);
        const int r = ParseVersionPart(right, rp);
        if (l < r) return true;
        if (l > r) return false;
        while (lp < left.size() && (left[lp] == '.' || std::isspace(static_cast<unsigned char>(left[lp])))) {
            ++lp;
        }
        while (rp < right.size() && (right[rp] == '.' || std::isspace(static_cast<unsigned char>(right[rp])))) {
            ++rp;
        }
    }
    return false;
}

std::string StripTagPrefix(std::string tag)
{
    if (!tag.empty() && tag[0] == 'v') {
        tag.erase(0, 1);
    }
    return tag;
}

} // namespace

std::string GetExecutablePath()
{
    std::array<char, 4096> buffer{};
    const ssize_t len = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (len <= 0) {
        return {};
    }
    buffer[static_cast<size_t>(len)] = '\0';
    return std::string(buffer.data());
}

UpdateInfo CheckForUpdate()
{
    UpdateInfo info;
    info.current_version = kMinerVersion;

    const std::string api =
        "https://api.github.com/repos/" + std::string(kRepoOwner) + "/" +
        kRepoName + "/releases/latest";
    const std::string cmd =
        "curl -fsSL --connect-timeout 10 --max-time 30 " + ShellQuote(api);
    const std::string body = RunCommand(cmd);
    if (body.empty()) {
        return info;
    }

    info.release_tag = ExtractJsonString(body, "tag_name");
    info.latest_version = StripTagPrefix(info.release_tag);
    info.release_notes = ExtractJsonString(body, "body");

    const std::string asset_needle = "\"name\":\"" + std::string(kReleaseAsset) + "\"";
    const size_t asset_pos = body.find(asset_needle);
    if (asset_pos != std::string::npos) {
        const size_t url_key = body.rfind("\"browser_download_url\":\"", asset_pos);
        if (url_key != std::string::npos) {
            const size_t start = url_key + 24;
            const size_t end = body.find('"', start);
            if (end != std::string::npos) {
                info.download_url = body.substr(start, end - start);
            }
        }
    }

    if (!info.latest_version.empty() && !info.download_url.empty() &&
        VersionLess(info.current_version, info.latest_version)) {
        info.update_available = true;
    }
    return info;
}

bool InstallUpdate(const UpdateInfo& info, std::string& error_out)
{
    if (!info.update_available || info.download_url.empty()) {
        error_out = "no update available";
        return false;
    }

    const std::string exe = GetExecutablePath();
    if (exe.empty()) {
        error_out = "could not resolve /proc/self/exe";
        return false;
    }

    namespace fs = std::filesystem;
    const fs::path state_dir =
        fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") /
        ".local/share/btx-nvidia-miner/update";
    std::error_code ec;
    fs::create_directories(state_dir, ec);

    const fs::path archive = state_dir / kReleaseAsset;
    const fs::path extract_dir = state_dir / "extract";
    const fs::path staged = state_dir / "btx-miner.new";

    const std::string download_cmd =
        "curl -fsSL --connect-timeout 15 --max-time 300 -o " +
        ShellQuote(archive.string()) + " " + ShellQuote(info.download_url);
    if (!RunCommandChecked(download_cmd, error_out)) {
        return false;
    }

    fs::remove_all(extract_dir, ec);
    fs::create_directories(extract_dir, ec);
    const std::string extract_cmd =
        "tar -xzf " + ShellQuote(archive.string()) + " -C " + ShellQuote(extract_dir.string());
    if (!RunCommandChecked(extract_cmd, error_out)) {
        return false;
    }

    const fs::path unpacked = extract_dir / "btx-miner";
    if (!fs::exists(unpacked)) {
        error_out = "release archive missing btx-miner binary";
        return false;
    }

    fs::copy_file(unpacked, staged, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        error_out = "failed to stage new binary: " + ec.message();
        return false;
    }
    fs::permissions(staged, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                    fs::perm_options::add, ec);

    const fs::path backup = fs::path(exe).string() + ".bak";
    fs::remove(backup, ec);
    fs::copy_file(exe, backup, fs::copy_options::overwrite_existing, ec);
    fs::rename(staged, exe, ec);
    if (ec) {
        error_out = "failed to replace binary: " + ec.message();
        return false;
    }
    return true;
}

void ReexecCurrentProcess(const std::vector<std::string>& argv)
{
    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& arg : argv) {
        cargv.push_back(const_cast<char*>(arg.c_str()));
    }
    cargv.push_back(nullptr);
    execv(cargv[0], cargv.data());
    std::perror("execv");
    _exit(127);
}

} // namespace common
} // namespace btx