#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

// Minimal stand-in for the node's uint256 so the pow reference can be self-contained.
// It stores 32 bytes. Comparison and data() are provided for the miner.
// The exact byte order for "ToCanonical" / sigma is handled inside the pow code to match the node.

class uint256 {
public:
    uint256() { m_data.fill(0); }
    explicit uint256(const std::array<uint8_t, 32>& d) : m_data(d) {}
    explicit uint256(const uint8_t* p) { std::memcpy(m_data.data(), p, 32); }

    const uint8_t* data() const { return m_data.data(); }
    uint8_t* data() { return m_data.data(); }

    bool operator==(const uint256& o) const { return m_data == o.m_data; }
    bool operator!=(const uint256& o) const { return !(*this == o); }

    void SetNull() { m_data.fill(0); }
    bool IsNull() const {
        for (auto b : m_data) if (b) return false;
        return true;
    }

    std::string GetHex() const; // implemented in .cpp for convenience

private:
    std::array<uint8_t, 32> m_data{};
};

inline uint256 uint256S(const char* hex); // simple parser in .cpp

namespace btx {
using uint256 = ::uint256;
}
