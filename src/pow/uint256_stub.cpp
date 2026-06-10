#include "pow/uint256_stub.h"

#include <iomanip>
#include <sstream>

std::string uint256::GetHex() const
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 31; i >= 0; --i) {   // display big-endian like Bitcoin
        ss << std::setw(2) << static_cast<unsigned>(m_data[i]);
    }
    return ss.str();
}

uint256 uint256S(const char* hex)
{
    uint256 out;
    // very minimal, assumes 64 hex chars, no 0x
    if (!hex) return out;
    size_t len = std::strlen(hex);
    if (len > 64) len = 64;
    for (size_t i = 0; i < len; i += 2) {
        unsigned byte = 0;
        sscanf(hex + i, "%2x", &byte);
        // store in the same order as constructor from bytes (internal little in our model)
        // For simplicity put MSB first in display order into high bytes.
        // This is only used for debug/CLI, not consensus.
        size_t byte_idx = (64 - len) / 2 + i/2;
        if (byte_idx < 32) {
            out.data()[31 - (byte_idx)] = static_cast<uint8_t>(byte); // rough
        }
    }
    return out;
}
