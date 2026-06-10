#include <iostream>
#include <string>

namespace btx {
namespace common {

void log_info(const std::string& s) { std::cout << "[info] " << s << std::endl; }
void log_warn(const std::string& s) { std::cout << "[warn] " << s << std::endl; }

} // namespace common
} // namespace btx
