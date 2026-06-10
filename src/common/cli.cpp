#include <string>
#include <vector>

namespace btx {
namespace common {

struct Options {
    bool benchmark = false;
    bool force_cpu = false;
    // ... more later
};

Options parse_cli(int argc, char** argv);

} // namespace common
} // namespace btx

btx::common::Options btx::common::parse_cli(int, char**) { return {}; }
