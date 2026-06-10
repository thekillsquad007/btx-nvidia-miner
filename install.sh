#!/usr/bin/env bash
#
# Easy installer for btx-nvidia-miner on a remote Linux machine with NVIDIA GPU.
#
# Usage (recommended one-liner):
#   curl -fsSL https://raw.githubusercontent.com/thekillsquad007/btx-nvidia-miner/main/install.sh | bash -s -- --address btx1zYOURADDRESS...
#
# The script will:
#   - Install basic build dependencies
#   - Check for NVIDIA GPU
#   - Require CUDA Toolkit (nvcc) to be installed on the rig (standard for mining boxes)
#   - Build the miner from source (CUDA enabled)
#   - Install the binary to ~/.local/bin/btx-miner (or /usr/local/bin if run as root)
#   - Print ready-to-use example commands for pool and solo (with dev fee applied)
#
# After install you can run it in tmux/screen or set up a systemd user service.
#
set -euo pipefail

DEV_FEE_ADDRESS="btx1z0069dewdztkwnrxx97lt9c5paynh0nynegqxq2kgykh0ct8xaggq0953gx"
DEFAULT_DEV_FEE_PCT="1.0"

REPO_URL="https://github.com/thekillsquad007/btx-nvidia-miner.git"   # thekillsquad007's btx-nvidia-miner fork
BINARY_NAME="btx-miner"

INSTALL_PREFIX="${HOME}/.local"
BIN_DIR="${INSTALL_PREFIX}/bin"

USER_ADDRESS=""
POOL_URL="stratum+tcp://stratum.minebtx.com:3333"
WORKER_NAME=""

print_help() {
    cat <<EOF
btx-nvidia-miner installer

Options:
  --address ADDR     Your BTX payout address (btx1z...). Required for examples.
  --pool URL         Pool to use in the printed example (default: $POOL_URL)
  --worker NAME      Worker name suffix (default: hostname)
  --prefix DIR       Install prefix (default: $INSTALL_PREFIX)
  -h, --help         This help

The installer builds from source and expects the CUDA Toolkit (nvcc) to be
available on \$PATH. This is the normal situation on a machine you intend to
mine with.

Dev fee of ${DEFAULT_DEV_FEE_PCT}% is built-in and goes to:
  ${DEV_FEE_ADDRESS}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --address) USER_ADDRESS="$2"; shift 2 ;;
        --pool)    POOL_URL="$2"; shift 2 ;;
        --worker)  WORKER_NAME="$2"; shift 2 ;;
        --prefix)  INSTALL_PREFIX="$2"; BIN_DIR="${INSTALL_PREFIX}/bin"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown arg: $1"; print_help; exit 1 ;;
    esac
done

if [[ -z "$USER_ADDRESS" ]]; then
    echo "ERROR: --address is required (your btx1z... payout address)"
    print_help
    exit 1
fi

if [[ -z "$WORKER_NAME" ]]; then
    WORKER_NAME="$(hostname -s 2>/dev/null || echo rig)"
fi

echo "=== btx-nvidia-miner installer ==="
echo "User address : $USER_ADDRESS"
echo "Dev fee addr : $DEV_FEE_ADDRESS (${DEFAULT_DEV_FEE_PCT}%)"
echo "Pool example : $POOL_URL"
echo "Worker       : ${USER_ADDRESS}.${WORKER_NAME}"
echo

# Basic deps (Ubuntu/Debian + some RedHat)
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y build-essential cmake git curl pkg-config \
        libcurl4-openssl-dev || true
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf groupinstall -y "Development Tools" || true
    sudo dnf install -y cmake git curl pkgconfig libcurl-devel || true
elif command -v yum >/dev/null 2>&1; then
    sudo yum groupinstall -y "Development Tools" || true
    sudo yum install -y cmake git curl pkgconfig libcurl-devel || true
fi

# Check for NVIDIA
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed."
else
    echo "GPU(s) detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

# Check for nvcc (CUDA Toolkit)
if ! command -v nvcc >/dev/null 2>&1; then
    cat <<'EOM'
ERROR: nvcc (CUDA Toolkit) not found in PATH.

On the mining rig you must have the CUDA Toolkit installed that matches your driver.
The full "cuda-toolkit-12-*" can be 20-30+ GB on disk (that's probably the "27gbs" you saw).

For building this miner you only need the compiler + basic runtime headers.
Use a much smaller/minimal install instead:

  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-compiler-12-6 cuda-cudart-dev-12-6

(If 12-6 is not available, try 12-5 or 12-4. You can list options with: apt-cache search cuda-compiler-12)

Then make sure nvcc is in PATH:

  export PATH=/usr/local/cuda/bin:$PATH
  nvcc --version

After that, re-run this exact installer command again.

The final btx-miner binary itself is small (~50-100 MB). The big space usage is only temporary for the build tools.
EOM
    exit 1
fi

echo "CUDA Toolkit found: $(nvcc --version | tail -1)"

# Check minimum version (we need CUDA 12+ for C++17/CUDA17)
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "0.0")
if [[ $(echo "$CUDA_VER < 12.0" | bc -l 2>/dev/null || echo 1) -eq 1 ]]; then
    cat <<'EOM'
ERROR: Your installed CUDA ($CUDA_VER) is too old.

This miner requires CUDA 12.0 or newer (for modern C++17 / CUDA17 support and good performance).

On Hive OS or Ubuntu, install a recent version using the commands below, then re-run this installer.

Typical for driver 550/580 series (CUDA 12.4 recommended):

  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-compiler-12-4 cuda-cudart-dev-12-4

After install:
  export PATH=/usr/local/cuda/bin:$PATH
  nvcc --version   # should show 12.x

Then re-run the full installer command.
EOM
    exit 1
fi

# Clone / update source — always fetch latest from origin/main (never silently keep stale code)
SRC_DIR="${HOME}/btx-nvidia-miner-src"
GIT_BRANCH="${GIT_BRANCH:-main}"

if [[ -d "$SRC_DIR/.git" ]]; then
    echo "Updating existing source in $SRC_DIR (branch: $GIT_BRANCH)"
    (
        cd "$SRC_DIR"
        git remote set-url origin "$REPO_URL"
        # Discard any local edits on the rig so updates are always clean.
        git fetch --depth 1 origin "$GIT_BRANCH"
        git checkout -B "$GIT_BRANCH" "origin/$GIT_BRANCH"
        git reset --hard "origin/$GIT_BRANCH"
    )
else
    echo "Cloning into $SRC_DIR (branch: $GIT_BRANCH)"
    rm -rf "$SRC_DIR"
    git clone --depth 1 --branch "$GIT_BRANCH" "$REPO_URL" "$SRC_DIR"
fi

cd "$SRC_DIR"
BUILD_COMMIT="$(git rev-parse --short HEAD)"
echo "Building commit: $BUILD_COMMIT — $(git log -1 --format=%s)"

# Build - always clean to avoid stale CMake cache (e.g. old "native" arch)
rm -rf build
mkdir -p build
cd build
echo "Configuring (CUDA enabled)..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBTX_MINER_ENABLE_CUDA=ON \
      -DCUDA_ARCHS="86;89;90" \
      ..

echo "Building (this can take a while on first build)..."
make -j"$(nproc)"

# Install the binary
mkdir -p "$BIN_DIR"
cp -f btx-miner "$BIN_DIR/$BINARY_NAME"
chmod +x "$BIN_DIR/$BINARY_NAME"

# Make sure ~/.local/bin is in PATH for the user
if ! echo "$PATH" | grep -q "$BIN_DIR"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" || true
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile" || true
fi

echo
echo "=== Install complete ==="
echo "Binary installed to: $BIN_DIR/$BINARY_NAME"
echo
echo "Quick pool start example (1% dev fee is automatic):"
echo "  $BIN_DIR/$BINARY_NAME \\"
echo "    --pool $POOL_URL \\"
echo "    --user ${USER_ADDRESS}.${WORKER_NAME} \\"
echo "    --pass x \\"
echo "    --devices all"
echo
echo "Quick solo start example (requires a running btxd with RPC enabled):"
echo "  $BIN_DIR/$BINARY_NAME \\"
echo "    --solo \\"
echo "    --rpc-url http://127.0.0.1:19334 \\"
echo "    --rpc-user btxrpc --rpc-password YOURPASS \\"
echo "    --address ${USER_ADDRESS} \\"
echo "    --devices all"
echo
echo "Run in tmux for persistence:"
echo "  tmux new -d -s btxminer '$BIN_DIR/$BINARY_NAME --pool $POOL_URL --user ${USER_ADDRESS}.${WORKER_NAME} --pass x --devices all'"
echo
echo "To update later: re-run this installer (it force-fetches origin/main)."
echo "Installed version: $("$BIN_DIR/$BINARY_NAME" --version 2>/dev/null || echo unknown)"
echo "Verify pool start also prints: btx-miner v0.2.12"
echo "Built from commit: $BUILD_COMMIT"
echo "Dev fee address (built-in): $DEV_FEE_ADDRESS"
echo

# Optional: print nvidia-smi utilization command the user can watch
echo "Watch GPU usage while mining:"
echo "  watch -n 1 nvidia-smi"
