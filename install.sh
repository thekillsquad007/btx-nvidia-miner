#!/usr/bin/env bash
#
# Easy installer for btx-nvidia-miner on a remote Linux machine with NVIDIA GPU.
#
# Usage (recommended one-liner):
#   curl -fsSL https://raw.githubusercontent.com/thekillsquad007/btx-nvidia-miner/main/install.sh | bash -s -- --address btx1zYOURADDRESS...
#
# Downloads a prebuilt CUDA binary (sm_86–sm_120) from GitHub Releases — no compile
# step on the rig. Only NVIDIA driver + CUDA runtime libraries are required.
#
# Optional: --build-from-source to compile locally (needs nvcc + ~2GB build deps).
#
set -euo pipefail

DEV_FEE_ADDRESS="btx1z0069dewdztkwnrxx97lt9c5paynh0nynegqxq2kgykh0ct8xaggq0953gx"
DEFAULT_DEV_FEE_PCT="1.0"

REPO_OWNER="thekillsquad007"
REPO_NAME="btx-nvidia-miner"
REPO_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"
BINARY_NAME="btx-miner"
# Pin to the release that ships prebuilt binaries. Bump when publishing a new release.
RELEASE_VERSION="${BTX_MINER_VERSION:-0.2.30}"
RELEASE_TAG="v${RELEASE_VERSION}"
RELEASE_ASSET="btx-miner-linux-x86_64.tar.gz"
RELEASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}/${RELEASE_ASSET}"

INSTALL_PREFIX="${HOME}/.local"
BIN_DIR="${INSTALL_PREFIX}/bin"

USER_ADDRESS=""
POOL_URL="stratum+tcp://stratum.minebtx.com:3333"
WORKER_NAME=""
BUILD_FROM_SOURCE=0

print_help() {
    cat <<EOF
btx-nvidia-miner installer

Options:
  --address ADDR          Your BTX payout address (btx1z...). Required for examples.
  --pool URL              Pool to use in the printed example (default: $POOL_URL)
  --worker NAME           Worker name suffix (default: hostname)
  --prefix DIR            Install prefix (default: $INSTALL_PREFIX)
  --version VER           Release to install (default: $RELEASE_VERSION)
  --build-from-source     Compile on this machine instead of using the release binary
  -h, --help              This help

Default install downloads a prebuilt binary targeting NVIDIA sm_86, sm_89, sm_90,
and sm_120 (RTX 30xx / 40xx / 50xx). You only need the NVIDIA driver and CUDA
runtime — not the full CUDA Toolkit.

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
        --version) RELEASE_VERSION="$2"; RELEASE_TAG="v${RELEASE_VERSION}"; shift 2 ;;
        --build-from-source) BUILD_FROM_SOURCE=1; shift ;;
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

RELEASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}/${RELEASE_ASSET}"

echo "=== btx-nvidia-miner installer ==="
echo "User address : $USER_ADDRESS"
echo "Dev fee addr : $DEV_FEE_ADDRESS (${DEFAULT_DEV_FEE_PCT}%)"
echo "Pool example : $POOL_URL"
echo "Worker       : ${USER_ADDRESS}.${WORKER_NAME}"
echo "Release      : ${RELEASE_TAG} (${BUILD_FROM_SOURCE:+build from source}${BUILD_FROM_SOURCE:-prebuilt binary})"
echo

# Check for NVIDIA driver
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers before mining."
    exit 1
fi

echo "GPU(s) detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || true
echo

mkdir -p "$BIN_DIR"

install_prebuilt() {
    echo "Downloading prebuilt ${BINARY_NAME} ${RELEASE_TAG}..."
    echo "  ${RELEASE_URL}"
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT

    if ! curl -fsSL -o "${TMP_DIR}/${RELEASE_ASSET}" "${RELEASE_URL}"; then
        echo "ERROR: failed to download release asset."
        echo "Check that ${RELEASE_TAG} exists at:"
        echo "  https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/tag/${RELEASE_TAG}"
        exit 1
    fi

    tar -xzf "${TMP_DIR}/${RELEASE_ASSET}" -C "${TMP_DIR}"
    if [[ ! -f "${TMP_DIR}/${BINARY_NAME}" ]]; then
        echo "ERROR: archive did not contain ${BINARY_NAME}"
        exit 1
    fi

    install -m 0755 "${TMP_DIR}/${BINARY_NAME}" "${BIN_DIR}/${BINARY_NAME}"
    echo "Installed prebuilt binary to ${BIN_DIR}/${BINARY_NAME}"
}

install_from_source() {
    echo "=== Building from source (requires CUDA Toolkit / nvcc) ==="

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

    if ! command -v nvcc >/dev/null 2>&1; then
        cat <<'EOM'
ERROR: nvcc (CUDA Toolkit) not found in PATH.

Use the default installer (no --build-from-source) to download a prebuilt binary,
or install the CUDA compiler on this rig and re-run with --build-from-source.

  export PATH=/usr/local/cuda/bin:$PATH
  nvcc --version
EOM
        exit 1
    fi

    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export PATH="/usr/local/cuda/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    fi

    echo "CUDA Toolkit found: $(nvcc --version | tail -1)"

    SRC_DIR="${HOME}/btx-nvidia-miner-src"
    GIT_BRANCH="${GIT_BRANCH:-main}"

    if [[ -d "$SRC_DIR/.git" ]]; then
        echo "Updating existing source in $SRC_DIR (branch: $GIT_BRANCH)"
        (
            cd "$SRC_DIR"
            git remote set-url origin "$REPO_URL"
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

    CUDA_ARCH_LIST=""
    while IFS= read -r _cap; do
        _cap="${_cap//[[:space:]]/}"
        [[ -z "$_cap" ]] && continue
        _sm="${_cap//./}"
        if [[ -n "$CUDA_ARCH_LIST" ]]; then
            CUDA_ARCH_LIST="${CUDA_ARCH_LIST};${_sm}"
        else
            CUDA_ARCH_LIST="${_sm}"
        fi
    done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u)
    if [[ -z "$CUDA_ARCH_LIST" ]]; then
        CUDA_ARCH_LIST="86;89;90;120"
        echo "WARNING: could not read GPU compute caps; defaulting to ${CUDA_ARCH_LIST}"
    fi
    echo "CUDA arch list: $CUDA_ARCH_LIST"

    rm -rf build
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBTX_MINER_ENABLE_CUDA=ON \
          -DBTX_MINER_BUILD_TESTS=OFF \
          -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_LIST}" \
          ..
    make -j"$(nproc)"
    install -m 0755 btx-miner "${BIN_DIR}/${BINARY_NAME}"
    echo "Built from commit: $BUILD_COMMIT"
}

if [[ "$BUILD_FROM_SOURCE" -eq 1 ]]; then
    install_from_source
else
    install_prebuilt
fi

# Runtime: ensure CUDA user-space libs are visible (driver usually provides these)
if [[ -d /usr/local/cuda/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

if ! echo "$PATH" | grep -q "$BIN_DIR"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc" || true
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile" || true
fi

# Quick sanity check
if ! "${BIN_DIR}/${BINARY_NAME}" --version >/dev/null 2>&1; then
    echo "ERROR: installed binary failed to run. Missing CUDA runtime?"
    echo "Try: ldd ${BIN_DIR}/${BINARY_NAME}"
    exit 1
fi

echo
echo "=== Install complete ==="
echo "Binary installed to: $BIN_DIR/$BINARY_NAME"
echo "Version: $("$BIN_DIR/$BINARY_NAME" --version 2>/dev/null || echo unknown)"
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
echo "To update: re-run this installer (fetches latest release ${RELEASE_TAG} by default)."
echo "Dev fee address (built-in): $DEV_FEE_ADDRESS"
echo
echo "Watch GPU usage while mining:"
echo "  watch -n 1 nvidia-smi"