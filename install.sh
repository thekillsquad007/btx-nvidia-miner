#!/usr/bin/env bash
#
# Easy installer for btx-nvidia-miner on a remote Linux machine with NVIDIA GPU.
#
# Usage (recommended one-liner):
#   curl -fsSL https://raw.githubusercontent.com/thekillsquad007/btx-nvidia-miner/main/install.sh | bash -s -- --address btx1zYOURADDRESS...
#
# Downloads a prebuilt CUDA binary (Pascal sm_60+ through Blackwell sm_120) — no compile
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
RELEASE_VERSION="${BTX_MINER_VERSION:-0.2.42}"
RELEASE_TAG="v${RELEASE_VERSION}"
RELEASE_ASSET="btx-miner-linux-x86_64.tar.gz"
RELEASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}/${RELEASE_ASSET}"

# User-writable default — no sudo or chmod tinkering required.
INSTALL_PREFIX="${HOME}/.local"
BIN_DIR="${INSTALL_PREFIX}/bin"
STATE_DIR="${INSTALL_PREFIX}/share/btx-nvidia-miner"

USER_ADDRESS=""
POOL_URL="stratum+tcp://stratum.bitminerpool.xyz:3333"
# Optional backup pool (auto-failover in btx-miner). Set to "none" to disable.
POOL_FALLBACK_URL="none"
WORKER_NAME=""
BUILD_FROM_SOURCE=0
UNINSTALL_ONLY=0
KEEP_SOURCE=0

print_help() {
    cat <<EOF
btx-nvidia-miner installer

Options:
  --address ADDR          Your BTX payout address (btx1z...). Required for examples.
  --pool URL              Primary pool in the printed example (default: $POOL_URL)
  --pool-fallback URL     Backup pool for failover (default: $POOL_FALLBACK_URL)
  --worker NAME           Worker name suffix (default: hostname)
  --prefix DIR            Install prefix (default: \$HOME/.local)
  --version VER           Release to install (default: $RELEASE_VERSION)
  --build-from-source     Compile on this machine instead of using the release binary
  --keep-source           Do not delete old source trees during uninstall
  --uninstall-only        Stop miner and remove previous install, then exit
  -h, --help              This help

Default install downloads a prebuilt fat binary for Pascal through Blackwell
(sm_60, sm_61, sm_70, sm_72, sm_75, sm_80, sm_86, sm_87, sm_89, sm_90, sm_100,
sm_120). Covers GTX 10xx, RTX 20xx–50xx, CMP 170HX, A100/H100, etc. You only need
the NVIDIA driver and CUDA 12 runtime — not the full CUDA Toolkit.

Each install removes any previous btx-miner binary and stops running miners first.

Dev fee of ${DEFAULT_DEV_FEE_PCT}% is built-in and goes to:
  ${DEV_FEE_ADDRESS}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --address) USER_ADDRESS="$2"; shift 2 ;;
        --pool)    POOL_URL="$2"; shift 2 ;;
        --pool-fallback) POOL_FALLBACK_URL="$2"; shift 2 ;;
        --worker)  WORKER_NAME="$2"; shift 2 ;;
        --prefix)  INSTALL_PREFIX="$2"; BIN_DIR="${INSTALL_PREFIX}/bin"; STATE_DIR="${INSTALL_PREFIX}/share/btx-nvidia-miner"; shift 2 ;;
        --version) RELEASE_VERSION="$2"; RELEASE_TAG="v${RELEASE_VERSION}"; shift 2 ;;
        --build-from-source) BUILD_FROM_SOURCE=1; shift ;;
        --keep-source) KEEP_SOURCE=1; shift ;;
        --uninstall-only) UNINSTALL_ONLY=1; shift ;;
        -h|--help) print_help; exit 0 ;;
        *) echo "Unknown arg: $1"; print_help; exit 1 ;;
    esac
done

if [[ "$UNINSTALL_ONLY" -eq 0 && -z "$USER_ADDRESS" ]]; then
    echo "ERROR: --address is required (your btx1z... payout address)"
    print_help
    exit 1
fi

if [[ -z "$WORKER_NAME" ]]; then
    WORKER_NAME="$(hostname -s 2>/dev/null || echo rig)"
fi

RELEASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${RELEASE_TAG}/${RELEASE_ASSET}"

# Candidate install locations from older runs / manual copies.
collect_install_paths() {
    local paths=()
    local seen="|"
    local p

    add_path() {
        local candidate="$1"
        [[ -z "$candidate" ]] && return
        case "$seen" in
            *"|${candidate}|"*) return ;;
        esac
        seen="${seen}${candidate}|"
        paths+=("$candidate")
    }

    add_path "${BIN_DIR}/${BINARY_NAME}"
    add_path "${HOME}/.local/bin/${BINARY_NAME}"
    add_path "${HOME}/bin/${BINARY_NAME}"
    add_path "/usr/local/bin/${BINARY_NAME}"

    if command -v "${BINARY_NAME}" >/dev/null 2>&1; then
        p="$(command -v "${BINARY_NAME}")"
        add_path "$p"
    fi

    while IFS= read -r p; do
        add_path "$p"
    done < <(find "${HOME}" /usr/local/bin -maxdepth 4 -name "${BINARY_NAME}" -type f 2>/dev/null || true)

    printf '%s\n' "${paths[@]}"
}

stop_running_miner() {
    local pids=""
    if command -v pgrep >/dev/null 2>&1; then
        pids="$(pgrep -f "[b]tx-miner" 2>/dev/null || true)"
    fi
    if [[ -z "$pids" ]]; then
        return 0
    fi

    echo "Stopping running btx-miner process(es): $pids"
    kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids="$(pgrep -f "[b]tx-miner" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        kill -KILL $pids 2>/dev/null || true
    fi
}

uninstall_existing() {
    echo "=== Removing previous btx-miner installation ==="
    stop_running_miner

    local removed=0
    local path
    while IFS= read -r path; do
        [[ -z "$path" ]] && continue
        if [[ -f "$path" ]]; then
            rm -f "$path"
            echo "  removed: $path"
            removed=$((removed + 1))
        fi
    done < <(collect_install_paths)

    if [[ "$KEEP_SOURCE" -eq 0 ]]; then
        local src
        for src in \
            "${HOME}/btx-nvidia-miner-src" \
            "${HOME}/btx-nvidia-miner" \
            "${HOME}/btx-nvidia-miner-git" \
            "/root/btx-nvidia-miner" \
            "/root/btx-nvidia-miner-src" \
            "/root/btx-nvidia-miner-git"; do
            if [[ -d "$src" && ! -d "${src}/.git" ]]; then
                rm -rf "$src"
                echo "  removed stale tree: $src"
                removed=$((removed + 1))
            fi
        done
    fi

    if [[ -f "${STATE_DIR}/install.env" ]]; then
        rm -f "${STATE_DIR}/install.env"
        echo "  removed: ${STATE_DIR}/install.env"
        removed=$((removed + 1))
    fi

    if [[ "$removed" -eq 0 ]]; then
        echo "  (no previous binary found)"
    fi
    echo
}

ensure_install_dirs() {
    mkdir -p "$BIN_DIR" "$STATE_DIR"
    # install(1) sets mode 0755; mkdir alone is enough for parent dirs.
    chmod 0755 "$INSTALL_PREFIX" "$BIN_DIR" "$STATE_DIR" 2>/dev/null || true
}

ensure_path_entry() {
    local line="export PATH=\"${BIN_DIR}:\$PATH\""
    local marker="# added by btx-nvidia-miner install.sh"
    local file

    for file in "$HOME/.bashrc" "$HOME/.profile"; do
        [[ -f "$file" ]] || touch "$file"
        if grep -Fq "$marker" "$file" 2>/dev/null; then
            continue
        fi
        if grep -Fq "${BIN_DIR}" "$file" 2>/dev/null; then
            continue
        fi
        {
            echo ""
            echo "$marker"
            echo "$line"
        } >> "$file"
    done
}

write_install_state() {
    cat > "${STATE_DIR}/install.env" <<EOF
BTX_MINER_VERSION=${RELEASE_VERSION}
BTX_MINER_BINARY=${BIN_DIR}/${BINARY_NAME}
BTX_MINER_INSTALLED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BTX_MINER_INSTALL_PREFIX=${INSTALL_PREFIX}
EOF
}

install_prebuilt() {
    echo "Downloading prebuilt ${BINARY_NAME} ${RELEASE_TAG}..."
    echo "  ${RELEASE_URL}"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT

    if ! curl -fsSL -o "${tmp_dir}/${RELEASE_ASSET}" "${RELEASE_URL}"; then
        echo "ERROR: failed to download release asset."
        echo "Check that ${RELEASE_TAG} exists at:"
        echo "  https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/tag/${RELEASE_TAG}"
        exit 1
    fi

    tar -xzf "${tmp_dir}/${RELEASE_ASSET}" -C "${tmp_dir}"
    if [[ ! -f "${tmp_dir}/${BINARY_NAME}" ]]; then
        echo "ERROR: archive did not contain ${BINARY_NAME}"
        exit 1
    fi

    install -m 0755 "${tmp_dir}/${BINARY_NAME}" "${BIN_DIR}/${BINARY_NAME}"
    echo "Installed prebuilt binary to ${BIN_DIR}/${BINARY_NAME}"
}

install_from_source() {
    echo "=== Building from source (requires CUDA Toolkit / nvcc) ==="

    if command -v apt-get >/dev/null 2>&1; then
        if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
            sudo apt-get update -y
            sudo apt-get install -y build-essential cmake git curl pkg-config \
                libcurl4-openssl-dev || true
        else
            echo "NOTE: skipping apt packages (no passwordless sudo). Install build-essential, cmake, git if missing."
        fi
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

    local src_dir="${HOME}/btx-nvidia-miner-src"
    local git_branch="${GIT_BRANCH:-main}"

    if [[ -d "$src_dir/.git" ]]; then
        echo "Updating existing source in $src_dir (branch: $git_branch)"
        (
            cd "$src_dir"
            git remote set-url origin "$REPO_URL"
            git fetch --depth 1 origin "$git_branch"
            git checkout -B "$git_branch" "origin/$git_branch"
            git reset --hard "origin/$git_branch"
        )
    else
        echo "Cloning into $src_dir (branch: $git_branch)"
        rm -rf "$src_dir"
        git clone --depth 1 --branch "$git_branch" "$REPO_URL" "$src_dir"
    fi

    cd "$src_dir"
    local build_commit
    build_commit="$(git rev-parse --short HEAD)"
    echo "Building commit: $build_commit — $(git log -1 --format=%s)"

    local cuda_arch_list=""
    local _cap _sm
    while IFS= read -r _cap; do
        _cap="${_cap//[[:space:]]/}"
        [[ -z "$_cap" ]] && continue
        _sm="${_cap//./}"
        if [[ -n "$cuda_arch_list" ]]; then
            cuda_arch_list="${cuda_arch_list};${_sm}"
        else
            cuda_arch_list="${_sm}"
        fi
    done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u)
    if [[ -z "$cuda_arch_list" ]]; then
        cuda_arch_list="60;61;70;72;75;80;86;87;89;90;100;120"
        echo "WARNING: could not read GPU compute caps; defaulting to release arch list"
    fi
    echo "CUDA arch list: $cuda_arch_list"

    rm -rf build
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBTX_MINER_ENABLE_CUDA=ON \
          -DBTX_MINER_BUILD_TESTS=OFF \
          -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch_list}" \
          ..
    cmake --build . -j"$(nproc)" --target btx-miner
    install -m 0755 btx-miner "${BIN_DIR}/${BINARY_NAME}"
    echo "Built from commit: $build_commit"
}

echo "=== btx-nvidia-miner installer ==="
if [[ -n "$USER_ADDRESS" ]]; then
    echo "User address : $USER_ADDRESS"
fi
echo "Dev fee addr : $DEV_FEE_ADDRESS (${DEFAULT_DEV_FEE_PCT}%)"
echo "Install path : ${BIN_DIR}"
if [[ -n "$USER_ADDRESS" ]]; then
    echo "Pool example : $POOL_URL"
    echo "Worker       : ${USER_ADDRESS}.${WORKER_NAME}"
fi
echo "Release      : ${RELEASE_TAG} ($([ "$BUILD_FROM_SOURCE" -eq 1 ] && echo build-from-source || echo prebuilt-binary))"
echo

uninstall_existing

if [[ "$UNINSTALL_ONLY" -eq 1 ]]; then
    echo "Uninstall complete."
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers before mining."
    exit 1
fi

echo "GPU(s) detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || true
echo

ensure_install_dirs

if [[ "$BUILD_FROM_SOURCE" -eq 1 ]]; then
    install_from_source
else
    install_prebuilt
fi

write_install_state
ensure_path_entry

if [[ -d /usr/local/cuda/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

export PATH="${BIN_DIR}:${PATH}"

if ! "${BIN_DIR}/${BINARY_NAME}" --version >/dev/null 2>&1; then
    echo "ERROR: installed binary failed to run. Missing CUDA runtime?"
    echo "Try: ldd ${BIN_DIR}/${BINARY_NAME}"
    exit 1
fi

echo
echo "=== Install complete ==="
echo "Binary installed to: ${BIN_DIR}/${BINARY_NAME}"
echo "Version: $("${BIN_DIR}/${BINARY_NAME}" --version 2>/dev/null || echo unknown)"
echo "State file: ${STATE_DIR}/install.env"
echo
if "${BIN_DIR}/${BINARY_NAME}" --print-gpu-batch >/dev/null 2>&1; then
    echo "Per-GPU launch batches (auto, omit --batch to use these):"
    "${BIN_DIR}/${BINARY_NAME}" --print-gpu-batch | sed 's/^/  /'
    echo
fi
echo "Quick pool start example (1% dev fee is automatic):"
echo "  ${BIN_DIR}/${BINARY_NAME} \\"
echo "    --pool ${POOL_URL} \\"
echo "    --pool-fallback ${POOL_FALLBACK_URL} \\"
echo "    --user ${USER_ADDRESS}.${WORKER_NAME} \\"
echo "    --pass x \\"
echo "    --devices all"
echo
echo "Set --pool-fallback to add a backup pool for auto-failover after ~60s."
echo "Dashboard: https://bitminerpool.xyz/"
echo
echo "Mixed-GPU rigs can pin per card (active GPU order from --print-gpu-batch):"
echo "  ... --batch 0,262144,65536,65536   # auto on GPU0, larger batch on fast cards"
echo
echo "Quick solo start example (requires a running btxd with RPC enabled):"
echo "  ${BIN_DIR}/${BINARY_NAME} \\"
echo "    --solo \\"
echo "    --rpc-url http://127.0.0.1:19334 \\"
echo "    --rpc-user btxrpc --rpc-password YOURPASS \\"
echo "    --address ${USER_ADDRESS} \\"
echo "    --devices all"
echo
echo "Run in tmux for persistence:"
echo "  tmux new -d -s btxminer '${BIN_DIR}/${BINARY_NAME} --pool ${POOL_URL} --pool-fallback ${POOL_FALLBACK_URL} --user ${USER_ADDRESS}.${WORKER_NAME} --pass x --devices all --auto-update'"
echo
echo "Manual update check: ${BIN_DIR}/${BINARY_NAME} --check-update"
echo "Force update:        ${BIN_DIR}/${BINARY_NAME} --update"
echo
echo "To update: re-run this installer (stops old miner, removes old binary, installs ${RELEASE_TAG})."
echo "To remove only: curl -fsSL .../install.sh | bash -s -- --uninstall-only"
echo "Dev fee address (built-in): ${DEV_FEE_ADDRESS}"
echo
echo "Watch GPU usage while mining:"
echo "  watch -n 1 nvidia-smi"