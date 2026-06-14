# btx-nvidia-miner

High-performance NVIDIA CUDA miner for BTX (btxchain/btx) MatMul Proof-of-Work.

**Optimized for matrix multiplication (512×512 over F_{2^31-1})**. GPU-first design with excellent multi-GPU scaling. Supports both solo mining (JSON-RPC against a local `btxd`) and pool mining against `stratum+tcp://stratum.minebtx.com:3333`.

## Features

- Native CUDA kernels for the full MatMul PoW (seed expansion, low-rank noise, blocked transcript accumulation + compression, SHA-256 transcript digest).
- Multi-GPU support with automatic device discovery and load distribution.
- Solo mining via `getblocktemplate` / `submitblock`.
- Stratum client compatible with the minebtx.com pool (stratum/2.0-matmul protocol).
- CPU reference implementation for verification and environments without CUDA.
- Low CPU overhead — designed to keep GPUs at >95% utilization.

## Requirements

- NVIDIA GPU (Pascal or newer recommended; sm_61+)
- CUDA Toolkit 12.0+ (nvcc)
- CMake 3.20+
- A C++17 compiler (GCC/Clang/MSVC)
- For pool: a BTX `btx1z...` P2MR address
- For solo: a synced `btxd` with `server=1` and RPC credentials

**Development without an NVIDIA GPU**: The CUDA kernels compile with the CUDA Toolkit even without hardware. For runtime testing of kernels you can use [ZLUDA](https://github.com/vosen/ZLUDA) (CUDA → Vulkan translation) on AMD/Intel GPUs or for validation. Actual performance numbers and CI runtime tests require real NVIDIA hardware or a self-hosted runner.

## Quick Build (Linux)

```bash
git clone https://github.com/<you>/btx-nvidia-miner.git
cd btx-nvidia-miner
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="native" ..
make -j$(nproc)
```

See `docs/build.md` for full options (static CUDA, specific arches, ZLUDA dev notes, Docker).

Pre-built binaries (when available) are published on the Releases page.

## Usage

A 1% dev fee is built into the miner for both solo and pool mining. It goes to:
`btx1z0069dewdztkwnrxx97lt9c5paynh0nynegqxq2kgykh0ct8xaggq0953gx`

You can override the percentage with `--dev-fee 0.5` or the `BTX_DEV_FEE_PCT` environment variable (0–5).

### Solo mining (local node)

```bash
./btx-miner \
  --solo \
  --rpc-url http://127.0.0.1:19334 \
  --rpc-user btxrpc --rpc-password yourpassword \
  --address btx1zYourPayoutAddressHere... \
  --devices 0,1
```

### Pool mining (minebtx)

```bash
./btx-miner \
  --pool stratum+tcp://stratum.minebtx.com:3333 \
  --user btx1zYourPayoutAddressHere.worker1 \
  --pass x \
  --devices all
```

The installer (see below) prints ready-to-run commands with your address and the dev fee already applied.

### Common options

- `--devices 0,1,2` or `all`
- `--intensity` / per-device batch & worker tuning (see `--help`)
- `--benchmark` — run a short local MatMul throughput test
- `--no-gpu` — force CPU reference path (slow, for verification)

Run `./btx-miner --help` for the full list.

## Performance Notes

512×512 MatMul is a small, compute-bound kernel on modern GPUs. The dominant win comes from:

- Batching thousands of independent nonce trials (each with its own sigma-derived noise).
- On-device everything: noise generation, perturbed matmul, running-block compression, and per-nonce SHA state.
- Overlapping H2D/D2H with kernel execution via streams.
- One high-occupancy kernel or a small pipeline of kernels per device rather than many small launches.

The reference node CUDA backend is already very good. This miner aims to be a lean, direct, multi-GPU-first alternative with fewer host-side staging steps.

Typical targets (very rough, real numbers depend on exact silicon + tuning):
- RTX 40/50-series: high hundreds to low thousands of full 512³ MatMuls per second per GPU (i.e. nonces/s).

Use `--benchmark` and the live dashboard (for pool) + `nvidia-smi` to tune.

## Project Structure

```
src/
  pow/                 # Self-contained CPU reference (field, matrix, noise, transcript, sigma, solve/verify)
  cuda/                # .cu kernels + host launchers, multi-GPU scheduler
  solo/                # getblocktemplate + submitblock client
  stratum/             # Stratum/2.0-matmul client + job management
  common/              # CLI, logging, work dispatcher, metrics
```

The CPU reference in `pow/` is bit-for-bit compatible with `btxchain/btx` `src/matmul/*` (M31 field, from_oracle PRF, running-block transcript compression, header sigma derivation, etc.).

## Verification & Correctness

- The CPU path is the source of truth for a given job.
- On startup (or with `--self-test`) the miner runs known-answer tests and cross-checks a few thousand random nonces between CPU reference and CUDA path (when a GPU is present).
- Pool shares and solo blocks are only submitted when the CPU reference also confirms the digest.

## License

MIT (see LICENSE). The MatMul PoW algorithm and protocol are defined by the BTX project (https://github.com/btxchain/btx).

## Acknowledgments

- BTX core developers for the MatMul PoW design, spec, reference node CUDA backend, and the public pool.
- The authors of "Proofs of Useful Work from Arbitrary Matrix Multiplication" (arXiv:2504.09971).
- Community miners (dexbtx, easyBTX, etc.) that proved out the stratum surface.

## Status

Core CPU reference PoW is solid and self-contained (matches the BTX node algorithm for sigma/noise/transcript).

CUDA kernels + full stratum/solo + dev fee application are the current focus (following the ordered plan in the code).

The `install.sh` makes it straightforward to get a working binary on a remote headless rig.

### Easy one-liner install on a remote Linux + NVIDIA rig

```bash
curl -fsSL https://raw.githubusercontent.com/thekillsquad007/btx-nvidia-miner/main/install.sh | bash -s -- --address btx1zYOURADDRESS
```

- Downloads the latest prebuilt release binary (NVIDIA driver + CUDA runtime only — no nvcc).
- Installs to `~/.local/bin/btx-miner` and prints ready-to-run commands (dev fee is automatic).
- If a CDN serves a stale installer, pin the release: add `--version 0.2.32` to the command above.
- Run the resulting binary in tmux/screen or as a service.

See `install.sh` and `docs/BUILD.md` for details and options.

A 1% dev fee (configurable via --dev-fee or BTX_DEV_FEE_PCT) to the address above is wired for both mining modes.
