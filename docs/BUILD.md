# Building btx-nvidia-miner

## CPU-only (reference + scaffolding)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBTX_MINER_ENABLE_CUDA=OFF ..
make -j$(nproc)
./btx-miner --benchmark
```

This exercises the self-contained MatMul PoW reference (field arithmetic over F_{2^31-1}, FromSeed, low-rank noise, blocked transcript accumulation + compression, sigma derivation from header fields, Solve/Verify).

## With CUDA (real mining)

You need the CUDA Toolkit (nvcc) matching your driver.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBTX_MINER_ENABLE_CUDA=ON \
      -DCUDA_ARCHS="native" \          # or release list in cmake/cuda_architectures.cmake
      ..
make -j$(nproc)
```

The binary will contain the optimized kernels when a GPU is present at runtime.

If you only want to compile kernels without a GPU at build time (e.g. on CI or a build server), the above still works — `nvcc` does not require a physical GPU to produce device code.

## Developing without an NVIDIA GPU (ZLUDA)

1. Install ZLUDA (https://github.com/vosen/ZLUDA).
2. Build the miner normally with CUDA enabled (on a machine that has the CUDA Toolkit but no NVIDIA card, or cross-compile).
3. Run with ZLUDA's preload:

   ```bash
   LD_LIBRARY_PATH=/path/to/zluda/lib ./btx-miner --benchmark --devices 0
   ```

   or the equivalent `ZLUDA` wrapper the project provides.

ZLUDA lets you validate kernel correctness and even do some perf tuning on AMD/Intel GPUs. Real production hashrates will only appear on native NVIDIA.

## GitHub Actions / CI

The repository includes example workflows that:
- Always build the CPU reference (for unit tests and PRs).
- Build the full CUDA binary in a `nvidia/cuda` Docker image (or on self-hosted GPU runners) when a release tag is pushed or on manual dispatch.

See `.github/workflows/`.

## Runtime requirements (when using CUDA)

- NVIDIA driver that supports your CUDA Toolkit version.
- The binary is dynamically linked against the CUDA runtime that was present at build time (or you can build with static cudart if desired).

## Troubleshooting

- "nvcc not found" → install CUDA Toolkit or set `CUDAToolkit_ROOT`.
- Very low hashrate / 0% GPU util → the CPU-side job prep is the bottleneck. Increase prepare workers / batch (the same knobs the reference `btx-gbt-solve` and dexbtx-miner use).
- Share rejections on pool → almost always a sigma / header-hash / bits / nonce64_start handling bug. The CPU reference `VerifySolution` is the oracle; any submitted share must also pass it.

## Next steps for kernel development

The high-leverage work is in `src/cuda/matmul_kernel.cu` and `src/cuda/cuda_solver.cpp`:

- Batched nonce trials (grid of nonces or persistent threads).
- On-device from_oracle (SHA256 PRF + rejection) for seeds and noise.
- Efficient 16×16 (or larger tile) blocked matmul with running accumulation into the transcript compressor.
- On-device SHA256 state updates for the transcript (or fused compression that feeds a few u32 per block product).
- Multi-stream / multi-device scheduler that keeps all GPUs busy while the host prepares the next job (new template or stratum notify).

The 512×512 size is tiny; the win is massive batching + minimal host<->device traffic per batch of nonces.
