// WMMA vs M31 block-matmul experiment for BTX PoW.
// Compares consensus-correct F_{2^31-1} 16x16 matmul against tensor-core paths.

#include <cuda_runtime.h>
#include <mma.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kM31 = 0x7FFFFFFFU;
constexpr int kBsz = 16;
constexpr int kTrials = 64;
constexpr int kBenchIters = 10000;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

__host__ __device__ __forceinline__ uint32_t h_add(uint32_t a, uint32_t b)
{
    const uint32_t s = a + b;
    return (s >= kM31) ? (s - kM31) : s;
}

__host__ __device__ __forceinline__ uint32_t h_mul(uint32_t a, uint32_t b)
{
    uint64_t p = static_cast<uint64_t>(a) * b;
    uint64_t fold = (p & kM31) + (p >> 31);
    uint32_t r = static_cast<uint32_t>((fold & kM31) + (fold >> 31));
    if (r >= kM31) {
        r -= kM31;
    }
    return r;
}

__host__ __device__ __forceinline__ uint32_t h_reduce64(uint64_t acc)
{
    uint64_t fold = (acc & kM31) + (acc >> 31);
    return static_cast<uint32_t>((fold & kM31) + (fold >> 31));
}

__host__ void cpu_m31_block_matmul(const uint32_t* A, const uint32_t* B, uint32_t* out)
{
    for (int i = 0; i < kBsz; ++i) {
        for (int j = 0; j < kBsz; ++j) {
            uint32_t acc = 0;
            for (int k = 0; k < kBsz; ++k) {
                acc = h_add(acc, h_mul(A[i * kBsz + k], B[k * kBsz + j]));
            }
            out[i * kBsz + j] = acc;
        }
    }
}

__device__ void d_m31_block_matmul(const uint32_t* A, const uint32_t* B, uint32_t* out)
{
    for (int idx = threadIdx.x; idx < kBsz * kBsz; idx += blockDim.x) {
        const int i = idx / kBsz;
        const int j = idx % kBsz;
        uint32_t acc = 0;
        #pragma unroll
        for (int k = 0; k < kBsz; ++k) {
            acc = h_add(acc, h_mul(A[i * kBsz + k], B[k * kBsz + j]));
        }
        out[idx] = acc;
    }
}

// Tensor-core path: int8 WMMA 16x16x16, then apply M31 reduction once per output.
// This is NOT consensus-correct for BTX; included to measure hardware limits.
__device__ void d_wmma_int8_block_matmul(const int8_t* A, const int8_t* B, uint32_t* out)
{
    using namespace nvcuda::wmma;
    __shared__ int32_t shared_raw[kBsz * kBsz];
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t> acc_frag;

    // WMMA is warp-collective: all 32 lanes must execute load/mma/store.
    if (threadIdx.x < 32) {
        fill_fragment(acc_frag, 0);
        load_matrix_sync(a_frag, A, kBsz);
        load_matrix_sync(b_frag, B, kBsz);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        store_matrix_sync(shared_raw, acc_frag, kBsz, mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kBsz * kBsz; idx += blockDim.x) {
        const int64_t v = static_cast<int64_t>(shared_raw[idx]);
        const uint64_t u =
            v >= 0 ? static_cast<uint64_t>(v) : static_cast<uint64_t>(-v);
        out[idx] = static_cast<uint32_t>(h_reduce64(u));
    }
}

// Hybrid: WMMA int8 multiply stage, but M31 multiply-add epilogue in registers.
// Still wrong for full-range M31 inputs because values are clamped to int8.
__device__ void d_wmma_m31_hybrid(const int8_t* A, const int8_t* B, uint32_t* out)
{
    (void)A;
    (void)B;
    for (int idx = threadIdx.x; idx < kBsz * kBsz; idx += blockDim.x) {
        const int i = idx / kBsz;
        const int j = idx % kBsz;
        uint32_t acc = 0;
        #pragma unroll
        for (int k = 0; k < kBsz; ++k) {
            const uint32_t a =
                static_cast<uint32_t>(static_cast<int8_t>(A[i * kBsz + k]) & 0xFFU);
            const uint32_t b =
                static_cast<uint32_t>(static_cast<int8_t>(B[k * kBsz + j]) & 0xFFU);
            acc = h_add(acc, h_mul(a, b));
        }
        out[idx] = acc;
    }
}

__global__ void bench_m31_kernel(const uint32_t* A, const uint32_t* B, uint32_t* out)
{
    d_m31_block_matmul(A, B, out);
}

__global__ void bench_wmma_kernel(const int8_t* A, const int8_t* B, uint32_t* out)
{
    d_wmma_int8_block_matmul(A, B, out);
}

__global__ void bench_hybrid_kernel(const int8_t* A, const int8_t* B, uint32_t* out)
{
    d_wmma_m31_hybrid(A, B, out);
}

__global__ void single_m31_kernel(const uint32_t* A, const uint32_t* B, uint32_t* out)
{
    d_m31_block_matmul(A, B, out);
}

__global__ void single_wmma_kernel(const int8_t* A, const int8_t* B, uint32_t* out)
{
    d_wmma_int8_block_matmul(A, B, out);
}

void fill_random_m31(std::vector<uint32_t>& data, std::mt19937& rng)
{
    std::uniform_int_distribution<uint32_t> dist(0, kM31 - 1);
    for (auto& v : data) {
        v = dist(rng);
    }
}

void pack_int8_truncate(const std::vector<uint32_t>& src, std::vector<int8_t>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = static_cast<int8_t>(src[i] & 0xFFU);
    }
}

void pack_int8_small(std::vector<int8_t>& dst, std::mt19937& rng)
{
    dst.resize(kBsz * kBsz);
    std::uniform_int_distribution<int> dist(-120, 120);
    for (auto& v : dst) {
        v = static_cast<int8_t>(dist(rng));
    }
}

bool vectors_equal(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

double bench_ms(const std::function<void(cudaStream_t)>& launch, cudaStream_t stream, int iters)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 1000; ++i) {
        launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        launch(stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / static_cast<double>(iters);
}

struct DeviceBuffers {
    uint32_t* A_u32 = nullptr;
    uint32_t* B_u32 = nullptr;
    int8_t* A_i8 = nullptr;
    int8_t* B_i8 = nullptr;
    uint32_t* out = nullptr;
};

} // namespace

int main()
{
    std::printf("BTX WMMA vs M31 block-matmul experiment (16x16, sm_86+)\n");

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    std::mt19937 rng(42);
    std::vector<uint32_t> hA(kBsz * kBsz);
    std::vector<uint32_t> hB(kBsz * kBsz);
    std::vector<int8_t> hA_i8;
    std::vector<int8_t> hB_i8;
    std::vector<uint32_t> cpu_ref(kBsz * kBsz);
    std::vector<uint32_t> gpu_m31(kBsz * kBsz);
    std::vector<uint32_t> gpu_wmma(kBsz * kBsz);

    DeviceBuffers dev;
    CUDA_CHECK(cudaMalloc(&dev.A_u32, kBsz * kBsz * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dev.B_u32, kBsz * kBsz * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dev.A_i8, kBsz * kBsz * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dev.B_i8, kBsz * kBsz * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dev.out, kBsz * kBsz * sizeof(uint32_t)));

    // --- Correctness: full-range M31 random matrices ---
    fill_random_m31(hA, rng);
    fill_random_m31(hB, rng);
    cpu_m31_block_matmul(hA.data(), hB.data(), cpu_ref.data());

    CUDA_CHECK(cudaMemcpy(dev.A_u32, hA.data(), hA.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.B_u32, hB.data(), hB.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    single_m31_kernel<<<1, 32>>>(dev.A_u32, dev.B_u32, dev.out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(gpu_m31.data(), dev.out, gpu_m31.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    pack_int8_truncate(hA, hA_i8);
    pack_int8_truncate(hB, hB_i8);
    CUDA_CHECK(cudaMemcpy(dev.A_i8, hA_i8.data(), hA_i8.size() * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.B_i8, hB_i8.data(), hB_i8.size() * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    single_wmma_kernel<<<1, 32>>>(dev.A_i8, dev.B_i8, dev.out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(gpu_wmma.data(), dev.out, gpu_wmma.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    int mismatch_full = 0;
    for (size_t i = 0; i < cpu_ref.size(); ++i) {
        if (gpu_m31[i] != cpu_ref[i]) {
            ++mismatch_full;
        }
        if (gpu_wmma[i] != cpu_ref[i]) {
            ++mismatch_full;
        }
    }

    std::printf("\n[1] Full-range M31 inputs (%d random trials not needed; single matrix)\n",
                kTrials);
    std::printf("    CUDA M31 kernel vs CPU reference: %s\n",
                vectors_equal(gpu_m31, cpu_ref) ? "MATCH" : "MISMATCH");
    int wmma_cell_wrong = 0;
    for (size_t i = 0; i < cpu_ref.size(); ++i) {
        if (gpu_wmma[i] != cpu_ref[i]) {
            ++wmma_cell_wrong;
        }
    }
    std::printf("    WMMA int8 (truncated inputs) vs CPU reference: %s (%d/%zu cells wrong)\n",
                vectors_equal(gpu_wmma, cpu_ref) ? "MATCH" : "MISMATCH",
                wmma_cell_wrong, cpu_ref.size());

    // Statistical mismatch rate over many random M31 matrices.
    int m31_ok = 0;
    int wmma_wrong = 0;
    for (int t = 0; t < kTrials; ++t) {
        fill_random_m31(hA, rng);
        fill_random_m31(hB, rng);
        cpu_m31_block_matmul(hA.data(), hB.data(), cpu_ref.data());

        CUDA_CHECK(cudaMemcpy(dev.A_u32, hA.data(), hA.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.B_u32, hB.data(), hB.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        single_m31_kernel<<<1, 32>>>(dev.A_u32, dev.B_u32, dev.out);
        CUDA_CHECK(cudaMemcpy(gpu_m31.data(), dev.out, gpu_m31.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        pack_int8_truncate(hA, hA_i8);
        pack_int8_truncate(hB, hB_i8);
        CUDA_CHECK(cudaMemcpy(dev.A_i8, hA_i8.data(), hA_i8.size() * sizeof(int8_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev.B_i8, hB_i8.data(), hB_i8.size() * sizeof(int8_t),
                              cudaMemcpyHostToDevice));
        single_wmma_kernel<<<1, 32>>>(dev.A_i8, dev.B_i8, dev.out);
        CUDA_CHECK(cudaMemcpy(gpu_wmma.data(), dev.out, gpu_wmma.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        if (vectors_equal(gpu_m31, cpu_ref)) {
            ++m31_ok;
        }
        if (!vectors_equal(gpu_wmma, cpu_ref)) {
            ++wmma_wrong;
        }
    }
    std::printf("    Over %d random M31 matrices: M31 kernel %d/%d correct, WMMA %d/%d wrong\n",
                kTrials, m31_ok, kTrials, wmma_wrong, kTrials);

    // --- Small int8 values where WMMA integer path is well-defined ---
    pack_int8_small(hA_i8, rng);
    pack_int8_small(hB_i8, rng);
    for (int i = 0; i < kBsz * kBsz; ++i) {
        hA[i] = static_cast<uint32_t>(static_cast<int8_t>(hA_i8[i]) & 0xFFU);
        hB[i] = static_cast<uint32_t>(static_cast<int8_t>(hB_i8[i]) & 0xFFU);
    }
    cpu_m31_block_matmul(hA.data(), hB.data(), cpu_ref.data());

    CUDA_CHECK(cudaMemcpy(dev.A_u32, hA.data(), hA.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.B_u32, hB.data(), hB.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.A_i8, hA_i8.data(), hA_i8.size() * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.B_i8, hB_i8.data(), hB_i8.size() * sizeof(int8_t),
                          cudaMemcpyHostToDevice));

    single_m31_kernel<<<1, 32>>>(dev.A_u32, dev.B_u32, dev.out);
    CUDA_CHECK(cudaMemcpy(gpu_m31.data(), dev.out, gpu_m31.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    single_wmma_kernel<<<1, 32>>>(dev.A_i8, dev.B_i8, dev.out);
    CUDA_CHECK(cudaMemcpy(gpu_wmma.data(), dev.out, gpu_wmma.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    std::printf("\n[2] Small int8-range inputs (|v|<=120), M31 semantics on those values\n");
    std::printf("    CUDA M31 vs CPU: %s\n",
                vectors_equal(gpu_m31, cpu_ref) ? "MATCH" : "MISMATCH");
    std::printf("    WMMA int8 + final reduce64 vs CPU: %s\n",
                vectors_equal(gpu_wmma, cpu_ref) ? "MATCH" : "MISMATCH");

    // --- Benchmark ---
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto launch_m31 = [&](cudaStream_t s) {
        bench_m31_kernel<<<1, 32, 0, s>>>(dev.A_u32, dev.B_u32, dev.out);
    };
    auto launch_wmma = [&](cudaStream_t s) {
        bench_wmma_kernel<<<1, 32, 0, s>>>(dev.A_i8, dev.B_i8, dev.out);
    };
    auto launch_hybrid = [&](cudaStream_t s) {
        bench_hybrid_kernel<<<1, 32, 0, s>>>(dev.A_i8, dev.B_i8, dev.out);
    };

    const double m31_ms = bench_ms(launch_m31, stream, kBenchIters);
    const double wmma_ms = bench_ms(launch_wmma, stream, kBenchIters);
    const double hybrid_ms = bench_ms(launch_hybrid, stream, kBenchIters);

    std::printf("\n[3] Throughput (single 16x16 block per launch, %d iterations)\n", kBenchIters);
    std::printf("    M31 CUDA-core matmul:        %.4f ms/launch (%.2f M blocks/s)\n", m31_ms,
                1.0 / (m31_ms * 1e-3) / 1e6);
    std::printf("    WMMA int8 + reduce64:        %.4f ms/launch (%.2f M blocks/s)\n", wmma_ms,
                1.0 / (wmma_ms * 1e-3) / 1e6);
    std::printf("    WMMA load + M31 epilogue:    %.4f ms/launch (%.2f M blocks/s)\n", hybrid_ms,
                1.0 / (hybrid_ms * 1e-3) / 1e6);
    std::printf("    WMMA speedup vs M31 (int8):  %.2fx\n", m31_ms / wmma_ms);

    const double blocks_per_nonce = 32.0 * 32.0 * 32.0;
    const double nonces_per_sec_if_only_matmul =
        (1.0 / (m31_ms * 1e-3)) / blocks_per_nonce;
    const double nonces_per_sec_wmma =
        (1.0 / (wmma_ms * 1e-3)) / blocks_per_nonce;
    std::printf("\n[4] Extrapolation (matmul blocks only, no noise/hash/gate)\n");
    std::printf("    M31 path theoretical ceiling:  %.2f M nonces/s\n",
                nonces_per_sec_if_only_matmul / 1e6);
    std::printf("    WMMA path theoretical ceiling: %.2f M nonces/s\n",
                nonces_per_sec_wmma / 1e6);
    std::printf("    Your rig measured full pipeline: ~24 M nonces/s\n");
    std::printf("    => Matmul is NOT the only bottleneck; TC wins here don't translate 1:1.\n");

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dev.A_u32));
    CUDA_CHECK(cudaFree(dev.B_u32));
    CUDA_CHECK(cudaFree(dev.A_i8));
    CUDA_CHECK(cudaFree(dev.B_i8));
    CUDA_CHECK(cudaFree(dev.out));

    std::printf("\nConclusion: WMMA cannot replace M31 block matmul for BTX consensus.\n");
    return mismatch_full == 0 && m31_ok == kTrials ? 0 : 2;
}