#pragma once

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace gpasha {

using Element = uint32_t;
constexpr Element MODULUS = 0x7FFFFFFFU;

void DeriveNoiseSeedsKernel_launch(
    const uint8_t* sigma_batch,
    uint8_t* noise_seeds,
    uint8_t* compress_seeds,
    uint32_t* noise_midstates,
    uint32_t* compress_midstates,
    uint32_t batch_size,
    cudaStream_t stream = 0);

void PrecomputeSeedMidstatesKernel_launch(
    const uint8_t* seeds,
    uint32_t* seed_midstates,
    uint32_t seed_count,
    cudaStream_t stream = 0);

void GenerateAllNoiseKernel_launch(
    const uint8_t* noise_seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t noise_left_elements,
    uint32_t noise_right_elements,
    Element* output_el,
    Element* output_er,
    Element* output_fl,
    Element* output_fr,
    cudaStream_t stream = 0);

void GenerateNoiseKernel_launch(
    const uint8_t* noise_seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t num_elements,
    uint32_t seed_index,
    Element* output,
    cudaStream_t stream = 0);

void GenerateCompressKernel_launch(
    const uint8_t* compress_seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t num_elements,
    Element* output,
    cudaStream_t stream = 0);

void HashTranscriptCompareKernel_launch(
    const Element* compressed_words,
    const uint8_t* sigma_batch,
    uint32_t words_per_nonce,
    uint32_t n,
    uint32_t b,
    uint32_t batch_size,
    const uint8_t* block_target,
    const uint8_t* share_target,
    uint8_t* digest_batch,
    int32_t* results,
    cudaStream_t stream = 0);

void HashTranscriptKernel_launch(
    const Element* compressed_words,
    const uint8_t* sigma_batch,
    uint32_t words_per_nonce,
    uint32_t n,
    uint32_t b,
    uint32_t batch_size,
    uint8_t* digest_batch,
    cudaStream_t stream = 0);

void CompareDigestsKernel_launch(
    const uint8_t* digest_batch,
    const uint8_t* block_target,
    const uint8_t* share_target,
    uint32_t batch_size,
    int32_t* results,
    cudaStream_t stream = 0);

void GenerateMatrixKernel_launch(
    const uint8_t* seeds,
    const uint32_t* seed_midstates,
    uint32_t batch_size,
    uint32_t n,
    Element* output,
    cudaStream_t stream = 0);

} // namespace gpasha