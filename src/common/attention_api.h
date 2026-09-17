#pragma once
#include <cuda_runtime.h>

// Three-kernel crossover study: dense exact, sliding-window+global, block-sparse.
// All three share the same tiled online-softmax structure and differ only in
// which K/V tiles each query tile streams.
enum class AttnKernel {
    Dense,           // Kernel A — FlashAttention-style exact baseline (HW6/7 carry-forward).
    DenseFP16,       // Kernel A2 — dense with FP16 inputs + Tensor Core (WMMA) matmuls.
    Windowed,        // Kernel B — sliding window of half-width w, plus G global tokens.
    WindowedFP16,    // Kernel B2 — windowed with FP16 inputs + Tensor Core (WMMA) matmuls.
    BlockSparse,     // Kernel C — per-row CSR list of active K/V blocks.
    BlockSparseFP16, // Kernel C2 — block-sparse with FP16 inputs + Tensor Core (WMMA) matmuls.
};

struct AttnConfig {
    int  N;        // sequence length
    int  d;        // head dimension (64 or 128)
    int  H;        // number of heads
    bool causal;
    AttnKernel kernel;

    // Kernel B (Windowed) parameters
    int  w;        // window half-width
    int  G;        // number of global tokens (typically the first G rows)

    // Kernel C (BlockSparse) parameters — CSR over (N/B_block) x (N/B_block) blocks.
    // Pointers are device pointers; ownership stays with the caller.
    const int* block_row_ptr;  // length num_block_rows + 1
    const int* block_col_idx;  // length nnz_blocks
    int        B_block;        // block side (e.g., 64)
    int        nnz_blocks;
};

// Single-head forward. Q, K, V, O are device pointers, row-major [N, d].
// Returns elapsed GPU time in ms (cudaEvent-based).
float attention_forward(
    const float* Q, const float* K, const float* V,
    float* O,
    const AttnConfig& cfg,
    cudaStream_t stream = 0);

// Multi-head wrapper. Layout: [H, N, d] row-major. One stream per head.
float attention_forward_mha(
    const float* Q, const float* K, const float* V,
    float* O,
    const AttnConfig& cfg);
