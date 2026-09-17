// ============================================================================
// Kernel C — Block-sparse Attention with CSR Block Layout (CUDA Implementation)
// ============================================================================
//
// ------------- OVERVIEW -------------
// This kernel implements block-sparse attention using a memory-efficient
// compressed sparse row (CSR) layout over the attention matrix's block grid.
// Only those blocks (tiles) of the attention matrix marked as "active" are
// computed, yielding substantial speedup on sparse patterns versus dense kernels.
//
// - Each CUDA block is responsible for one "query block-row" (i.e. 64 consecutive
//   query positions, 'Br=64').
// - K/V tiles are selected per block-row according to the CSR pointers:
//     * row_ptr points into col_idx, which lists which K/V block-tiles to visit.
// - The main difference versus the dense/windowed kernels is how block tiling
//   is selected; everything else (online softmax, output accumulation, SMEM layout)
//   is almost identical.
//
//     * Restrictions (v1):
//         - Only FP32 is supported
//         - Only d=64 (head size) is supported
//         - Only non-causal (i.e. no left-masking) is supported for now
//           (see dense/windowed kernels for per-element masking if needed)

#include <cstdio>
#include <cmath>
#include <type_traits>
#include <cuda_runtime.h>

#include "attention_api.h"
#include "kernels_internal.h"
#include "utils.cuh"

namespace {

// ============================================================================
// blocksparse_forward_kernel
//
// Template parameters:
//    Br: Number of query positions ("rows") handled by this CUDA block
//    Bc: Number of K/V positions loaded and processed per tile
//    D:  Embedding dimension (head size); here, always 64
//
//      Q: [N, D] — Query sequence
//      K: [N, D] — Key sequence
//      V: [N, D] — Value sequence
//      O: [N, D] — Output sequence (final attention results)
//      row_ptr: CSR block row pointer array (size ≈ N/Br + 1)
//      col_idx: CSR block nonzero column indices (see row_ptr for indirection)
//      N: Full sequence length
//
template <int Br, int Bc, int D>
__global__ void blocksparse_forward_kernel(
    const float* __restrict__ Q,       // Input query matrix
    const float* __restrict__ K,       // Input key matrix
    const float* __restrict__ V,       // Input value matrix
    float* __restrict__ O,             // Output matrix
    const int*   __restrict__ row_ptr, // CSR row pointer (per block-row)
    const int*   __restrict__ col_idx, // CSR col indices (per block-col in each row)
    int N,                             // Total number of rows/tokens
    bool causal                        // If true, mask scores where col > row
)
{
    // -- SHARED MEMORY TILE LAYOUT --
    // Layout in shared memory:
    //   [  Qs  |  Ks   |  Vs  ]
    //   Br*Dp   Bc*Dp   Bc*Dp      where Dp = D+1 (add 1 for bank conflict avoidance)
    constexpr int Dp = D + 1;
    extern __shared__ float smem[];
    float (*Qs)[Dp] = reinterpret_cast<float (*)[Dp]>(smem);                      // Query tile
    float (*Ks)[Dp] = reinterpret_cast<float (*)[Dp]>(smem + Br * Dp);            // Key tile
    float (*Vs)[Dp] = reinterpret_cast<float (*)[Dp]>(smem + (Br + Bc) * Dp);     // Value tile

    // -- THREAD & ROW INDEXING --
    const int tx       = threadIdx.x;              // Local thread ID (0 ... Br-1); handles one query row
    const int row_base = blockIdx.x * Br;          // Which group of query rows is this block handling?
    const int row      = row_base + tx;            // Absolute index of the sequence row (global token idx)
    const bool active  = (row < N);                // Is this thread mapped to a real row or padded past-end?

    // -- LOAD QUERIES INTO SHMEM --
    // Each thread copies its assigned row of Q into Qs; padding threads get zeros.
    // Cooperative float4 / coalesced loader.
    load_tile_f4<Br, Br, D, Dp>(Qs, Q, row_base, N, tx);

    // -- PER-THREAD STATE TRACKING --
    float m_i = -INFINITY;    // Keeps running max for online softmax (numerical stability)
    float l_i = 0.0f;         // Keeps running sum for online denominator
    float O_i[D];             // Per-thread accumulation of the output value
    #pragma unroll
    for (int k = 0; k < D; ++k) O_i[k] = 0.0f;

    const float scale = rsqrtf((float)D);   // Standard attention scaling (1/sqrt(D))

    // -- CSR TILE SELECTION (block-row iteration) --
    // Get which [block_col] indices to process for this block-row using CSR
    const int start = row_ptr[blockIdx.x];        // Start of indices for this block-row
    const int end   = row_ptr[blockIdx.x + 1];    // End of indices for this block-row

    // -------- MAIN TILE LOOP (over block-columns) --------
    for (int idx = start; idx < end; ++idx) {
        const int block_col = col_idx[idx];       // Which K/V block tile do we need for this block-row?
        const int col_base  = block_col * Bc;     // Absolute index of the first col in this block (K/V)

        // -- LOAD K & V TILE INTO SHARED MEMORY --
        // Each thread copies the Ks/Vs for its assigned tile row; out-of-bounds tile
        // rows are zeroed. Cooperative float4 / coalesced loader.
        load_tile_f4<Br, Bc, D, Dp>(Ks, K, col_base, N, tx);
        load_tile_f4<Br, Bc, D, Dp>(Vs, V, col_base, N, tx);
        __syncthreads(); // Ensure all Ks/Vs loaded for all tile rows before proceeding

        // -- COMPUTE ATTENTION & ACCUMULATE OUTPUT FOR THIS TILE --
        // Chunked compute + online softmax (Cn=16 cols). Eliminates the
        // per-thread float s[Bc=64] register buffer in favour of a smaller
        // float s_chunk[16], at the cost of 4 corrections per tile instead
        // of 1. See dense.cu for the full rationale.
        if (active) {
            constexpr int Cn = 16;
            #pragma unroll
            for (int j_base = 0; j_base < Bc; j_base += Cn) {
                float s_chunk[Cn];
                #pragma unroll
                for (int jj = 0; jj < Cn; ++jj) {
                    const int j   = j_base + jj;
                    const int col = col_base + j;
                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < D; ++k) acc += Qs[tx][k] * Ks[j][k];
                    const bool ok = (col < N) && (!causal || col <= row);
                    s_chunk[jj] = ok ? (acc * scale) : -INFINITY;
                }

                float m_chunk = -INFINITY;
                #pragma unroll
                for (int jj = 0; jj < Cn; ++jj) m_chunk = fmaxf(m_chunk, s_chunk[jj]);

                const float m_new      = fmaxf(m_i, m_chunk);
                const float correction = (m_i == -INFINITY) ? 0.0f
                                       : expf(m_i - m_new);
                l_i = l_i * correction;
                #pragma unroll
                for (int k = 0; k < D; ++k) O_i[k] *= correction;

                #pragma unroll
                for (int jj = 0; jj < Cn; ++jj) {
                    const float p = (s_chunk[jj] == -INFINITY) ? 0.0f
                                  : expf(s_chunk[jj] - m_new);
                    l_i += p;
                    #pragma unroll
                    for (int k = 0; k < D; ++k) O_i[k] += p * Vs[j_base + jj][k];
                }
                m_i = m_new;
            }
        }
        __syncthreads();
        // Note: __syncthreads() is SAFE here: all threads participate, either as active or as padding.
    }

    // -- FINAL WRITE-BACK (normalize output and write to O) --
    if (active) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;    // Defensive: check l_i>0 to avoid div by 0
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O[row * D + k] = O_i[k] * inv_l;    // Write out normalized attentive sum for each D
    }
}
// ============================================================================

}  // namespace


// ============================================================================
// blocksparse_forward (Host/Driver Function)
//
// This is the kernel launcher function—called from the C++ host code.
// It checks kernel suitability, preps parameters & shared memory size, 
// and launches the kernel on the provided CUDA stream.
//
//    Q, K, V: pointers to query/key/value matrices [N, D] (column-major)
//    O: pointer to output matrix [N, D]
//    cfg: AttnConfig struct, layout/size/blocking info
//    stream: CUDA stream for launching kernel
//    measure: if true, measures GPU forward time using CudaTimer
//
// Returns: 0.0f (success, not measured), or measured time in ms (if measure).
//           Returns negative value and prints error if unsupported config.
//
float blocksparse_forward(const float* Q, const float* K, const float* V,
                          float* O, const AttnConfig& cfg, cudaStream_t stream,
                          bool measure)
{
    if (cfg.d != 64 && cfg.d != 128) {
        fprintf(stderr, "[blocksparse_forward] d=%d not supported (only d=64 or d=128)\n", cfg.d);
        return -1.0f;
    }
    // The kernel uses Br == B_block (one CUDA block per CSR block-row), so
    // they must match. We support (d=64, B=64) and (d=128, B=32); the d=128
    // case shrinks the block to fit SMEM.
    const int expected_B = (cfg.d == 64) ? 64 : 32;
    if (cfg.B_block != expected_B) {
        fprintf(stderr, "[blocksparse_forward] d=%d requires B_block=%d (got %d)\n",
                cfg.d, expected_B, cfg.B_block);
        return -1.0f;
    }
    if (cfg.block_row_ptr == nullptr) {
        fprintf(stderr, "[blocksparse_forward] block_row_ptr is null\n");
        return -1.0f;
    }

    auto launch = [&](auto Br_v, auto Bc_v, auto D_v) -> float {
        constexpr int Br = Br_v.value, Bc = Bc_v.value, D = D_v.value;
        constexpr int Dp = D + 1;
        constexpr int smem_bytes = (Br + 2 * Bc) * Dp * (int)sizeof(float);

        static bool smem_opted_in = false;
        if (!smem_opted_in) {
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)blocksparse_forward_kernel<Br, Bc, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
            smem_opted_in = true;
        }
        const int blocks = (cfg.N + Br - 1) / Br;
        dim3 grid(blocks), block(Br);
        if (!measure) {
            blocksparse_forward_kernel<Br, Bc, D>
                <<<grid, block, smem_bytes, stream>>>(Q, K, V, O,
                    cfg.block_row_ptr, cfg.block_col_idx, cfg.N, cfg.causal);
            return 0.0f;
        }
        CudaTimer timer;
        timer.start(stream);
        blocksparse_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(Q, K, V, O,
                cfg.block_row_ptr, cfg.block_col_idx, cfg.N, cfg.causal);
        return timer.stop(stream);
    };

    if (cfg.d == 64) {
        return launch(std::integral_constant<int, 64>{},
                      std::integral_constant<int, 64>{},
                      std::integral_constant<int, 64>{});
    } else {
        return launch(std::integral_constant<int, 32>{},
                      std::integral_constant<int, 32>{},
                      std::integral_constant<int, 128>{});
    }
}
// ============================================================================