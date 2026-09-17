// Kernel A — dense exact attention, FlashAttention-style fused forward.
//
// This file implements a dense, non-causal, single-head attention. 
// - Each CUDA block computes attention for Br query rows, fully in parallel.
// - Each thread computes one row - all of its computation is private in registers.
// - There is no inter-thread reduction: softmax and output are computed privately per thread.
// - Only F32 is supported here, and only d=64 for both Q/K/V/O. (d=128 may arrive in the future.)
// - For multi-head, dispatch.cu will launch this per head in a separate CUDA stream.

#include <cstdio>
#include <cmath>
#include <type_traits>
#include <cuda_runtime.h>

#include "attention_api.h"
#include "kernels_internal.h"
#include "utils.cuh"

namespace {

// The primary computation kernel for dense attention. 
// Br: Number of query rows handled per block (and thus per kernel launch block).
// Bc: Number of key/value rows loaded per tile iteration (i.e., tile width).
// D:  Head dimension.
template <int Br, int Bc, int D>
__global__ void dense_forward_kernel(
    const float* __restrict__ Q,  // [N, D] Query matrix
    const float* __restrict__ K,  // [N, D] Key matrix
    const float* __restrict__ V,  // [N, D] Value matrix
    float* __restrict__ O,        // [N, D] Output matrix (result)
    int N,                        // Number of rows (sequence length)
    bool causal                   // If true, mask scores where col > row
)
{
    // To minimize shared memory bank conflicts, we pad the col dimension by +1 (Dp = D+1).
    // We allocate a block of shared memory, organized as rows of Qs, Ks, and Vs.
    //   smem[0 ... Br*Dp)      : Qs
    //   smem[Br*Dp ... (Br+Bc)*Dp) : Ks
    //   smem[(Br+Bc)*Dp ... (Br+2*Bc)*Dp) : Vs
    constexpr int Dp = D + 1;
    extern __shared__ float smem[];
    float (*Qs)[Dp] = reinterpret_cast<float (*)[Dp]>(smem);
    float (*Ks)[Dp] = reinterpret_cast<float (*)[Dp]>(smem + Br * Dp);
    float (*Vs)[Dp] = reinterpret_cast<float (*)[Dp]>(smem + (Br + Bc) * Dp); // Note: layout is contiguous

    // Thread/block indices
    const int tx       = threadIdx.x;   // [0, Br): which query row this thread handles within the block
    const int row_base = blockIdx.x * Br; // Starting absolute row for this block
    const int row      = row_base + tx; // Absolute sequence row index for this thread
    const bool active  = (row < N);     // Is this thread within the valid rows bounds (may be padding at end of N)?

    // ---- Step 1: Cooperatively load the Q block into shared memory using
    // float4 vectorized, fully coalesced loads. All Br threads participate.
    load_tile_f4<Br, Br, D, Dp>(Qs, Q, row_base, N, tx);

    // ---- Step 2: Per-thread state for online softmax and output computation ----
    // m_i: running max for numerical stability in softmax (per row)
    // l_i: running normalizer (denominator for softmax, per row)
    // O_i: partial sum of attention-weighted V_j for this row, dimension D
    float m_i = -INFINITY; // Initial max (-inf means none seen yet)
    float l_i = 0.0f;      // Initial normalizer is zero (nothing seen yet)
    float O_i[D];          // Partial output row accumulator
    #pragma unroll
    for (int k = 0; k < D; ++k) O_i[k] = 0.0f;

    // Precompute the scaling for the Q·K dot product (standard attention scaling)
    const float scale = rsqrtf((float)D);

    // We iterate over tiles of the K/V rows. Each tile loads Bc K rows and Bc V rows.
    const int num_kv_tiles = (N + Bc - 1) / Bc;

    // For causal attention, no row in this block attends past col = row_base + Br - 1,
    // so we can stop iterating tiles after the one containing that column.
    int t_end = num_kv_tiles;
    if (causal) {
        const int t_end_causal = (row_base + Br - 1) / Bc + 1;
        if (t_end_causal < t_end) t_end = t_end_causal;
    }

    // ---- Step 3: Loop over K/V tiles ----
    for (int t = 0; t < t_end; ++t) {
        const int col_base = t * Bc; // Global index in K/V we're loading this tile

        // Cooperative coalesced loads of the K and V tiles (float4 per thread).
        load_tile_f4<Br, Bc, D, Dp>(Ks, K, col_base, N, tx);
        load_tile_f4<Br, Bc, D, Dp>(Vs, V, col_base, N, tx);
        __syncthreads();

        // ---- Step 4: Compute scores + online softmax update, CHUNKED ----
        //
        // Instead of materializing all Bc=64 scores in a per-thread register
        // buffer (s[64] -> ~64 regs/thread of pressure), we process the tile
        // in chunks of Cn=16 cols. Each chunk: compute Cn scores, find the
        // chunk's max, apply one correction to (l_i, O_i), then accumulate
        // the Cn probabilities. Mathematically identical to the one-pass
        // FA-style update; the only cost is a few extra corrections per tile.
        // Saves ~48 registers / thread, lifting occupancy.
        if (active) {
            constexpr int Cn = 16;
            #pragma unroll
            for (int j_base = 0; j_base < Bc; j_base += Cn) {
                float s_chunk[Cn];

                // Compute Cn dot products + masks.
                #pragma unroll
                for (int jj = 0; jj < Cn; ++jj) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < D; ++k)
                        acc += Qs[tx][k] * Ks[j_base + jj][k];
                    const int col = col_base + j_base + jj;
                    const bool ok = (col < N) && (!causal || col <= row);
                    s_chunk[jj] = ok ? (acc * scale) : -INFINITY;
                }

                // Reduce max over the chunk.
                float m_chunk = -INFINITY;
                #pragma unroll
                for (int jj = 0; jj < Cn; ++jj)
                    m_chunk = fmaxf(m_chunk, s_chunk[jj]);

                // Online-softmax correction step. If m_chunk == -INFINITY
                // (all Cn cols were masked), m_new == m_i and correction is
                // 1 (or 0 when m_i was also -INFINITY), so l_i / O_i are
                // unchanged and no probabilities are added below.
                const float m_new      = fmaxf(m_i, m_chunk);
                const float correction = (m_i == -INFINITY) ? 0.0f
                                       : expf(m_i - m_new);
                l_i = l_i * correction;
                #pragma unroll
                for (int k = 0; k < D; ++k) O_i[k] *= correction;

                // Accumulate the chunk's probabilities.
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
        __syncthreads(); // Wait before modifying shared memory in next tile
    }

    // ---- Step 6: After all tiles, write normalized output for this row ----
    if (active) {
        // Normalize by softmax denominator, unless it is zero (abnormal; should not happen)
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll
        for (int k = 0; k < D; ++k) {
            // Store output O[i, k] to global memory
            O[row * D + k] = O_i[k] * inv_l;
        }
    }
}

}  // namespace

// Host-side entry point for this dense kernel.
// Arguments: Pointers for Q, K, V, O; config struct; CUDA stream; measure flag.
// When measure=true (default), returns the kernel's elapsed time via cudaEvent
// (which calls cudaEventSynchronize and therefore blocks the host).
// When measure=false, just enqueues the launch on `stream` and returns 0.0f
// — the caller (e.g. attention_forward_mha) is responsible for any sync, and
// kernels on different streams can overlap.
float dense_forward(const float* Q, const float* K, const float* V,
                    float* O, const AttnConfig& cfg, cudaStream_t stream,
                    bool measure)
{
    if (cfg.d != 64 && cfg.d != 128) {
        fprintf(stderr, "[dense_forward] d=%d not supported (only d=64 or d=128)\n", cfg.d);
        return -1.0f;
    }

    // Dispatch on d. At d=64 we use the original 64x64x64 tile shape; at
    // d=128 we shrink Br=Bc=32 because the larger D doubles the per-row SMEM
    // footprint and the 64x64x128 design overflows the 64 KB Turing limit.
    auto launch = [&](auto Br_v, auto Bc_v, auto D_v) -> float {
        constexpr int Br = Br_v.value, Bc = Bc_v.value, D = D_v.value;
        constexpr int Dp = D + 1;
        constexpr int smem_bytes = (Br + 2 * Bc) * Dp * (int)sizeof(float);

        static bool smem_opted_in = false;
        if (!smem_opted_in) {
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)dense_forward_kernel<Br, Bc, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
            smem_opted_in = true;
        }

        const int blocks = (cfg.N + Br - 1) / Br;
        dim3 grid(blocks), block(Br);

        if (!measure) {
            dense_forward_kernel<Br, Bc, D>
                <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, cfg.N, cfg.causal);
            return 0.0f;
        }
        CudaTimer timer;
        timer.start(stream);
        dense_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, cfg.N, cfg.causal);
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
