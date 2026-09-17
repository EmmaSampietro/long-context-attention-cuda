// ============================================================================
// Kernel C2 - Block-sparse attention via FP16 + Tensor Cores (WMMA).
// ============================================================================
//
// Mirrors dense_fp16.cu and windowed_fp16.cu's design: 4 warps per block,
// m16n16k16 fragment shape, FP16 inputs, FP32 softmax + accumulator, Ks/Ps
// union in SMEM to fit the 64 KB Turing budget. The only differences from
// windowed_fp16 are:
//
//   1. Tile-iteration is driven by the CSR row_ptr/col_idx (same as
//      blocksparse.cu): each query block-row visits only the active K/V
//      block-columns enumerated by col_idx[row_ptr[r] .. row_ptr[r+1]).
//   2. The per-element mask is the simple bounds check col < N plus an
//      optional causal gate col <= row; there is no window check.
//
// Restrictions: d=64, B_block=64, non-causal in v1 (the gate is in place but
// a tile-level skip for entirely-above-diagonal blocks is not implemented;
// per-element masking handles correctness either way).

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <type_traits>

#include "attention_api.h"
#include "kernels_internal.h"
#include "utils.cuh"

using namespace nvcuda;

namespace {

template <int Br, int Bc, int D>
__global__ __launch_bounds__((Br/16)*32, 1) void blocksparse_fp16_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    int N, bool causal)
{
    static_assert((Br == 64 && Bc == 64 && D == 64) ||
                  (Br == 32 && Bc == 32 && D == 128),
                  "blocksparse_fp16: supported tiles are 64x64x64 (d=64) or 32x32x128 (d=128)");
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int NUM_WARPS = Br / WM;
    constexpr int THREADS   = NUM_WARPS * 32;
    constexpr int N_FRAGS_C = Bc / WN;
    constexpr int N_FRAGS_O = D  / WN;

    constexpr int Dp = D + 8;
    constexpr int Bp = (Bc > D) ? Bc : D;   // Ss holds both QK^T scores (Br x Bc) and PV partial output (Br x D); size to the larger.

    extern __shared__ unsigned char smem_raw[];
    auto Qs = reinterpret_cast<__half (*)[Dp]>(smem_raw);
    auto Ks = reinterpret_cast<__half (*)[Dp]>(reinterpret_cast<unsigned char*>(Qs) + Br * Dp * sizeof(__half));
    auto Ps = reinterpret_cast<__half (*)[Dp]>(reinterpret_cast<unsigned char*>(Ks));
    auto Vs = reinterpret_cast<__half (*)[Dp]>(reinterpret_cast<unsigned char*>(Ks) + Bc * Dp * sizeof(__half));
    auto Ss = reinterpret_cast<float  (*)[Bp]>(reinterpret_cast<unsigned char*>(Vs) + Bc * Dp * sizeof(__half));
    auto Os = reinterpret_cast<float  (*)[Dp]>(reinterpret_cast<unsigned char*>(Ss) + Br * Bp * sizeof(float));
    auto m_shared = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(Os) + Br * Dp * sizeof(float));
    auto l_shared = m_shared + Br;

    const int tid          = threadIdx.x;
    const int warp_id      = tid / 32;
    const int lane         = tid % 32;
    const int row_base     = blockIdx.x * Br;
    const int warp_row_off = warp_id * WM;

    if (tid < Br) {
        m_shared[tid] = -INFINITY;
        l_shared[tid] = 0.0f;
    }
    #pragma unroll
    for (int i = tid; i < Br * Dp; i += THREADS) {
        (reinterpret_cast<float*>(Os))[i] = 0.0f;
    }

    #pragma unroll
    for (int i = tid; i < Br * D; i += THREADS) {
        const int r = i / D, c = i % D;
        const int g_row = row_base + r;
        Qs[r][c] = (g_row < N) ? __float2half(Q[g_row * D + c]) : __float2half(0.f);
    }
    __syncthreads();

    // CSR iterate: visit the active block-cols for this block-row.
    const int start = row_ptr[blockIdx.x];
    const int end   = row_ptr[blockIdx.x + 1];

    for (int idx = start; idx < end; ++idx) {
        const int block_col = col_idx[idx];
        const int col_base  = block_col * Bc;

        // Optional whole-tile skip when causal and the entire tile is above
        // the diagonal for every row in this block.
        if (causal && col_base > row_base + Br - 1) continue;

        // Load K, V tiles, cast FP32 -> FP16.
        #pragma unroll
        for (int i = tid; i < Bc * D; i += THREADS) {
            const int r = i / D, c = i % D;
            const int g_row = col_base + r;
            const bool ok = (g_row < N);
            Ks[r][c] = ok ? __float2half(K[g_row * D + c]) : __float2half(0.f);
            Vs[r][c] = ok ? __float2half(V[g_row * D + c]) : __float2half(0.f);
        }
        __syncthreads();

        // QK^T via WMMA.
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag[N_FRAGS_C];
        #pragma unroll
        for (int j = 0; j < N_FRAGS_C; ++j) wmma::fill_fragment(s_frag[j], 0.0f);

        wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> b_frag;

        #pragma unroll
        for (int k_step = 0; k_step < D / WK; ++k_step) {
            wmma::load_matrix_sync(a_frag, &Qs[warp_row_off][k_step * WK], Dp);
            #pragma unroll
            for (int j = 0; j < N_FRAGS_C; ++j) {
                wmma::load_matrix_sync(b_frag, &Ks[j * WN][k_step * WK], Dp);
                wmma::mma_sync(s_frag[j], a_frag, b_frag, s_frag[j]);
            }
        }

        #pragma unroll
        for (int j = 0; j < N_FRAGS_C; ++j) {
            wmma::store_matrix_sync(&Ss[warp_row_off][j * WN], s_frag[j], Bp, wmma::mem_row_major);
        }
        __syncthreads();

        // Softmax with bounds + optional causal mask. Two-pass design to
        // eliminate the per-thread scores[Bc] stack array (see windowed_fp16
        // for the d=128 rationale). Only lanes [0, WM) do the per-row work.
        if (lane < WM) {
            const int row = warp_row_off + lane;
            const int g_row = row_base + row;
            const float scale_f = rsqrtf((float)D);

            float row_max = -INFINITY;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                const int col = col_base + j;
                const bool ok = (col < N) && (!causal || col <= g_row);
                if (ok) row_max = fmaxf(row_max, Ss[row][j] * scale_f);
            }

            const float m_old      = m_shared[row];
            const float m_new      = fmaxf(m_old, row_max);
            const float correction = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);

            #pragma unroll
            for (int k = 0; k < D; ++k) Os[row][k] *= correction;

            float l_new = l_shared[row] * correction;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                const int col = col_base + j;
                const bool ok = (col < N) && (!causal || col <= g_row);
                float p = 0.0f;
                if (ok) p = expf(Ss[row][j] * scale_f - m_new);
                l_new += p;
                Ps[row][j] = __float2half(p);
            }
            m_shared[row] = m_new;
            l_shared[row] = l_new;
        }
        __syncthreads();

        // PV via WMMA.
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_part[N_FRAGS_O];
        #pragma unroll
        for (int j = 0; j < N_FRAGS_O; ++j) wmma::fill_fragment(o_part[j], 0.0f);

        wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> v_frag;

        #pragma unroll
        for (int k_step = 0; k_step < Bc / WK; ++k_step) {
            wmma::load_matrix_sync(p_frag, &Ps[warp_row_off][k_step * WK], Dp);
            #pragma unroll
            for (int j = 0; j < N_FRAGS_O; ++j) {
                wmma::load_matrix_sync(v_frag, &Vs[k_step * WK][j * WN], Dp);
                wmma::mma_sync(o_part[j], p_frag, v_frag, o_part[j]);
            }
        }

        #pragma unroll
        for (int j = 0; j < N_FRAGS_O; ++j) {
            wmma::store_matrix_sync(&Ss[warp_row_off][j * WN], o_part[j], Bp, wmma::mem_row_major);
        }
        __syncthreads();

        #pragma unroll
        for (int i = tid; i < Br * D; i += THREADS) {
            const int r = i / D, c = i % D;
            Os[r][c] += Ss[r][c];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = tid; i < Br * D; i += THREADS) {
        const int r = i / D, c = i % D;
        const int g_row = row_base + r;
        if (g_row < N) {
            const float l_v = l_shared[r];
            const float inv = (l_v > 0.0f) ? (1.0f / l_v) : 0.0f;
            O[g_row * D + c] = Os[r][c] * inv;
        }
    }
}

} // namespace

float blocksparse_fp16_forward(const float* Q, const float* K, const float* V,
                               float* O, const AttnConfig& cfg, cudaStream_t stream,
                               bool measure)
{
    if (cfg.d != 64 && cfg.d != 128) {
        fprintf(stderr, "[blocksparse_fp16] d=%d not supported (only d=64 or d=128)\n", cfg.d);
        return -1.0f;
    }
    const int expected_B = (cfg.d == 64) ? 64 : 32;
    if (cfg.B_block != expected_B) {
        fprintf(stderr, "[blocksparse_fp16] d=%d requires B_block=%d (got %d)\n",
                cfg.d, expected_B, cfg.B_block);
        return -1.0f;
    }
    if (cfg.block_row_ptr == nullptr) {
        fprintf(stderr, "[blocksparse_fp16] block_row_ptr is null\n");
        return -1.0f;
    }

    auto launch = [&](auto Br_v, auto Bc_v, auto D_v) -> float {
        constexpr int Br = Br_v.value, Bc = Bc_v.value, D = D_v.value;
        constexpr int Dp = D + 8;
        constexpr int Bp = (Bc > D) ? Bc : D;
        constexpr int NUM_WARPS = Br / 16;
        constexpr int THREADS   = NUM_WARPS * 32;
        constexpr int smem_bytes =
              Br * Dp * (int)sizeof(__half)
            + Bc * Dp * (int)sizeof(__half)
            + Bc * Dp * (int)sizeof(__half)
            + Br * Bp * (int)sizeof(float)
            + Br * Dp * (int)sizeof(float)
            + 2 * Br * (int)sizeof(float);

        static bool smem_opted_in = false;
        if (!smem_opted_in) {
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)blocksparse_fp16_forward_kernel<Br, Bc, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
            smem_opted_in = true;
        }
        const int blocks = (cfg.N + Br - 1) / Br;
        dim3 grid(blocks), block(THREADS);
        if (!measure) {
            blocksparse_fp16_forward_kernel<Br, Bc, D>
                <<<grid, block, smem_bytes, stream>>>(
                    Q, K, V, O, cfg.block_row_ptr, cfg.block_col_idx, cfg.N, cfg.causal);
            return 0.0f;
        }
        CudaTimer timer;
        timer.start(stream);
        blocksparse_fp16_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(
                Q, K, V, O, cfg.block_row_ptr, cfg.block_col_idx, cfg.N, cfg.causal);
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
