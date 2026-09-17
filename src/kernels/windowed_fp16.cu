// ============================================================================
// Kernel B2 - Sliding-window attention via FP16 + Tensor Cores (WMMA).
// ============================================================================
//
// Mirrors dense_fp16.cu's design: 4 warps per block, m16n16k16 fragment shape,
// FP16 inputs, FP32 softmax + accumulator, Ks/Ps unioned in SMEM to fit the
// 64 KB Turing budget. Differences from the dense FP16 kernel:
//
//   1. Tile-iteration range. Only iterate K/V tiles whose column range
//      [t*Bc, (t+1)*Bc) intersects the per-block window
//      [row_base - w, row_base + Br - 1 + w]. At N=4096, w=256, Br=64 this
//      is ~8 tiles per block-row instead of 64, an algorithmic ~8x win
//      that compounds with the tensor-core implementation lift.
//   2. Per-element window mask in the softmax phase. After store_matrix_sync
//      writes FP32 scores to Ss, the per-row softmax checks
//      |i - j| <= w and col < N; masked scores are set to -INFINITY so they
//      contribute nothing to the running max / sum.
//
// Restrictions (v1): non-causal, G = 0 (no global tokens). Both follow the
// same per-element-mask pattern as the FP32 windowed kernel and are
// straightforward extensions.

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

// Python-style floor division (handles negatives correctly).
__device__ __forceinline__ int floor_div_neg(int a, int b) {
    int q = a / b;
    if ((a % b) != 0 && ((a ^ b) < 0)) --q;
    return q;
}

template <int Br, int Bc, int D>
__global__ __launch_bounds__((Br/16)*32, 1) void windowed_fp16_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int w, bool causal)
{
    static_assert((Br == 64 && Bc == 64 && D == 64) ||
                  (Br == 32 && Bc == 32 && D == 128),
                  "windowed_fp16 kernel currently fixed at 64x64x64 tiles");
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int NUM_WARPS = Br / WM;          // 4
    constexpr int THREADS   = NUM_WARPS * 32;   // 128
    constexpr int N_FRAGS_C = Bc / WN;          // 4
    constexpr int N_FRAGS_O = D  / WN;          // 4

    constexpr int Dp = D + 8;
    constexpr int Bp = (Bc > D) ? Bc : D;   // Ss holds both QK^T scores (Br x Bc) and PV partial output (Br x D); size to the larger.

    extern __shared__ unsigned char smem_raw[];
    auto Qs = reinterpret_cast<__half (*)[Dp]>(smem_raw);
    auto Ks = reinterpret_cast<__half (*)[Dp]>(reinterpret_cast<unsigned char*>(Qs) + Br * Dp * sizeof(__half));
    auto Ps = reinterpret_cast<__half (*)[Dp]>(reinterpret_cast<unsigned char*>(Ks));  // unioned with Ks
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

    // ---- Per-row state init + zero running O ----
    if (tid < Br) {
        m_shared[tid] = -INFINITY;
        l_shared[tid] = 0.0f;
    }
    #pragma unroll
    for (int i = tid; i < Br * Dp; i += THREADS) {
        (reinterpret_cast<float*>(Os))[i] = 0.0f;
    }

    // ---- Load Q tile (FP32 -> FP16) ----
    #pragma unroll
    for (int i = tid; i < Br * D; i += THREADS) {
        const int r = i / D, c = i % D;
        const int g_row = row_base + r;
        Qs[r][c] = (g_row < N) ? __float2half(Q[g_row * D + c]) : __float2half(0.f);
    }
    __syncthreads();

    // ---- Compute the windowed tile range ----
    // Window for this block covers global cols [row_base - w, row_base + Br - 1 + w].
    const int num_tiles = (N + Bc - 1) / Bc;
    int t_lo = floor_div_neg(row_base - w, Bc);
    if (t_lo < 0) t_lo = 0;
    int t_hi = (row_base + Br - 1 + w) / Bc + 1;
    if (t_hi > num_tiles) t_hi = num_tiles;

    // ---- Tile loop ----
    for (int t = t_lo; t < t_hi; ++t) {
        const int col_base = t * Bc;

        // ---- Load K, V FP32 -> FP16 ----
        #pragma unroll
        for (int i = tid; i < Bc * D; i += THREADS) {
            const int r = i / D, c = i % D;
            const int g_row = col_base + r;
            const bool ok = (g_row < N);
            Ks[r][c] = ok ? __float2half(K[g_row * D + c]) : __float2half(0.f);
            Vs[r][c] = ok ? __float2half(V[g_row * D + c]) : __float2half(0.f);
        }
        __syncthreads();

        // ---- QK^T via WMMA ----
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

        // ---- Store scores ----
        #pragma unroll
        for (int j = 0; j < N_FRAGS_C; ++j) {
            wmma::store_matrix_sync(&Ss[warp_row_off][j * WN], s_frag[j], Bp, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Softmax over scores with window + bounds mask ----
        // Two-pass design: first pass finds row_max by reading Ss; second pass
        // recomputes scores (cheap: just mask + scale + exp) and writes Ps.
        // The mask + exp on the second pass costs a few extra ALU ops per
        // element but removes the per-thread scores[Bc] register stack array
        // that was tipping the d=128 instantiation into a runtime fault.
        if (lane < WM) {
            const int row = warp_row_off + lane;
            const int g_row_idx = row_base + row;
            const float scale_f = rsqrtf((float)D);

            float row_max = -INFINITY;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                const int col        = col_base + j;
                const int dij        = g_row_idx - col;
                const int adij       = dij < 0 ? -dij : dij;
                const bool in_window = (adij <= w);
                const bool in_bounds = (col < N);
                const bool causal_ok = !causal || (col <= g_row_idx);
                if (in_window && in_bounds && causal_ok) {
                    row_max = fmaxf(row_max, Ss[row][j] * scale_f);
                }
            }

            const float m_old      = m_shared[row];
            const float m_new      = fmaxf(m_old, row_max);
            const float correction = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);

            #pragma unroll
            for (int k = 0; k < D; ++k) Os[row][k] *= correction;

            float l_new = l_shared[row] * correction;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                const int col        = col_base + j;
                const int dij        = g_row_idx - col;
                const int adij       = dij < 0 ? -dij : dij;
                const bool in_window = (adij <= w);
                const bool in_bounds = (col < N);
                const bool causal_ok = !causal || (col <= g_row_idx);
                float p = 0.0f;
                if (in_window && in_bounds && causal_ok) {
                    p = expf(Ss[row][j] * scale_f - m_new);
                }
                l_new += p;
                Ps[row][j] = __float2half(p);
            }
            m_shared[row] = m_new;
            l_shared[row] = l_new;
        }
        __syncthreads();

        // ---- PV via WMMA, accumulate into Os ----
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

        // Store fragments to Ss (scratch) and accumulate into Os.
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

    // ---- Normalize and write back ----
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

// ============================================================================
// Sequence-parallel variant of the WMMA windowed kernel.
//
// Mirrors windowed_fp16_forward_kernel, with two extra parameters:
//   N_q          : number of local query rows (= N/P in the seq-par driver)
//   N_k          : number of local K rows including halos on each side
//   local_offset : left-halo size; the kernel's local Q row r corresponds to
//                  local K col r + local_offset on the diagonal.
// The window check becomes |i + local_offset - j| <= w in local coordinates.
// Causal in seq-par uses the local-coord diagonal: col <= row + local_offset.
// G > 0 is not yet supported here; the host driver enforces G == 0.
// ============================================================================
template <int Br, int Bc, int D>
__global__ __launch_bounds__((Br/16)*32, 1) void windowed_fp16_seqpar_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N_q, int N_k, int w, int local_offset, bool causal)
{
    static_assert((Br == 64 && Bc == 64 && D == 64) ||
                  (Br == 32 && Bc == 32 && D == 128),
                  "windowed_fp16_seqpar fixed at 64x64x64 tiles");
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

    // Load Q
    #pragma unroll
    for (int i = tid; i < Br * D; i += THREADS) {
        const int r = i / D, c = i % D;
        const int g_row = row_base + r;
        Qs[r][c] = (g_row < N_q) ? __float2half(Q[g_row * D + c]) : __float2half(0.f);
    }
    __syncthreads();

    // Local K tile range, in local-K coords. Window for local row r covers
    // local K cols [r + local_offset - w, r + local_offset + w].
    const int num_kv_tiles = (N_k + Bc - 1) / Bc;
    int t_lo = floor_div_neg(row_base + local_offset - w, Bc);
    if (t_lo < 0) t_lo = 0;
    int t_hi = (row_base + Br - 1 + local_offset + w) / Bc + 1;
    if (t_hi > num_kv_tiles) t_hi = num_kv_tiles;

    // Causal: in local coords no row attends past col = row + local_offset.
    if (causal) {
        const int t_hi_causal = (row_base + Br - 1 + local_offset) / Bc + 1;
        if (t_hi_causal < t_hi) t_hi = t_hi_causal;
    }

    for (int t = t_lo; t < t_hi; ++t) {
        const int col_base = t * Bc;

        #pragma unroll
        for (int i = tid; i < Bc * D; i += THREADS) {
            const int r = i / D, c = i % D;
            const int g_row = col_base + r;
            const bool ok = (g_row < N_k);
            Ks[r][c] = ok ? __float2half(K[g_row * D + c]) : __float2half(0.f);
            Vs[r][c] = ok ? __float2half(V[g_row * D + c]) : __float2half(0.f);
        }
        __syncthreads();

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

        // Softmax with local-coord windowed mask + optional causal gate.
        // Two-pass design (see windowed_fp16_forward_kernel for rationale).
        if (lane < WM) {
            const int row = warp_row_off + lane;
            const float scale_f = rsqrtf((float)D);

            float row_max = -INFINITY;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                const int col        = col_base + j;
                const int dij        = (row + local_offset) - col;
                const int adij       = dij < 0 ? -dij : dij;
                const bool in_window = (adij <= w);
                const bool in_bounds = (col < N_k);
                const bool causal_ok = !causal || (col <= row + local_offset);
                if (in_window && in_bounds && causal_ok) {
                    row_max = fmaxf(row_max, Ss[row][j] * scale_f);
                }
            }

            const float m_old      = m_shared[row];
            const float m_new      = fmaxf(m_old, row_max);
            const float correction = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);

            #pragma unroll
            for (int k = 0; k < D; ++k) Os[row][k] *= correction;

            float l_new = l_shared[row] * correction;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                const int col        = col_base + j;
                const int dij        = (row + local_offset) - col;
                const int adij       = dij < 0 ? -dij : dij;
                const bool in_window = (adij <= w);
                const bool in_bounds = (col < N_k);
                const bool causal_ok = !causal || (col <= row + local_offset);
                float p = 0.0f;
                if (in_window && in_bounds && causal_ok) {
                    p = expf(Ss[row][j] * scale_f - m_new);
                }
                l_new += p;
                Ps[row][j] = __float2half(p);
            }
            m_shared[row] = m_new;
            l_shared[row] = l_new;
        }
        __syncthreads();

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
        if (g_row < N_q) {
            const float l_v = l_shared[r];
            const float inv = (l_v > 0.0f) ? (1.0f / l_v) : 0.0f;
            O[g_row * D + c] = Os[r][c] * inv;
        }
    }
}

} // namespace

float windowed_fp16_forward(const float* Q, const float* K, const float* V,
                            float* O, const AttnConfig& cfg, cudaStream_t stream,
                            bool measure)
{
    if (cfg.d != 64 && cfg.d != 128) {
        fprintf(stderr, "[windowed_fp16] d=%d not supported (only d=64 or d=128)\n", cfg.d);
        return -1.0f;
    }
    if (cfg.G != 0) {
        fprintf(stderr, "[windowed_fp16] G>0 not yet supported\n");
        return -1.0f;
    }
    if (cfg.w < 0) {
        fprintf(stderr, "[windowed_fp16] invalid w=%d\n", cfg.w);
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
                (const void*)windowed_fp16_forward_kernel<Br, Bc, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
            smem_opted_in = true;
        }

        const int blocks = (cfg.N + Br - 1) / Br;
        dim3 grid(blocks), block(THREADS);
        if (!measure) {
            windowed_fp16_forward_kernel<Br, Bc, D>
                <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, cfg.N, cfg.w, cfg.causal);
            return 0.0f;
        }
        CudaTimer timer;
        timer.start(stream);
        windowed_fp16_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, cfg.N, cfg.w, cfg.causal);
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

// ----------------------------------------------------------------------------
// Seq-par host wrapper. Mirrors windowed_seqpar_forward (FP32) but launches
// the WMMA kernel. Used by the MPI seq-par driver in src/mpi/seq_parallel.cu.
// ----------------------------------------------------------------------------
float windowed_fp16_seqpar_forward(const float* Q, const float* K, const float* V,
                                   float* O, int N_q, int N_k, int d, int w,
                                   int local_offset, bool causal,
                                   cudaStream_t stream, bool measure)
{
    if (d != 64 && d != 128) {
        fprintf(stderr, "[windowed_fp16_seqpar] d=%d not supported (only 64 or 128)\n", d);
        return -1.0f;
    }
    if (w < 0 || local_offset < 0) {
        fprintf(stderr, "[windowed_fp16_seqpar] invalid w=%d or local_offset=%d\n",
                w, local_offset);
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
                (const void*)windowed_fp16_seqpar_forward_kernel<Br, Bc, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
            smem_opted_in = true;
        }
        const int blocks = (N_q + Br - 1) / Br;
        dim3 grid(blocks), block(THREADS);
        if (!measure) {
            windowed_fp16_seqpar_forward_kernel<Br, Bc, D>
                <<<grid, block, smem_bytes, stream>>>(
                    Q, K, V, O, N_q, N_k, w, local_offset, causal);
            return 0.0f;
        }
        CudaTimer timer;
        timer.start(stream);
        windowed_fp16_seqpar_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(
                Q, K, V, O, N_q, N_k, w, local_offset, causal);
        return timer.stop(stream);
    };

    if (d == 64) {
        return launch(std::integral_constant<int, 64>{},
                      std::integral_constant<int, 64>{},
                      std::integral_constant<int, 64>{});
    } else {
        return launch(std::integral_constant<int, 32>{},
                      std::integral_constant<int, 32>{},
                      std::integral_constant<int, 128>{});
    }
}
