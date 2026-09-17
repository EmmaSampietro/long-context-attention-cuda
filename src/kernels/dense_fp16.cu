// ============================================================================
// Kernel A2 - Dense attention via FP16 + Tensor Cores (WMMA).
// ============================================================================
//
// Same FlashAttention-style fused tiled forward as dense.cu, but the two
// matmuls (QK^T and PV) run on Turing's tensor cores via the nvcuda::wmma
// API. Tile shape is m16n16k16 with FP16 inputs and FP32 accumulators.
// Softmax statistics (m_i, l_i) and the running output O_i stay FP32 for
// numerical stability; only the matmul operands are downcast to FP16.
//
// Inputs and outputs remain FP32 for apples-to-apples comparison with the
// FP32 kernel. The cast happens on load into shared memory (Q, K, V) and
// on the second matmul's "P" operand (softmax probabilities cast to FP16
// just before PV).
//
// Threading layout:
//   - Br = 64 query rows per block, Bc = 64 key cols per K/V tile, D = 64.
//   - 4 warps per block (128 threads). Each warp owns 16 query rows.
//   - Each warp materialises 4 score fragments (one per 16-col stripe of Bc)
//     in registers and 4 output fragments (one per 16-col stripe of D).
//
// Shared-memory layout (must fit under 64 KB on Turing):
//   Qs[Br][Dp]   FP16   9216 B   (Dp = D + 8 padding to avoid bank conflicts)
//   Ks[Bc][Dp]   FP16   9216 B   (also Ps -- temporally disjoint, same memory)
//   Vs[Bc][Dp]   FP16   9216 B
//   Ss[Br][Bp]   FP32  16384 B   (scores after QK^T; Bp = Bc, no padding)
//   Os[Br][Dp]   FP32  18432 B   (running output accumulator across tiles)
//   m_shared, l_shared:  512 B
//   Total:             62976 B   ~61.5 KB. We opt into the >48 KB dynamic
//                                 SMEM allowance (64 KB max on sm_75).
// Ks / Ps share their region because Ks is consumed by the QK^T mma before
// any thread writes a probability into Ps. The first attempt at this kernel
// allocated them separately and overflowed the 64 KB limit.
//
// Restrictions: d=64 only, non-causal in this v1 (the per-element causal
// gate is straightforward to add by masking after store_matrix_sync; left
// for follow-up if time permits).

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <mma.h>

#include "attention_api.h"
#include "kernels_internal.h"
#include "utils.cuh"

using namespace nvcuda;

namespace {

template <int Br, int Bc, int D>
__global__ __launch_bounds__((Br/16)*32, 1) void dense_fp16_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, bool causal)
{
    static_assert((Br == 64 && Bc == 64 && D == 64) ||
                  (Br == 32 && Bc == 32 && D == 128),
                  "dense_fp16: supported tiles are 64x64x64 (d=64) or 32x32x128 (d=128)");
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int NUM_WARPS = Br / WM;          // 4 warps per block
    constexpr int THREADS   = NUM_WARPS * 32;   // 128
    constexpr int N_FRAGS_C = Bc / WN;          // 4 score fragments per warp
    constexpr int N_FRAGS_O = D  / WN;          // 4 output fragments per warp

    // SMEM layout:
    //   Qs[Br][Dp]   FP16,  9216 B   (Dp = D + 8 padding to avoid bank conflicts)
    //   KsPs[Bc][Dp] FP16,  9216 B   (Ks during QK^T, Ps after softmax — same memory)
    //   Vs[Bc][Dp]   FP16,  9216 B
    //   Ss[Br][Bp]   FP32, 16384 B   (Bp = Bc, no padding — fits anyway)
    //   Os[Br][Dp]   FP32, 18432 B
    //   m_shared, l_shared:  512 B
    //   Total:             62976 B   ~61.5 KB, under the 64 KB opt-in limit.
    // KsPs union works because Ks is consumed entirely by QK^T (and the
    // c_frag store to Ss) before any thread writes to Ps. Bp is unpadded
    // because Ss reads in the softmax phase are row-strided and don't trip
    // the same 32-way conflict pattern as the WMMA loads of Qs / Ks / Vs.
    constexpr int Dp = D + 8;
    constexpr int Bp = (Bc > D) ? Bc : D;   // Ss holds both QK^T scores (Br x Bc) and PV partial output (Br x D); size to the larger.

    extern __shared__ unsigned char smem_raw[];
    auto Qs = reinterpret_cast<__half (*)[Dp]>(smem_raw);
    auto Ks = reinterpret_cast<__half (*)[Dp]>(reinterpret_cast<unsigned char*>(Qs) + Br * Dp * sizeof(__half));
    auto Ps = reinterpret_cast<__half (*)[Dp]>(reinterpret_cast<unsigned char*>(Ks));   // SAME memory as Ks
    auto Vs = reinterpret_cast<__half (*)[Dp]>(reinterpret_cast<unsigned char*>(Ks) + Bc * Dp * sizeof(__half));
    auto Ss = reinterpret_cast<float  (*)[Bp]>(reinterpret_cast<unsigned char*>(Vs) + Bc * Dp * sizeof(__half));
    auto Os = reinterpret_cast<float  (*)[Dp]>(reinterpret_cast<unsigned char*>(Ss) + Br * Bp * sizeof(float));
    auto m_shared = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(Os) + Br * Dp * sizeof(float));
    auto l_shared = m_shared + Br;

    const int tid          = threadIdx.x;
    const int warp_id      = tid / 32;
    const int lane         = tid % 32;
    const int row_base     = blockIdx.x * Br;
    const int warp_row_off = warp_id * WM;        // 0, 16, 32, 48

    // ---- Initialize per-row state and the running O accumulator ----
    if (tid < Br) {
        m_shared[tid] = -INFINITY;
        l_shared[tid] = 0.0f;
    }
    // Os is Br * Dp = 64 * 72 = 4608 floats, 128 threads = 36 floats per thread.
    // Pack into a simple 1D zero-loop.
    #pragma unroll
    for (int i = tid; i < Br * Dp; i += THREADS) {
        (reinterpret_cast<float*>(Os))[i] = 0.0f;
    }

    // ---- Load Q tile, cast FP32 -> FP16 ----
    // Q is [Br][D] in global; written as [Br][Dp] in SMEM (pad cols are unused).
    #pragma unroll
    for (int i = tid; i < Br * D; i += THREADS) {
        const int r = i / D, c = i % D;
        const int g_row = row_base + r;
        Qs[r][c] = (g_row < N) ? __float2half(Q[g_row * D + c]) : __float2half(0.f);
    }
    __syncthreads();

    // ---- Loop over K/V tiles ----
    const int num_tiles = (N + Bc - 1) / Bc;
    for (int t = 0; t < num_tiles; ++t) {
        const int col_base = t * Bc;

        // ---- Load K, V tiles, cast FP32 -> FP16 ----
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
        // Per warp: 1 row-block (16 rows) × 4 col-blocks (16 cols each) of S.
        // a_frag covers (warp_row_off + 0..16, k_step*16 .. k_step*16+16) of Q
        // b_frag covers (j*16..j*16+16, k_step*16..k_step*16+16) of K (= K^T col-major)
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag[N_FRAGS_C];
        #pragma unroll
        for (int j = 0; j < N_FRAGS_C; ++j) wmma::fill_fragment(s_frag[j], 0.0f);

        wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> b_frag;

        #pragma unroll
        for (int k_step = 0; k_step < D / WK; ++k_step) {
            // Load Q[warp_row_off : warp_row_off+16, k_step*16 : k_step*16+16].
            wmma::load_matrix_sync(a_frag, &Qs[warp_row_off][k_step * WK], Dp);
            #pragma unroll
            for (int j = 0; j < N_FRAGS_C; ++j) {
                // K^T[k_step*16:k_step*16+16, j*16:j*16+16] = K[j*16:j*16+16, k_step*16:k_step*16+16]
                // as col_major with leading dim = Dp (stride down a "column" of K^T == stride along D of K).
                wmma::load_matrix_sync(b_frag, &Ks[j * WN][k_step * WK], Dp);
                wmma::mma_sync(s_frag[j], a_frag, b_frag, s_frag[j]);
            }
        }

        // ---- Store scores to SMEM (FP32) ----
        #pragma unroll
        for (int j = 0; j < N_FRAGS_C; ++j) {
            wmma::store_matrix_sync(&Ss[warp_row_off][j * WN], s_frag[j], Bp, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- Per-row online softmax over the 64 scores ----
        // Each warp processes its 16 rows; lanes 0..15 take one row each
        // (lanes 16..31 idle - this is a known waste worth ~2x speedup in
        // a later pass, but keeps softmax simple for v1).
        if (lane < WM) {
            const int row = warp_row_off + lane;
            const float scale_f = rsqrtf((float)D);

            float row_max = -INFINITY;
            float scores[Bc];   // 64 FP32 - the same register-pressure issue as the
                                // FP32 kernel; for v1 we keep it simple, can chunk later.
            const int g_row = row_base + row;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                const int col = col_base + j;
                float s = Ss[row][j] * scale_f;
                if (col >= N) s = -INFINITY;
                if (causal && col > g_row) s = -INFINITY;
                scores[j] = s;
                row_max  = fmaxf(row_max, s);
            }

            const float m_old      = m_shared[row];
            const float m_new      = fmaxf(m_old, row_max);
            const float correction = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);

            // Scale running O for this row.
            #pragma unroll
            for (int k = 0; k < D; ++k) Os[row][k] *= correction;

            float l_new = l_shared[row] * correction;
            #pragma unroll
            for (int j = 0; j < Bc; ++j) {
                const float p = (scores[j] == -INFINITY) ? 0.0f
                              : expf(scores[j] - m_new);
                l_new += p;
                Ps[row][j] = __float2half(p);
            }
            m_shared[row] = m_new;
            l_shared[row] = l_new;
        }
        __syncthreads();

        // ---- PV via WMMA, accumulate into the running O ----
        // Per warp: 1 row-block (16 rows) × 4 col-blocks (D=64 / 16 = 4) of O.
        // a_frag now reads from Ps[warp_row_off..warp_row_off+16, k_step*16 .. k_step*16+16].
        // b_frag reads from Vs[k_step*16..k_step*16+16, j*16..j*16+16].
        // Note matrix_b is row_major here (V is row-major, no transpose).
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_part[N_FRAGS_O];
        #pragma unroll
        for (int j = 0; j < N_FRAGS_O; ++j) wmma::fill_fragment(o_part[j], 0.0f);

        wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> v_frag;

        #pragma unroll
        for (int k_step = 0; k_step < Bc / WK; ++k_step) {
            // Ps shares Ks's memory and is laid out with stride Dp (not Bp);
            // ldm must reflect the physical row stride of the SMEM allocation.
            wmma::load_matrix_sync(p_frag, &Ps[warp_row_off][k_step * WK], Dp);
            #pragma unroll
            for (int j = 0; j < N_FRAGS_O; ++j) {
                wmma::load_matrix_sync(v_frag, &Vs[k_step * WK][j * WN], Dp);
                wmma::mma_sync(o_part[j], p_frag, v_frag, o_part[j]);
            }
        }

        // Add o_part (partial contribution from this tile) to Os.
        // Store fragments to a temporary location in SMEM (reusing Ps which we
        // no longer need for this tile), then add to Os.
        #pragma unroll
        for (int j = 0; j < N_FRAGS_O; ++j) {
            // Reuse Ss as a scratch FP32 buffer for this warp's contribution.
            // Ss is Br x Bp FP32 - per warp we use [warp_row_off..+16, j*16..+16].
            wmma::store_matrix_sync(&Ss[warp_row_off][j * WN], o_part[j], Bp, wmma::mem_row_major);
        }
        __syncthreads();

        // Accumulate the FP32 partial into Os.
        // 128 threads, Br * D = 4096 FP32 adds.
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

} // namespace

float dense_fp16_forward(const float* Q, const float* K, const float* V,
                         float* O, const AttnConfig& cfg, cudaStream_t stream,
                         bool measure)
{
    if (cfg.d != 64 && cfg.d != 128) {
        fprintf(stderr, "[dense_fp16] d=%d not supported (only d=64 or d=128)\n", cfg.d);
        return -1.0f;
    }

    // d=64: Br=Bc=64 (4 warps/block). d=128: Br=Bc=32 (2 warps/block) so the
    // SMEM layout (Qs + Ks/Ps + Vs in FP16, Ss + Os in FP32) fits 64 KB.
    auto launch = [&](auto Br_v, auto Bc_v, auto D_v) -> float {
        constexpr int Br = Br_v.value, Bc = Bc_v.value, D = D_v.value;
        constexpr int Dp = D + 8;
        constexpr int Bp = (Bc > D) ? Bc : D;   // Ss holds both QK^T scores (Br x Bc) and PV partial output (Br x D); size to the larger.
        constexpr int NUM_WARPS = Br / 16;
        constexpr int THREADS   = NUM_WARPS * 32;
        constexpr int smem_bytes =
              Br * Dp * (int)sizeof(__half)     // Qs
            + Bc * Dp * (int)sizeof(__half)     // Ks  (Ps overlaps)
            + Bc * Dp * (int)sizeof(__half)     // Vs
            + Br * Bp * (int)sizeof(float)      // Ss
            + Br * Dp * (int)sizeof(float)      // Os
            + 2 * Br * (int)sizeof(float);      // m_shared + l_shared

        static bool smem_opted_in = false;
        if (!smem_opted_in) {
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)dense_fp16_forward_kernel<Br, Bc, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
            smem_opted_in = true;
        }

        const int blocks = (cfg.N + Br - 1) / Br;
        dim3 grid(blocks), block(THREADS);

        if (!measure) {
            dense_fp16_forward_kernel<Br, Bc, D>
                <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, cfg.N, cfg.causal);
            return 0.0f;
        }
        CudaTimer timer;
        timer.start(stream);
        dense_fp16_forward_kernel<Br, Bc, D>
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
