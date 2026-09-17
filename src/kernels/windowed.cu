// **********************************************
// Kernel B — Sliding Window Attention + Global Tokens 
// (inspired by FlashAttention fused forward).
// **********************************************
//
// This CUDA implementation fuses softmax and attention accumulation, similar
// to FlashAttention, but restricts each query block to only process a limited
// band ("window") of key-value tiles, accelerating block-sparse computation.
//
// Key concepts demonstrated here:
//   - Tiled, streaming loads of K/V
//   - Two-level window attention (regular window + "global" tokens)
//   - Fused online softmax and value accumulation (numerically stable)
//   - Per-row edge-masking, causal masking, and efficient range calculations
//   - Full block-level cooperation via shared memory
//
// This kernel is written for maximal clarity for students and researchers.
// All variables and symmetry-breaking features are heavily commented.
//
// Limitations (for simplicity!): 
//   - FP32 only, single attention head, d=64
//   - Non-causal unless explicitly requested
//   - Supports both local-windowed and global-token attention paths

#include <cstdio>
#include <cmath>
#include <type_traits>
#include <cuda_runtime.h>

#include "attention_api.h"
#include "kernels_internal.h"
#include "utils.cuh"

namespace {

// ---------------------------------------------------------------------------
// A helper function for floor-division that matches Python/NumPy, NOT C/C++'s
// division (which can lead to off-by-ones for negative denominators, e.g. when
// dealing with window boundaries).
__device__ __forceinline__ int floor_div(int a, int b) {
    int q = a / b;
    // If there is a remainder and the operands have opposite signs, decrement quotient.
    if ((a % b) != 0 && ((a ^ b) < 0)) --q;
    return q;
}

// ---------------------------------------------------------------------------
// Main sliding-windowed attention kernel.
// This kernel is heavily annotated for beginners & as a reference for block-sparse windowed attention.
//
//      Q: [N, D] (queries)
//      K: [N, D] (keys)
//      V: [N, D] (values)
//      O: [N, D] (output)
//  N: sequence length
//  w: window radius (each query attends [i-w, i+w])
//  G: number of "global" tokens, which attend everywhere and are attended by everyone
//  causal: if true, apply lower-triangular (causal) masking
//
template <int Br, int Bc, int D>
__global__ void windowed_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int w, int G, bool causal)
{
    // We'll use an extra register per vector to avoid shared memory bank conflicts.
    constexpr int Dp = D + 1;

    // Allocate shared memory for Q, K, V tiles. The layout:
    // Qs: Br rows    (for this block)
    // Ks: Bc rows    (current K/V tile)
    // Vs: Bc rows
    extern __shared__ float smem[];
    float (*Qs)[Dp] = reinterpret_cast<float (*)[Dp]>(smem);             // Block of Q's
    float (*Ks)[Dp] = reinterpret_cast<float (*)[Dp]>(smem + Br * Dp);   // Tile of K's
    float (*Vs)[Dp] = reinterpret_cast<float (*)[Dp]>(smem + (Br + Bc) * Dp); // Tile of V's

    // Each CUDA thread in the block processes one row in the block.
    const int tx       = threadIdx.x;
    const int row_base = blockIdx.x * Br;   // Start row of this block
    const int row      = row_base + tx;     // Global row for this thread
    const bool active  = (row < N);         // Mask threads outside sequence

    // Determine if this thread's row is a global row:
    // Global rows (row < G) attend everywhere, both inside and outside their window.
    const bool row_is_global = active && (row < G);

    // ---- Q Load: Each thread loads its row's Q vector into shared memory ----
    // Cooperative float4 / coalesced loader; out-of-bounds rows are zero-padded.
    load_tile_f4<Br, Br, D, Dp>(Qs, Q, row_base, N, tx);

    // ---- Per-thread softmax state ------------------------------------------
    // Initialize softmax maximum (m_i), normalization factor (l_i), and output accumulator.
    float m_i = -INFINITY;      // running max so far
    float l_i = 0.0f;           // running softmax normalization
    float O_i[D];               // running output vector (accumulated sum)
    #pragma unroll
    for (int k = 0; k < D; ++k) O_i[k] = 0.0f;

    // Softmax scale: scale dot-products by 1/sqrt(D)
    const float scale         = rsqrtf((float)D);

    // How many K/V tiles are there? Each tile has Bc columns.
    const int   num_kv_tiles  = (N + Bc - 1) / Bc;

    // ------------- TILE SELECTION LOGIC --------------------------------------
    // To increase speed, we avoid visiting all K/V, but only tiles that overlap this block's window.
    // We must also include tiles for global tokens as required.

    // 1. MAIN WINDOW: tiles whose column range intersects [row_base-w, row_base+Br-1+w]
    int local_t_lo = floor_div(row_base - w, Bc);             // First tile overlapping left window
    if (local_t_lo < 0) local_t_lo = 0;
    int local_t_hi = (row_base + Br - 1 + w) / Bc + 1;        // One past last tile overlapping window
    if (local_t_hi > num_kv_tiles) local_t_hi = num_kv_tiles;

    // 2. GLOBAL COLUMNS: tiles covering [0, G)
    int g_t_hi = (G + Bc - 1) / Bc;
    if (g_t_hi > num_kv_tiles) g_t_hi = num_kv_tiles;

    // 3. SPECIAL CASE FOR GLOBAL ROWS: If ANY row in this block is a global row,
    // we must process *all* tiles so those rows can attend everywhere.
    const bool block_has_global_row = (row_base < G);

    int t_start, t_end;
    if (block_has_global_row) {
        t_start = 0;
        t_end   = num_kv_tiles;
    } else {
        t_start = local_t_lo;
        t_end   = local_t_hi;
    }

    // ----- Causal (autoregressive): restrict tiles to those not strictly above diagonal -----
    // No row in this block can attend to col > its row.
    // For partial tiles containing the diagonal, we'll depend on a per-element mask.
    if (causal) {
        int t_end_causal = (row_base + Br - 1) / Bc + 1; // Tile index just past the diagonal
        if (t_end_causal > num_kv_tiles) t_end_causal = num_kv_tiles;
        if (t_end > t_end_causal)   t_end   = t_end_causal;
        if (g_t_hi > t_end_causal)  g_t_hi  = t_end_causal; // Clamp global-col tiles!
    }

    // ----------- Phase-based iteration over union of [0,g_t_hi) and [t_start,t_end) -----------
    // We never want to visit a tile twice, and [0,g_t_hi) and [t_start,t_end) may overlap.
    // So we decompose into two phases:
    //   1. "Global" phase: tiles 0..min(g_t_hi, t_start)
    //   2. Main window:   tiles t_start..max(g_t_hi, t_end)
    // Some of these ranges may be empty (e.g. when G==0).
    const int phase1_end   = (g_t_hi < t_start) ? g_t_hi : t_start;
    const int phase2_end   = (g_t_hi > t_end)   ? g_t_hi : t_end;
    const int phase1_count = phase1_end;                 // how many in phase 1?
    const int phase2_count = phase2_end - t_start;       // how many in phase 2?
    const int total_iters  = phase1_count + phase2_count;// sum = total tiles to process

    // -------- TILE LOOP: Stream over the selected K/V tiles for this block --------
    for (int it = 0; it < total_iters; ++it) {
        // Pick real tile index (t): either in phase 1 or phase 2
        const int t = (it < phase1_count) ? it : (t_start + (it - phase1_count));
        const int col_base = t * Bc;       // Global leftmost column in this tile

        // Cooperative load: each thread loads one row of K and one of V into shared
        // memory (Bc = Br). Float4 / coalesced loader; out-of-bounds rows are zeroed.
        load_tile_f4<Br, Bc, D, Dp>(Ks, K, col_base, N, tx);
        load_tile_f4<Br, Bc, D, Dp>(Vs, V, col_base, N, tx);
        __syncthreads(); // Ensure shared tile loaded before softmax

        // ------------ Chunked compute + online softmax (Cn=16 cols at a time) -------
        // Same math as the original full-tile pass, but processing 16 cols
        // per chunk so the per-thread score buffer shrinks from 64 floats
        // (~64 regs) to 16 floats. Saves ~48 registers/thread and lifts
        // occupancy on Turing. Cost: 4 corrections per tile instead of 1.
        if (active) {
            constexpr int Cn = 16;
            #pragma unroll
            for (int j_base = 0; j_base < Bc; j_base += Cn) {
                float s_chunk[Cn];
                #pragma unroll
                for (int jj = 0; jj < Cn; ++jj) {
                    const int  j         = j_base + jj;
                    const int  col       = col_base + j;
                    const int  dij       = row - col;
                    const int  adij      = dij < 0 ? -dij : dij;
                    const bool in_window = (adij <= w);
                    const bool in_g_col  = (col < G);
                    const bool in_bounds = (col < N);
                    const bool causal_ok = !causal || (col <= row);
                    const bool allow     = in_bounds && causal_ok
                                         && (in_window || in_g_col || row_is_global);

                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < D; ++k) acc += Qs[tx][k] * Ks[j][k];
                    s_chunk[jj] = allow ? (acc * scale) : -INFINITY;
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
        __syncthreads(); // All threads done with this tile
    }

    // ---- Write back: Normalize result or zero if all masked -----
    if (active) {
        // If all values were masked (l_i == 0), output zeros to preserve nan/inf-safety
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O[row * D + k] = O_i[k] * inv_l;
    }
} // end kernel

}  // namespace

// -----------------------------------------------------------------------------
// Host API for windowed forward kernel.
// Responsible for configuration, error checking, and managing shared memory size 
// for the kernel above. Also offers timing for benchmarking.
//
float windowed_forward(const float* Q, const float* K, const float* V,
                       float* O, const AttnConfig& cfg, cudaStream_t stream,
                       bool measure)
{
    if (cfg.d != 64 && cfg.d != 128) {
        fprintf(stderr, "[windowed_forward] d=%d not supported (only d=64 or d=128)\n", cfg.d);
        return -1.0f;
    }
    if (cfg.w < 0 || cfg.G < 0) {
        fprintf(stderr, "[windowed_forward] invalid w=%d or G=%d (must be >= 0)\n",
                cfg.w, cfg.G);
        return -1.0f;
    }

    // d=64 uses the original 64x64x64 tiling. d=128 shrinks Br=Bc=32 so that
    // (Br + 2 Bc) * Dp * 4 bytes fits the 64 KB Turing SMEM opt-in limit.
    auto launch = [&](auto Br_v, auto Bc_v, auto D_v) -> float {
        constexpr int Br = Br_v.value, Bc = Bc_v.value, D = D_v.value;
        constexpr int Dp = D + 1;
        constexpr int smem_bytes = (Br + 2 * Bc) * Dp * (int)sizeof(float);

        static bool smem_opted_in = false;
        if (!smem_opted_in) {
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)windowed_forward_kernel<Br, Bc, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
            smem_opted_in = true;
        }

        const int blocks = (cfg.N + Br - 1) / Br;
        dim3 grid(blocks), block(Br);
        if (!measure) {
            windowed_forward_kernel<Br, Bc, D>
                <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, cfg.N, cfg.w, cfg.G, cfg.causal);
            return 0.0f;
        }
        CudaTimer timer;
        timer.start(stream);
        windowed_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, cfg.N, cfg.w, cfg.G, cfg.causal);
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
// Sequence-parallel windowed kernel (Kernel B variant for MPI seq-parallel).
//
// Each rank holds a local slice of Q and a local slice of K/V with halos of
// width w from neighbors. The kernel processes the N_q local Q rows; for each
// row it attends to its local K cols inside the window. The trick is that the
// Q row index is in the LOCAL Q coordinate system, while the K col index is in
// the LOCAL K coordinate system, and they differ by halo_left (the size of the
// left halo region prepended to K). local_offset = halo_left translates Q rows
// into the K coordinate system: a query at local row r corresponds to K col
// (r + local_offset) at the diagonal.
//
// G > 0 (global keys): the host broadcasts rank 0's first G K/V rows to all
// ranks and concatenates them at positions [N_k, N_k + G) of the local K/V
// buffer. The kernel sweeps one extra tile loop over those G columns; they
// are always allowed (no window / no causal gate applied), since by Longformer
// convention global keys are visible to every query. Note: this implements
// "global keys" only. The complementary "global queries" path (rows i < G
// attending to the full sequence) requires a full Allgather of K/V to rank 0
// and a separate dense kernel pass; that is not implemented here and is noted
// as a limitation in the report. Causal masking IS supported on the windowed
// range via the same per-element check + tile-skipping clamp used in the
// single-GPU kernels, with the diagonal expressed in local coordinates as
// `col <= row + local_offset`.
// ============================================================================
//
// Parallelizes a single large sequence across multiple GPUs/ranks. Each rank
// owns a contiguous chunk of Q and a halo'd chunk of K/V (we require a window radius w on either side).
// This enables scaling up windowed attention across many devices without duplicating full sequence memory on each.
//
// Major adaptations from single-GPU case:
// - Each rank has a "local_offset" corresponding to its offset into the global K buffer
// - Each Q row sees its corresponding "diagonal" at col = local_row + local_offset
// - This kernel assumes G=0 (no global tokens; extension possible).
// - Both local halo logic and causal masking are supported.

namespace {

// -----------------------------------------------------------------------------
// Sequence-parallel kernel: each block processes a "tile" of query rows from its local partition.
// Haloing is managed by the user; this kernel does NOT do communication.
//
template <int Br, int Bc, int D>
__global__ void windowed_seqpar_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N_q, int N_k, int w, int local_offset, bool causal, int G)
{
    constexpr int Dp = D + 1;
    extern __shared__ float smem[];
    float (*Qs)[Dp] = reinterpret_cast<float (*)[Dp]>(smem);
    float (*Ks)[Dp] = reinterpret_cast<float (*)[Dp]>(smem + Br * Dp);
    float (*Vs)[Dp] = reinterpret_cast<float (*)[Dp]>(smem + (Br + Bc) * Dp);

    // Each thread processes one row in [row_base, row_base+Br)
    const int tx       = threadIdx.x;
    const int row_base = blockIdx.x * Br;
    const int row      = row_base + tx;     // Local query row index within partition
    const bool active  = (row < N_q);       // Is this thread processing in-bounds?

    // Each thread loads its query, or zero if out of bounds.
    // Cooperative float4 / coalesced loader.
    load_tile_f4<Br, Br, D, Dp>(Qs, Q, row_base, N_q, tx);

    // Per-thread softmax state (see above kernel for details)
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_i[D];
    #pragma unroll
    for (int k = 0; k < D; ++k) O_i[k] = 0.0f;

    const float scale         = rsqrtf((float)D);
    const int   num_kv_tiles  = (N_k + Bc - 1) / Bc;

    // Compute tile indices (in K) that this Q block must attend to.
    // Each local Q row (r) sees K in [r + local_offset - w, r + local_offset + w]
    int t_lo = floor_div(row_base + local_offset - w, Bc); // Leftmost overlapping tile
    if (t_lo < 0) t_lo = 0;
    int t_hi = (row_base + local_offset + Br - 1 + w) / Bc + 1; // Rightmost overlapping+1
    if (t_hi > num_kv_tiles) t_hi = num_kv_tiles;

    // Causal (autoregressive): don't process tiles *entirely past* the diagonal col 
    // for final row in this block. Row's diagonal is at local_offset + row.
    if (causal) {
        const int t_hi_causal = (row_base + Br - 1 + local_offset) / Bc + 1;
        if (t_hi_causal < t_hi) t_hi = t_hi_causal;
    }

    // ----------- Main tile streaming loop -------------
    for (int t = t_lo; t < t_hi; ++t) {
        const int col_base = t * Bc;        // Column index of start of this tile (local K)

        // Cooperative tile load (K and V): float4 / coalesced loader; out-of-bounds
        // tile rows are zeroed (so they are masked later).
        load_tile_f4<Br, Bc, D, Dp>(Ks, K, col_base, N_k, tx);
        load_tile_f4<Br, Bc, D, Dp>(Vs, V, col_base, N_k, tx);
        __syncthreads();

        // Chunked compute + online softmax (Cn=16) — see dense.cu for rationale.
        if (active) {
            constexpr int Cn = 16;
            #pragma unroll
            for (int j_base = 0; j_base < Bc; j_base += Cn) {
                float s_chunk[Cn];
                #pragma unroll
                for (int jj = 0; jj < Cn; ++jj) {
                    const int  j         = j_base + jj;
                    const int  col       = col_base + j;
                    const int  dij       = (row + local_offset) - col;
                    const int  adij      = dij < 0 ? -dij : dij;
                    const bool in_window = (adij <= w);
                    const bool in_bounds = (col < N_k);
                    const bool causal_ok = !causal || (col <= row + local_offset);
                    const bool allow     = in_bounds && in_window && causal_ok;

                    float acc = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < D; ++k) acc += Qs[tx][k] * Ks[j][k];
                    s_chunk[jj] = allow ? (acc * scale) : -INFINITY;
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
        __syncthreads(); // All done with this tile before reuse
    }

    // Global-key sweep. The G broadcast global K/V columns are stored at
    // K[N_k .. N_k + G) and V[N_k .. N_k + G). They are always allowed: no
    // window check, and (by Longformer convention) the causal gate does not
    // apply since these are typically [CLS]-style prefix tokens that all
    // queries are intended to see.
    if (G > 0) {
        const int g_tiles = (G + Bc - 1) / Bc;
        // The G globals live at K[N_k..N_k+G); offset the source pointer once
        // and reuse the same coalesced loader with row_base in [0, G).
        const float* gK = K + (size_t)N_k * D;
        const float* gV = V + (size_t)N_k * D;
        for (int gt = 0; gt < g_tiles; ++gt) {
            const int g_base = gt * Bc;           // offset within the G globals
            load_tile_f4<Br, Bc, D, Dp>(Ks, gK, g_base, G, tx);
            load_tile_f4<Br, Bc, D, Dp>(Vs, gV, g_base, G, tx);
            __syncthreads();

            // Chunked compute + online softmax (Cn=16) for the global-key tiles.
            if (active) {
                constexpr int Cn = 16;
                #pragma unroll
                for (int j_base = 0; j_base < Bc; j_base += Cn) {
                    float s_chunk[Cn];
                    #pragma unroll
                    for (int jj = 0; jj < Cn; ++jj) {
                        const int  j    = j_base + jj;
                        const bool in_g = (g_base + j < G);
                        float acc = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < D; ++k) acc += Qs[tx][k] * Ks[j][k];
                        s_chunk[jj] = in_g ? (acc * scale) : -INFINITY;
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
        }
    }

    // Normalize and write back
    if (active) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O[row * D + k] = O_i[k] * inv_l;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host interface for the sequence-parallel windowed kernel (for distributed windowed attention).
//
float windowed_seqpar_forward(const float* Q, const float* K, const float* V,
                              float* O, int N_q, int N_k, int d, int w,
                              int local_offset, bool causal, int G,
                              cudaStream_t stream, bool measure)
{
    if (d != 64 && d != 128) {
        fprintf(stderr, "[windowed_seqpar_forward] d=%d not supported (only 64 or 128)\n", d);
        return -1.0f;
    }
    if (w < 0 || local_offset < 0 || G < 0) {
        fprintf(stderr, "[windowed_seqpar_forward] invalid w=%d local_offset=%d G=%d\n",
                w, local_offset, G);
        return -1.0f;
    }

    auto launch = [&](auto Br_v, auto Bc_v, auto D_v) -> float {
        constexpr int Br = Br_v.value, Bc = Bc_v.value, D = D_v.value;
        constexpr int Dp = D + 1;
        constexpr int smem_bytes = (Br + 2 * Bc) * Dp * (int)sizeof(float);

        static bool smem_opted_in = false;
        if (!smem_opted_in) {
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)windowed_seqpar_forward_kernel<Br, Bc, D>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes));
            smem_opted_in = true;
        }
        const int blocks = (N_q + Br - 1) / Br;
        dim3 grid(blocks), block(Br);
        if (!measure) {
            windowed_seqpar_forward_kernel<Br, Bc, D>
                <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, N_q, N_k, w, local_offset, causal, G);
            return 0.0f;
        }
        CudaTimer timer;
        timer.start(stream);
        windowed_seqpar_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(Q, K, V, O, N_q, N_k, w, local_offset, causal, G);
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
