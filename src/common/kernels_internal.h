#pragma once
// Internal header — kernel implementations expose these to dispatch.cu.
// Public callers should use attention_api.h.
#include <cuda_runtime.h>
#include "attention_api.h"

// Single-head forward; returns elapsed GPU time (ms) when measure=true,
// or 0 (no host sync) when measure=false. The MHA wrapper passes measure=false
// so per-head launches don't serialize on cudaEventSynchronize, letting
// kernels on different streams overlap.
float dense_forward      (const float* Q, const float* K, const float* V,
                          float* O, const AttnConfig& cfg, cudaStream_t stream,
                          bool measure = true);

// FP16 + Tensor-Core (WMMA) dense attention. Inputs and outputs are FP32;
// the kernel casts Q/K/V to FP16 on load (and the softmax probabilities to
// FP16 before the second matmul) but keeps all softmax statistics and the
// output accumulator in FP32 for numerical stability. d=64 only, Turing+.
float dense_fp16_forward (const float* Q, const float* K, const float* V,
                          float* O, const AttnConfig& cfg, cudaStream_t stream,
                          bool measure = true);

// FP16 + Tensor-Core (WMMA) windowed attention. Same FP32 host API as the
// FP32 windowed kernel and the same windowed tile iterator; the per-tile
// matmuls run on tensor cores. Causal supported; G = 0 only.
float windowed_fp16_forward(const float* Q, const float* K, const float* V,
                            float* O, const AttnConfig& cfg, cudaStream_t stream,
                            bool measure = true);

// Seq-par WMMA variant of the windowed kernel. Q has N_q local rows; K, V
// have N_k local rows (= N_q + halos on each side). The kernel uses the
// per-tile WMMA matmul of windowed_fp16 with the local-coord window check
// |i + local_offset - j| <= w; the multi-head loop happens in the driver.
float windowed_fp16_seqpar_forward(const float* Q, const float* K, const float* V,
                                   float* O,
                                   int N_q, int N_k, int d, int w,
                                   int local_offset, bool causal,
                                   cudaStream_t stream = 0,
                                   bool measure = true);

// FP16 + Tensor-Core (WMMA) block-sparse attention. Same FP32 host API as
// the FP32 block-sparse kernel and the same CSR iterator; the per-tile
// matmuls run on tensor cores. Causal supported.
float blocksparse_fp16_forward(const float* Q, const float* K, const float* V,
                               float* O, const AttnConfig& cfg, cudaStream_t stream,
                               bool measure = true);

float windowed_forward   (const float* Q, const float* K, const float* V,
                          float* O, const AttnConfig& cfg, cudaStream_t stream,
                          bool measure = true);

float blocksparse_forward(const float* Q, const float* K, const float* V,
                          float* O, const AttnConfig& cfg, cudaStream_t stream,
                          bool measure = true);

// Sequence-parallel variant of the windowed kernel. Q is N_q local rows,
// K/V are N_k local rows including halos of width w from neighbors plus an
// optional G "global key" rows broadcast from rank 0 and concatenated at
// positions [N_k, N_k + G). The kernel translates the Q row index by
// `local_offset` (= halo_left) into the local K coordinate system.
// Q/K/V/O are per-head pointers; the multi-head loop happens in the driver.
float windowed_seqpar_forward(const float* Q, const float* K, const float* V,
                              float* O,
                              int N_q, int N_k, int d, int w,
                              int local_offset, bool causal, int G = 0,
                              cudaStream_t stream = 0,
                              bool measure = true);
