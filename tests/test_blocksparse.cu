// In-process tests for the block-sparse kernel.
//
// 1. Random masks: block-sparse at multiple rho values, compared against a CPU
//    3-pass reference that uses the same block mask expanded to element level.
// 2. rho=1.0 sanity: block-sparse with a fully-dense block mask must match the
//    dense kernel (every block active means every position attended). This is
//    the natural CUDA-vs-CUDA cross-check; it exercises the CSR iterator end
//    to end. Note that block-sparse with window_as_block_mask(w) does NOT
//    match windowed — block-level masking is coarser than the windowed
//    kernel's element-level masking.
//
// Exit code 0 on success.

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "attention_api.h"
#include "sparse_formats.h"
#include "utils.cuh"

namespace {

// Expand a (num_block_rows x num_block_cols) block mask to a per-element [N, N]
// mask. Caller passes the expected N (= num_block_rows * B, clipped).
void expand_block_mask(const std::vector<unsigned char>& block_mask,
                      int num_block_rows, int num_block_cols, int B, int N,
                      std::vector<unsigned char>& elem_mask)
{
    elem_mask.assign((size_t)N * N, 0);
    for (int br = 0; br < num_block_rows; ++br) {
        for (int bc = 0; bc < num_block_cols; ++bc) {
            if (!block_mask[(size_t)br * num_block_cols + bc]) continue;
            const int r0 = br * B, r1 = std::min(br * B + B, N);
            const int c0 = bc * B, c1 = std::min(bc * B + B, N);
            for (int i = r0; i < r1; ++i)
                for (int j = c0; j < c1; ++j)
                    elem_mask[(size_t)i * N + j] = 1;
        }
    }
}

// CPU reference: 3-pass attention with an element-wise mask.
void cpu_masked_attention(const std::vector<float>& Q, const std::vector<float>& K,
                         const std::vector<float>& V, std::vector<float>& O,
                         const std::vector<unsigned char>& mask,
                         int N, int D)
{
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> S((size_t)N * N), P((size_t)N * N);
    for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
        if (!mask[(size_t)i * N + j]) { S[(size_t)i * N + j] = -INFINITY; continue; }
        float acc = 0.0f;
        for (int k = 0; k < D; ++k) acc += Q[(size_t)i * D + k] * K[(size_t)j * D + k];
        S[(size_t)i * N + j] = acc * scale;
    }
    for (int i = 0; i < N; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < N; ++j) m = std::fmax(m, S[(size_t)i * N + j]);
        if (!std::isfinite(m)) {
            for (int j = 0; j < N; ++j) P[(size_t)i * N + j] = 0.0f;
            continue;
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            P[(size_t)i * N + j] = (S[(size_t)i * N + j] == -INFINITY)
                                   ? 0.0f : std::exp(S[(size_t)i * N + j] - m);
            sum += P[(size_t)i * N + j];
        }
        if (sum > 0.0f)
            for (int j = 0; j < N; ++j) P[(size_t)i * N + j] /= sum;
    }
    for (int i = 0; i < N; ++i)
    for (int k = 0; k < D; ++k) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += P[(size_t)i * N + j] * V[(size_t)j * D + k];
        O[(size_t)i * D + k] = acc;
    }
}

bool diff(const std::vector<float>& a, const std::vector<float>& b,
          float atol, float rtol, float& max_abs, float& max_rel)
{
    max_abs = 0.0f; max_rel = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) return false;
        const float ad = std::fabs(a[i] - b[i]);
        const float rd = ad / (std::fabs(b[i]) + 1e-6f);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
    }
    return !(max_abs > atol && max_rel > rtol);
}

void fill_sincos(std::vector<float>& v, float phase_step, float phase0) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(phase_step * (float)i + phase0);
}

bool smoke_random(int N, int B, float rho, unsigned int seed) {
    constexpr int D = 64;
    const size_t per_tensor = (size_t)N * D;
    std::vector<float> Q(per_tensor), K(per_tensor), V(per_tensor),
                       O_gpu(per_tensor), O_cpu(per_tensor);
    fill_sincos(Q, 0.010f, 0.0f);
    fill_sincos(K, 0.011f, 0.5f);
    fill_sincos(V, 0.013f, 1.0f);

    // Build random mask + expanded version for CPU reference
    const int nb = (N + B - 1) / B;
    std::vector<unsigned char> bmask((size_t)nb * nb, 0);
    {
        // Same LCG as sparse_formats.cu::random_block_mask for determinism
        uint64_t s = (uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL;
        for (int r = 0; r < nb; ++r)
            for (int c = 0; c < nb; ++c) {
                s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                const float u = (float)((s >> 32) / (double)(uint64_t)0xFFFFFFFFULL);
                if (u < rho) bmask[(size_t)r * nb + c] = 1;
            }
    }
    std::vector<unsigned char> emask;
    expand_block_mask(bmask, nb, nb, B, N, emask);

    cpu_masked_attention(Q, K, V, O_cpu, emask, N, D);

    BlockCSR csr = block_csr_from_dense_mask(bmask.data(), nb, nb, B);

    float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr;
    const size_t bytes = per_tensor * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dQ, bytes));
    CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes));
    CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), bytes, cudaMemcpyHostToDevice));

    AttnConfig cfg{};
    cfg.kernel = AttnKernel::BlockSparse;
    cfg.N = N; cfg.d = D; cfg.H = 1; cfg.causal = false;
    cfg.B_block = B;
    cfg.block_row_ptr = csr.row_ptr;
    cfg.block_col_idx = csr.col_idx;
    cfg.nnz_blocks = csr.nnz_blocks;

    const float ms = attention_forward(dQ, dK, dV, dO, cfg, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_gpu.data(), dO, bytes, cudaMemcpyDeviceToHost));

    float max_abs, max_rel;
    const bool ok = diff(O_gpu, O_cpu, 1e-3f, 1e-3f, max_abs, max_rel);
    fprintf(stdout,
            "[smoke-bsp random] N=%d B=%d rho=%.2f nnz=%d  ms=%.3f  "
            "max_abs=%.3e  max_rel=%.3e  %s\n",
            N, B, rho, csr.nnz_blocks, ms, max_abs, max_rel, ok ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    block_csr_free(csr);
    return ok;
}

// Block-sparse with a fully-dense block mask must match the dense kernel: every
// block is active, so every (i,j) is attended. This is a CUDA-vs-CUDA check
// that exercises the CSR iterator end-to-end.
bool cross_validate_dense(int N, int B) {
    constexpr int D = 64;
    const size_t per_tensor = (size_t)N * D;
    std::vector<float> Q(per_tensor), K(per_tensor), V(per_tensor),
                       O_dense(per_tensor), O_bsp(per_tensor);
    fill_sincos(Q, 0.012f, 0.2f);
    fill_sincos(K, 0.014f, 0.7f);
    fill_sincos(V, 0.009f, 1.3f);

    float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr;
    const size_t bytes = per_tensor * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dQ, bytes));
    CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes));
    CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), bytes, cudaMemcpyHostToDevice));

    // (a) Dense kernel.
    {
        AttnConfig cfg{};
        cfg.kernel = AttnKernel::Dense;
        cfg.N = N; cfg.d = D; cfg.H = 1; cfg.causal = false;
        const float ms = attention_forward(dQ, dK, dV, dO, cfg, 0);
        if (ms < 0.0f) {
            cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); return false;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(O_dense.data(), dO, bytes, cudaMemcpyDeviceToHost));
    }

    // (b) Block-sparse with full block mask (rho = 1.0).
    const int nb = (N + B - 1) / B;
    std::vector<unsigned char> bmask((size_t)nb * nb, 1);
    BlockCSR csr = block_csr_from_dense_mask(bmask.data(), nb, nb, B);
    {
        AttnConfig cfg{};
        cfg.kernel = AttnKernel::BlockSparse;
        cfg.N = N; cfg.d = D; cfg.H = 1; cfg.causal = false;
        cfg.B_block = B;
        cfg.block_row_ptr = csr.row_ptr;
        cfg.block_col_idx = csr.col_idx;
        cfg.nnz_blocks = csr.nnz_blocks;
        const float ms = attention_forward(dQ, dK, dV, dO, cfg, 0);
        if (ms < 0.0f) {
            cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
            block_csr_free(csr); return false;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(O_bsp.data(), dO, bytes, cudaMemcpyDeviceToHost));
    }
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    float max_abs, max_rel;
    const bool ok = diff(O_bsp, O_dense, 1e-3f, 1e-3f, max_abs, max_rel);
    fprintf(stdout,
            "[cross-bsp full=dense] N=%d B=%d nnz=%d  "
            "max_abs=%.3e  max_rel=%.3e  %s\n",
            N, B, csr.nnz_blocks, max_abs, max_rel, ok ? "PASS" : "FAIL");

    block_csr_free(csr);
    return ok;
}

}  // namespace

int main() {
    int failures = 0;

    // Random masks: cover small / medium N, a couple of densities.
    if (!smoke_random(128, 64, 0.25f, 42)) ++failures;
    if (!smoke_random(128, 64, 0.50f, 17)) ++failures;
    if (!smoke_random(256, 64, 0.10f,  7)) ++failures;

    // Cross-validation: full block mask must equal dense.
    if (!cross_validate_dense(128, 64)) ++failures;
    if (!cross_validate_dense(256, 64)) ++failures;

    if (failures > 0) {
        fprintf(stderr, "[smoke-bsp] %d / 5 cases FAILED\n", failures);
        return 3;
    }
    fprintf(stdout, "[smoke-bsp] PASS (5 cases)\n");
    return 0;
}
