// In-process smoke test. Generates small inputs, runs the dense kernel, checks
// the output is finite and roughly the right shape (within a wide tolerance
// against a *naive on-host* reference). The big numerical check is in
// bench/validate.py against PyTorch.
//
// Exit code 0 on success, non-zero on failure.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "attention_api.h"
#include "utils.cuh"

namespace {

void cpu_dense_attention(
    const std::vector<float>& Q, const std::vector<float>& K,
    const std::vector<float>& V, std::vector<float>& O,
    int N, int D)
{
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> S(N * N), P(N * N);

    for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < D; ++k) acc += Q[i*D + k] * K[j*D + k];
        S[i*N + j] = acc * scale;
    }
    for (int i = 0; i < N; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < N; ++j) m = std::fmax(m, S[i*N + j]);
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) { P[i*N + j] = std::exp(S[i*N + j] - m); sum += P[i*N + j]; }
        for (int j = 0; j < N; ++j) P[i*N + j] /= sum;
    }
    for (int i = 0; i < N; ++i)
    for (int k = 0; k < D; ++k) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += P[i*N + j] * V[j*D + k];
        O[i*D + k] = acc;
    }
}

}  // namespace

int main() {
    constexpr int N = 128;
    constexpr int D = 64;
    constexpr int H = 1;
    const size_t per_tensor = (size_t)N * D;
    const size_t bytes = per_tensor * sizeof(float);

    std::vector<float> Q(per_tensor), K(per_tensor), V(per_tensor),
                       O_gpu(per_tensor), O_cpu(per_tensor);
    for (size_t i = 0; i < per_tensor; ++i) {
        // Tiny deterministic pattern.
        Q[i] = std::sin(0.01f * (float)i);
        K[i] = std::cos(0.011f * (float)i);
        V[i] = std::sin(0.013f * (float)i + 0.5f);
    }

    float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, bytes));
    CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes));
    CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), bytes, cudaMemcpyHostToDevice));

    AttnConfig cfg{};
    cfg.kernel = AttnKernel::Dense;
    cfg.N = N; cfg.d = D; cfg.H = H; cfg.causal = false;
    cfg.B_block = 64;

    const float ms = attention_forward(dQ, dK, dV, dO, cfg, 0);
    if (ms < 0.0f) { fprintf(stderr, "[smoke] kernel returned -1\n"); return 1; }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_gpu.data(), dO, bytes, cudaMemcpyDeviceToHost));

    cpu_dense_attention(Q, K, V, O_cpu, N, D);

    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < per_tensor; ++i) {
        if (!std::isfinite(O_gpu[i])) {
            fprintf(stderr, "[smoke] non-finite output at idx %zu\n", i);
            return 2;
        }
        const float a = O_gpu[i], b = O_cpu[i];
        const float ad = std::fabs(a - b);
        const float rd = ad / (std::fabs(b) + 1e-6f);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    const float atol = 1e-3f, rtol = 1e-3f;
    fprintf(stdout, "[smoke] N=%d d=%d  ms=%.3f  max_abs=%.3e  max_rel=%.3e\n",
            N, D, ms, max_abs, max_rel);
    if (max_abs > atol && max_rel > rtol) {
        fprintf(stderr, "[smoke] FAIL: tolerance exceeded\n");
        return 3;
    }
    fprintf(stdout, "[smoke] PASS\n");
    return 0;
}
