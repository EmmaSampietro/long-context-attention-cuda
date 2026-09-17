// In-process smoke test for the windowed kernel. Mirrors test_dense.cu:
// generates small deterministic inputs, runs the GPU kernel, compares against
// a naive on-host CPU reference within a wide tolerance. The deeper numerical
// check is in bench/validate.py against the numpy reference.
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

// Naive 3-pass CPU windowed attention. Same masking rule as the kernel:
//   allow(i, j) := (|i - j| <= w) || (j < G) || (i < G)
void cpu_windowed_attention(
    const std::vector<float>& Q, const std::vector<float>& K,
    const std::vector<float>& V, std::vector<float>& O,
    int N, int D, int w, int G)
{
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> S(N * N), P(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const bool allow = (std::abs(i - j) <= w) || (j < G) || (i < G);
            if (!allow) {
                S[i * N + j] = -INFINITY;
                continue;
            }
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) acc += Q[i * D + k] * K[j * D + k];
            S[i * N + j] = acc * scale;
        }
    }
    for (int i = 0; i < N; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < N; ++j) m = std::fmax(m, S[i * N + j]);
        if (!std::isfinite(m)) {
            // Fully-masked row (impossible for any w>=0, but be defensive).
            for (int j = 0; j < N; ++j) P[i * N + j] = 0.0f;
            continue;
        }
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            P[i * N + j] = (S[i * N + j] == -INFINITY) ? 0.0f
                                                       : std::exp(S[i * N + j] - m);
            sum += P[i * N + j];
        }
        if (sum > 0.0f) for (int j = 0; j < N; ++j) P[i * N + j] /= sum;
    }
    for (int i = 0; i < N; ++i)
    for (int k = 0; k < D; ++k) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += P[i * N + j] * V[j * D + k];
        O[i * D + k] = acc;
    }
}

bool run_one(int N, int D, int w, int G) {
    const size_t per_tensor = (size_t)N * D;
    const size_t bytes = per_tensor * sizeof(float);

    std::vector<float> Q(per_tensor), K(per_tensor), V(per_tensor),
                       O_gpu(per_tensor), O_cpu(per_tensor);
    for (size_t i = 0; i < per_tensor; ++i) {
        Q[i] = std::sin(0.01f  * (float)i);
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
    cfg.kernel = AttnKernel::Windowed;
    cfg.N = N; cfg.d = D; cfg.H = 1; cfg.causal = false;
    cfg.w = w; cfg.G = G; cfg.B_block = 64;

    const float ms = attention_forward(dQ, dK, dV, dO, cfg, 0);
    if (ms < 0.0f) {
        fprintf(stderr, "[smoke-windowed] kernel returned -1 at N=%d w=%d G=%d\n", N, w, G);
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
        return false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O_gpu.data(), dO, bytes, cudaMemcpyDeviceToHost));

    cpu_windowed_attention(Q, K, V, O_cpu, N, D, w, G);

    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < per_tensor; ++i) {
        if (!std::isfinite(O_gpu[i])) {
            fprintf(stderr, "[smoke-windowed] non-finite output at idx %zu (N=%d w=%d G=%d)\n",
                    i, N, w, G);
            cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
            return false;
        }
        const float a = O_gpu[i], b = O_cpu[i];
        const float ad = std::fabs(a - b);
        const float rd = ad / (std::fabs(b) + 1e-6f);
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
    }
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    const float atol = 1e-3f, rtol = 1e-3f;
    const bool ok = !(max_abs > atol && max_rel > rtol);
    fprintf(stdout,
            "[smoke-windowed] N=%d d=%d w=%d G=%d  ms=%.3f  max_abs=%.3e  max_rel=%.3e  %s\n",
            N, D, w, G, ms, max_abs, max_rel, ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    constexpr int D = 64;
    // Coverage: tight window, medium window, w >= N (dense-equivalent shape),
    // and a G > 0 case to exercise the global-token path.
    struct Case { int N, w, G; };
    Case cases[] = {
        {128,  16,   0},
        {128,  64,   0},
        {128, 256,   0},   // w >= N → dense-equivalent
        {128,  16,   8},
        {256,  32,   0},
    };
    int failures = 0;
    for (const auto& c : cases) {
        if (!run_one(c.N, D, c.w, c.G)) ++failures;
    }
    if (failures > 0) {
        fprintf(stderr, "[smoke-windowed] %d / %zu cases FAILED\n",
                failures, sizeof(cases) / sizeof(cases[0]));
        return 3;
    }
    fprintf(stdout, "[smoke-windowed] PASS (%zu cases)\n",
            sizeof(cases) / sizeof(cases[0]));
    return 0;
}
