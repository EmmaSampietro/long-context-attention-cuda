// Public-API dispatch — switches on cfg.kernel and forwards to the per-kernel
// implementation declared in kernels_internal.h. 
// The following code provides entry points to the attention computation 
// for both single-head and multi-head cases with extensive comments 
// to explain every step. This is annotated for educational purposes.

#include <cstdio>
#include <cuda_runtime.h>

#include "attention_api.h"
#include "kernels_internal.h"
#include "utils.cuh"

namespace {

// Internal dispatch: switches on cfg.kernel and threads `measure` through.
// When measure=true, the per-kernel forward times the launch with cudaEvent
// (which forces a host sync). When measure=false, the launch is async and
// the caller is responsible for any synchronization — used by MHA so that
// per-head launches can overlap on different streams.
float attention_forward_dispatch(const float* Q, const float* K, const float* V,
                                 float* O, const AttnConfig& cfg,
                                 cudaStream_t stream, bool measure)
{
    switch (cfg.kernel) {
    case AttnKernel::Dense:           return dense_forward      (Q, K, V, O, cfg, stream, measure);
    case AttnKernel::DenseFP16:       return dense_fp16_forward (Q, K, V, O, cfg, stream, measure);
    case AttnKernel::Windowed:        return windowed_forward   (Q, K, V, O, cfg, stream, measure);
    case AttnKernel::WindowedFP16:    return windowed_fp16_forward(Q, K, V, O, cfg, stream, measure);
    case AttnKernel::BlockSparse:     return blocksparse_forward(Q, K, V, O, cfg, stream, measure);
    case AttnKernel::BlockSparseFP16: return blocksparse_fp16_forward(Q, K, V, O, cfg, stream, measure);
    }
    fprintf(stderr, "[attention_forward] unknown kernel\n");
    return -1.0f;
}

}  // namespace

// Single-head attention forward API.
// Q, K, V, O are device pointers. 'cfg' holds all attention parameters.
// 'stream' is the CUDA stream to use.
// Returns elapsed GPU time in ms (cudaEvent-measured, host-synchronous),
// or -1 on error.
float attention_forward(const float* Q, const float* K, const float* V,
                        float* O, const AttnConfig& cfg, cudaStream_t stream)
{
    return attention_forward_dispatch(Q, K, V, O, cfg, stream, /*measure=*/true);
}

// Multi-head attention forward API.
// Handles an entire batch of attention heads in parallel using separate CUDA streams.
// Q, K, V, O: row-major, shape [H, N, d], where H=head count.
// Returns total wall-clock time (ms) for all heads, or -1 on error.
float attention_forward_mha(const float* Q, const float* K, const float* V,
                            float* O, const AttnConfig& cfg)
{
    // We'll use multiple CUDA streams to process different heads in parallel.
    // This is possible because the heads are independent computations.
    constexpr int kMaxStreams = 16; // Safety: limit number of concurrent CUDA streams.
    cudaStream_t streams[kMaxStreams]; // Array of streams to use.

    // Determine how many streams to create: the lesser of head count and max allowed.
    // This way, we don't create excessive streams if model has many heads.
    const int S = (cfg.H < kMaxStreams) ? cfg.H : kMaxStreams;

    // Create CUDA streams to run attention heads in parallel.
    for (int s = 0; s < S; ++s)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // Each head works on a contiguous chunk of the Q, K, V, O arrays.
    // Compute number of elements per head: N = sequence length, d = embedding dim.
    const size_t head_elems = (size_t)cfg.N * (size_t)cfg.d;

    // Timer to measure the (wall-clock) time taken to process all heads.
    CudaTimer timer;
    timer.start(0); // Use default stream for timing.

    // Process each head independently.
    for (int h = 0; h < cfg.H; ++h) {
        // Compute base pointers for this head's Q, K, V, and O blocks.
        const float* Qh = Q + h * head_elems;
        const float* Kh = K + h * head_elems;
        const float* Vh = V + h * head_elems;
        float*       Oh = O + h * head_elems;

        // Assign this head to a CUDA stream (streams wrap around if H > S).
        cudaStream_t st = streams[h % S];

        // Prepare a config struct for a *single* head.
        // This is important so the low-level kernel runs only for one head.
        AttnConfig per_head = cfg;
        per_head.H = 1;

        // Async launch: pass measure=false so the per-head call doesn't
        // call cudaEventSynchronize (which would block the host and force
        // heads to serialize). The outer wall-clock timer still captures
        // total time across all heads after MPI_Waitall below.
        const float ms = attention_forward_dispatch(
            Qh, Kh, Vh, Oh, per_head, st, /*measure=*/false);

        // -1 still indicates a hard failure (validation / unsupported config).
        if (ms < 0.0f) {
            for (int s = 0; s < S; ++s)
                cudaStreamDestroy(streams[s]);
            return -1.0f;
        }
    }

    // Wait for all streams to finish (ensure all heads are done).
    for (int s = 0; s < S; ++s)
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));

    // Get elapsed wall time in ms for the total multi-head computation.
    const float total_ms = timer.stop(0);

    // Cleanup: destroy all streams we created.
    for (int s = 0; s < S; ++s)
        cudaStreamDestroy(streams[s]);

    // Return total wall-clock time for the multi-head attention operation.
    return total_ms;
}
