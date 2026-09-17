#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//
// ============================================================================
// CUDA_CHECK macro
// ============================================================================
// This macro wraps every CUDA runtime API call for error checking.
// If the CUDA function returns an error, the macro will:
//   - Print a descriptive error message to stderr, including:
//       - the failing expression
//       - the current source file (__FILE__)
//       - the failing line number (__LINE__)
//       - the human-readable error string from cudaGetErrorString
//   - Then abort the program (via std::abort), ensuring that errors never go undetected.
//
// Usage:
//   CUDA_CHECK(cudaMalloc(...));
//   CUDA_CHECK(cudaMemcpy(...));
// Any failure (e.g. out-of-memory, illegal access, etc) is immediately and clearly reported.
//
#define CUDA_CHECK(x) do {                                                \
    cudaError_t _err = (x);                                               /* Evaluate the CUDA expression and store the error code. */           \
    if (_err != cudaSuccess) {                                            /* If the call did not succeed...  */                                 \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   /* Print the error (macro will show which call failed and details). */ \
                #x, __FILE__, __LINE__, cudaGetErrorString(_err));        \
        std::abort();                                                     /* Terminate the program. */                                          \
    }                                                                     \
} while (0)

//
// ============================================================================
// CudaTimer class
// ============================================================================
// A simple utility to measure elapsed GPU time (in ms) using CUDA events.
// This is an extremely useful tool for performance profiling.
//
// CUDA events are GPU-side timing primitives that:
//   - Can synchronize to a stream (they track when GPU work is done).
//   - Provide microsecond accuracy (reported in milliseconds).
//   - Do not measure CPU-side scheduling, only GPU time between 'start' and 'stop'.
//
// Typical Usage Pattern:
//   CudaTimer timer;
//   timer.start();     // Record the start event (can pass stream to time only operations in that stream)
//     ... launch kernels ...
//   float ms = timer.stop();   // Record stop event, sync, return elapsed time (waits for the kernels to complete).
//
class CudaTimer {
    cudaEvent_t start_, stop_;  // CUDA events for marking the start and end points for measurement.
public:
    // Constructor: create the start and stop events.
    // These are CUDA resources allocated on construction and freed on destruction.
    CudaTimer()  {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    // Destructor: clean up event resources
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    // start:
    //   Records a CUDA event at the current point in 'stream' (default: stream 0).
    //   This marks the beginning of your timed region.
    //   By using CUDA streams, you can time asynchronous execution and overlap.
    void  start(cudaStream_t s = 0) {
        cudaEventRecord(start_, s);
    }

    // stop:
    //   Records a CUDA event at the current point in 'stream' (default: stream 0).
    //   Then, synchronizes with the event (waits for all prior GPU work in that stream to finish).
    //   Measures and returns the elapsed time (in milliseconds) between start and stop events.
    //   Note: This will synchronize the CPU thread until all GPU work is done, so total wall time is measured.
    float stop (cudaStream_t s = 0) {
        cudaEventRecord(stop_, s);          // Mark the end event in the stream.
        cudaEventSynchronize(stop_);        // Wait until the stop event has actually occurred (i.e., all prior GPU work is done).
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);  // Measure elapsed time from start_ to stop_.
        return ms;
    }
};

// Coalesced cooperative load of a [TileRows, D] source slab into a shared-memory
// tile Ks[TileRows][Dp] using float4. Each thread loads ONE float4 (16 B) per
// outer iteration, so the warp issues a 512 B contiguous request that lands
// on aligned cache lines (vs. the original scalar pattern, where consecutive
// threads were a full row apart and burned ~88% of memory sectors).
//
// Preconditions:
//   - D and TileRows are multiples of 4.
//   - BlockThreads >= D/4 and BlockThreads is a multiple of D/4.
//   - The source pointer S has D*4-byte-aligned rows (cudaMalloc gives 256 B).
//
// Out-of-range rows (src_row >= N_src) are zero-filled, which matches the
// padding behaviour of the original scalar loads.
template <int BlockThreads, int TileRows, int D, int Dp>
__device__ __forceinline__ void load_tile_f4(
    float (*Ks)[Dp], const float* __restrict__ S,
    int row_base, int N_src, int tx)
{
    constexpr int F   = 4;
    constexpr int LpR = D / F;             // float4s per row (16 when D=64)
    constexpr int RpL = BlockThreads / LpR; // rows loaded per outer iteration
    const int t_row = tx / LpR;
    const int t_col = (tx % LpR) * F;
    #pragma unroll
    for (int r_base = 0; r_base < TileRows; r_base += RpL) {
        const int r       = r_base + t_row;
        const int src_row = row_base + r;
        if (src_row < N_src) {
            const float4 v = *reinterpret_cast<const float4*>(
                &S[src_row * D + t_col]);
            Ks[r][t_col + 0] = v.x;
            Ks[r][t_col + 1] = v.y;
            Ks[r][t_col + 2] = v.z;
            Ks[r][t_col + 3] = v.w;
        } else {
            Ks[r][t_col + 0] = 0.0f;
            Ks[r][t_col + 1] = 0.0f;
            Ks[r][t_col + 2] = 0.0f;
            Ks[r][t_col + 3] = 0.0f;
        }
    }
}
