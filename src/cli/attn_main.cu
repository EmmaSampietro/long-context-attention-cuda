// *** EDUCATIONAL FULLY COMMENTED MINIMAL CLI DRIVER FOR ATTENTION BENCHMARKING ***
//
// This program serves as a minimal benchmark and data I/O driver for various attention kernels
// (dense, windowed, and block-sparse). It loads input tensors (Q, K, V) from a binary file or
// generates them deterministically from a PRNG, runs the requested attention kernel via
// `attention_forward_mha`, writes the result tensor (O) to a binary file, and finally
// prints a one-line CSV-style summary of the run including timing stats and config summary.
//
// --- Input/output binary layout for the CLI ---
// * --in:  Format is contiguous float32 stream -- [ Q (H*N*d), K (H*N*d), V (H*N*d) ]
// * --out: Format is contiguous float32 stream -- [ O (H*N*d) ]
//
// Major structures, parse helpers, input generators, and file I/O follow.

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

#include "attention_api.h"      // Exposes kernel enums, config struct, and run function.
#include "sparse_formats.h"     // Exposes helpers for block-sparse masks.
#include "utils.cuh"            // CUDA error wrapper and misc utilities.

// Put all implementation details in an anonymous namespace (C++ best practice for file-locality)
namespace {

// Arguments structure with all supported CLI switches, with defaults for quick experimentation.
// These variables control: kernel type, tensor sizes, window/block settings, and file I/O.
struct Args {
    std::string kernel = "dense";    // Kernel variant: "dense", "windowed", or "blocksparse"
    int  N = 1024;                   // Sequence length (tokens)
    int  d = 64;                     // Per-head dimensionality
    int  H = 1;                      // Number of attention heads
    int  w = 128;                    // Windowed: half-width for attention
    int  G = 0;                      // Windowed: count of global tokens (always attend everywhere)
    float rho = 0.1f;                // Block-sparse: probability of any block present (density)
    int  B_block = 64;               // Block-sparse: size of each block in one dimension
    std::string mask = "random";     // Block-sparse: "random" or "window" block mask pattern
    bool causal = false;             // Whether to apply causal masking (left-to-right only)
    int  warmup = 5;                 // Number of unmeasured warmup iterations
    int  iters  = 10;                // Number of measured iterations (median/min/max reported)
    uint64_t seed = 0;               // PRNG seed for reproducibility
    std::string in_path  = "";       // Optional: path to binary input file (for Q/K/V)
    std::string out_path = "";       // Optional: path to binary output file (for O)
};

// === Simple argument parsing helpers for "--name=value" style CLI switches ===

// Generic key=value parser for string types. Returns true if "arg" matches expected key pattern.
bool parse_kv(const char* arg, const std::string& key, std::string& out) {
    const std::string a(arg);
    const std::string pre = "--" + key + "=";
    if (a.rfind(pre, 0) == 0) { out = a.substr(pre.size()); return true; }
    return false;
}
// Parses CLI "name" as integer value of any type, stores to out.
template <typename T>
bool parse_int(const char* arg, const std::string& key, T& out) {
    std::string s; if (!parse_kv(arg, key, s)) return false;
    out = (T)std::stoll(s);
    return true;
}
// Parse CLI "name" as float value.
bool parse_float(const char* arg, const std::string& key, float& out) {
    std::string s; if (!parse_kv(arg, key, s)) return false;
    out = std::stof(s); return true;
}
// Parse CLI "name" as a boolean ("1"/"true"/"True" = true).
bool parse_bool(const char* arg, const std::string& key, bool& out) {
    std::string s; if (!parse_kv(arg, key, s)) return false;
    out = (s == "1" || s == "true" || s == "True");
    return true;
}

// Parses all supported CLI arguments and returns an Args struct.
// Exits with error for unknown switches. Parsing style: --name=value for all.
Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const char* x = argv[i];
        if (parse_kv  (x, "kernel", a.kernel)) continue;
        if (parse_int (x, "N",      a.N))      continue;
        if (parse_int (x, "d",      a.d))      continue;
        if (parse_int (x, "H",      a.H))      continue;
        if (parse_int (x, "w",      a.w))      continue;
        if (parse_int (x, "G",      a.G))      continue;
        if (parse_float(x, "rho",   a.rho))    continue;
        if (parse_int (x, "B",      a.B_block)) continue;       // block size
        if (parse_kv   (x, "mask",  a.mask))   continue;
        if (parse_bool(x, "causal", a.causal)) continue;
        if (parse_int (x, "warmup", a.warmup)) continue;
        if (parse_int (x, "iters",  a.iters))  continue;
        if (parse_int (x, "seed",   a.seed))   continue;
        if (parse_kv  (x, "in",     a.in_path))  continue;
        if (parse_kv  (x, "out",    a.out_path)) continue;
        // Unknown argument detected: print message and exit.
        fprintf(stderr, "unknown arg: %s\n", x);
        std::exit(1);
    }
    return a;
}

// Helper to map kernel name string to API enum (exits if unknown)
AttnKernel kernel_from_name(const std::string& s) {
    if (s == "dense")            return AttnKernel::Dense;
    if (s == "dense_fp16")       return AttnKernel::DenseFP16;
    if (s == "windowed")         return AttnKernel::Windowed;
    if (s == "windowed_fp16")    return AttnKernel::WindowedFP16;
    if (s == "blocksparse")      return AttnKernel::BlockSparse;
    if (s == "blocksparse_fp16") return AttnKernel::BlockSparseFP16;
    fprintf(stderr, "unknown kernel: %s\n", s.c_str());
    std::exit(1);
}

// === Input generation: PRNG fallback if input file not used ===
// Tiny deterministic LCG (linear congruential) RNG for seed-deterministic output on host.
// Generates values in [-1, 1) suitable for Q, K, V inputs.
void fill_lcg(std::vector<float>& v, uint64_t seed) {
    // The LCG parameters are chosen for good randomness quality.
    uint64_t s = (seed * 6364136223846793005ULL) + 1442695040888963407ULL;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        // Upper 32 bits: map to float in [-1, 1).
        v[i] = (float)((int64_t)(s >> 32) / (double)(1LL << 31));
    }
}

// === Safe file I/O helpers with error checking ===
// Reads file contents into buf, exits if file or read is bad.
void read_file(const std::string& path, void* buf, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s for read\n", path.c_str()); std::exit(2); }
    if (std::fread(buf, 1, bytes, f) != bytes) {
        fprintf(stderr, "short read from %s\n", path.c_str()); std::exit(2);
    }
    std::fclose(f);
}
// Writes buf to file, verifies success.
void write_file(const std::string& path, const void* buf, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s for write\n", path.c_str()); std::exit(2); }
    if (std::fwrite(buf, 1, bytes, f) != bytes) {
        fprintf(stderr, "short write to %s\n", path.c_str()); std::exit(2);
    }
    std::fclose(f);
}

}  // end anonymous namespace

// === MAIN: End-to-End CLI Orchestration ===
int main(int argc, char** argv) {
    // Parse CLI arguments. This sets runtime config, file paths, kernel kind, etc.
    Args a = parse(argc, argv);

    // === Create and fill attention kernel config structure ===
    // All kernel-required metadata (tensor sizes, kernel variant, mask/csr pointers, etc)
    AttnConfig cfg{};
    cfg.kernel  = kernel_from_name(a.kernel); // Choose which kernel variant to run.
    cfg.N       = a.N;        // Sequence length
    cfg.d       = a.d;        // Per-head dimension
    cfg.H       = a.H;        // Count of heads
    cfg.causal  = a.causal;   // Whether to use causal masking
    cfg.w       = a.w;        // Windowed: half-width
    cfg.G       = a.G;        // Windowed: global tokens
    cfg.B_block = a.B_block;  // Sparse: block size
    // Block mask: These remain null for dense and windowed kernels.
    cfg.block_row_ptr = nullptr; 
    cfg.block_col_idx = nullptr;
    cfg.nnz_blocks = 0;       // Sparse only: nonzero block count

    // Baseline free memory before any user allocations on device. Forces the
    // CUDA context to initialize via a no-op so its overhead is not counted
    // in our footprint. Everything after this (BlockCSR, Q/K/V/O buffers,
    // any kernel-side workspace) goes into peak_mb below.
    CUDA_CHECK(cudaFree(nullptr));
    size_t free_baseline = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_baseline, &total_mem));

    // --- If block-sparse mode, generate and own a CSR-format sparsity mask. ---
    // This configures which blocks are present/zero for block-sparse matmul. The
    // CSR device buffers are owned here; freed before exit.
    //   --mask=random — i.i.d. block mask with density rho (for benchmarks)
    //   --mask=window — block-aligned window pattern at half-width w; matches
    //                   the windowed kernel exactly, used for cross-validation
    BlockCSR bs_csr{};    // Helper struct to own block mask device buffers.
    if (cfg.kernel == AttnKernel::BlockSparse ||
        cfg.kernel == AttnKernel::BlockSparseFP16) {
        if (a.mask == "window") {
            // Generate block-aligned window mask.
            bs_csr = window_as_block_mask(a.N, a.B_block, a.w, a.causal);
        } else {
            // Generate a random (Bernoulli) block mask at density a.rho.
            bs_csr = random_block_mask(a.N, a.B_block, a.rho, (unsigned int)a.seed);
        }
        cfg.block_row_ptr = bs_csr.row_ptr;
        cfg.block_col_idx = bs_csr.col_idx;
        cfg.nnz_blocks    = bs_csr.nnz_blocks; // Important for reporting and kernel use.
    }

    // === Compute total number of elements in a Q/K/V/O tensor ===
    // All four tensors share identical shapes: (H x N x d)
    const size_t per_tensor = (size_t)cfg.H * cfg.N * cfg.d;
    const size_t bytes_per  = per_tensor * sizeof(float);

    // Allocate host vectors for Q, K, V, and to hold O output.
    std::vector<float> hQ(per_tensor), hK(per_tensor), hV(per_tensor), hO(per_tensor);

    // === Input loading/generation for Q, K, V ===
    if (!a.in_path.empty()) {
        // Input path provided: read Q, K, V tensors from binary file at once (contiguously).
        std::vector<float> blob(3 * per_tensor);
        read_file(a.in_path, blob.data(), 3 * bytes_per);
        // Slice out Q, K, V from contiguous blob (layout explained at top).
        std::memcpy(hQ.data(), blob.data() + 0 * per_tensor, bytes_per);
        std::memcpy(hK.data(), blob.data() + 1 * per_tensor, bytes_per);
        std::memcpy(hV.data(), blob.data() + 2 * per_tensor, bytes_per);
    } else {
        // No input file: Generate deterministic inputs for Q, K, V via host-side LCG per tensor.
        // Using (seed+X) ensures Q/K/V have distinct but reproducible data.
        fill_lcg(hQ, a.seed + 1);
        fill_lcg(hK, a.seed + 2);
        fill_lcg(hV, a.seed + 3);
    }

    // === Allocate device buffers for all four tensors (Q, K, V, O) ===
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    // GPU-side Q, K, V, O buffers, one per tensor, allocated for total elements.
    CUDA_CHECK(cudaMalloc(&dQ, bytes_per));
    CUDA_CHECK(cudaMalloc(&dK, bytes_per));
    CUDA_CHECK(cudaMalloc(&dV, bytes_per));
    CUDA_CHECK(cudaMalloc(&dO, bytes_per));
    // Upload Q/K/V from host to device, O will be written by the kernel.
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), bytes_per, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), bytes_per, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), bytes_per, cudaMemcpyHostToDevice));

    // === Warmup: perform several unmeasured kernel iterations to stabilize CUDA runtime/timing ===
    for (int i = 0; i < a.warmup; ++i) {
        const float ms = attention_forward_mha(dQ, dK, dV, dO, cfg);
        if (ms < 0.0f) { 
            // Kernel errors are expected to report -1
            fprintf(stderr, "kernel returned -1\n"); 
            return 3; 
        }
    }
    // Ensure all GPU work complete before measurement phase. Helps stabilize timing.
    CUDA_CHECK(cudaDeviceSynchronize());

    // === Benchmark: timed iterations, storing milliseconds per run ===
    std::vector<float> times(a.iters);
    for (int i = 0; i < a.iters; ++i) {
        // Each call runs one end-to-end forward pass and should return wallclock ms.
        times[i] = attention_forward_mha(dQ, dK, dV, dO, cfg);
    }
    // Wait for all launches to finish before further timing stats/use of outputs.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sample free memory after kernels have run so any lazy kernel-side
    // allocations (stream pool, etc.) are included in the peak.
    size_t free_peak = free_baseline;
    CUDA_CHECK(cudaMemGetInfo(&free_peak, &total_mem));
    const double peak_mb = (double)(free_baseline - free_peak) / (1024.0 * 1024.0);

    // === Compute statistics: median, min, and max runtime (in milliseconds) ===
    std::vector<float> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    // Median: robust to outliers, preferred for reporting (sorted middle element)
    const float ms_median = sorted[sorted.size() / 2];

    // === Download output tensor O from device to host ===
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, bytes_per, cudaMemcpyDeviceToHost));
    // Optionally, write O output to file for correctness runs/validation.
    if (!a.out_path.empty()) {
        write_file(a.out_path, hO.data(), bytes_per);
    }

    // === Print summary line for harness: fast machine readability ===
    // CSV-friendly string. For block-sparse, includes actual achieved block density and nnz.
    if (cfg.kernel == AttnKernel::BlockSparse ||
        cfg.kernel == AttnKernel::BlockSparseFP16) {
        // Calculate block grid (nb x nb) and actual density
        const int nb = (a.N + a.B_block - 1) / a.B_block;
        const float achieved_rho = (float)cfg.nnz_blocks / (float)(nb * nb);
        // Print full config including blocksparse details, timing summary (median/min/max ms)
        fprintf(stdout,
                "kernel=%s N=%d d=%d H=%d B=%d rho=%.4f nnz=%d causal=%d "
                "ms_median=%.4f ms_min=%.4f ms_max=%.4f mem_mb=%.2f\n",
                a.kernel.c_str(), a.N, a.d, a.H, a.B_block, achieved_rho,
                cfg.nnz_blocks, (int)a.causal,
                ms_median, sorted.front(), sorted.back(), peak_mb);
    } else {
        // For dense/windowed, print appropriate configuration
        fprintf(stdout,
                "kernel=%s N=%d d=%d H=%d w=%d G=%d causal=%d "
                "ms_median=%.4f ms_min=%.4f ms_max=%.4f mem_mb=%.2f\n",
                a.kernel.c_str(), a.N, a.d, a.H, a.w, a.G, (int)a.causal,
                ms_median, sorted.front(), sorted.back(), peak_mb);
    }

    // === Cleanup: free all device memory and mask/csr device buffers if needed ===
    cudaFree(dQ); 
    cudaFree(dK); 
    cudaFree(dV); 
    cudaFree(dO);
    if (cfg.kernel == AttnKernel::BlockSparse ||
        cfg.kernel == AttnKernel::BlockSparseFP16) block_csr_free(bs_csr);
    // Success exit
    return 0;
}
