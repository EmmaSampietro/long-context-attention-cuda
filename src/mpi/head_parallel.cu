// MPI head-parallel attention forward.
//
// Strategy: heads are independent, so we split them across ranks (one GPU per
// rank). Rank 0 owns the full Q/K/V/O tensors on its device; an MPI_Scatter
// sends each rank its slice of heads, each rank runs attention_forward_mha
// locally on H/P heads, and an MPI_Gather collects the per-rank outputs back
// into rank 0's full O buffer. All MPI calls use device pointers (CUDA-aware
// MPI confirmed on the ICME cluster, nvhpc/24.1).
//
// Compile-time constraint: H must be divisible by the number of ranks.
//
// CLI mirrors src/cli/attn_main.cu — same flags, plus the binary is launched
// under mpirun:
//   mpirun -np 4 build/attn_mpi_headpar --kernel=dense --N=2048 --H=16 ...

#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

#include "attention_api.h"
#include "utils.cuh"
#include "sparse_formats.h"

namespace {

struct Args {
    std::string kernel = "dense";
    int  N = 1024;
    int  d = 64;
    int  H = 4;
    int  w = 128;
    int  G = 0;
    bool causal = false;
    int  warmup = 5;
    int  iters  = 10;
    uint64_t seed = 0;
    // Block-sparse parameters; only consulted when kernel ∈ {blocksparse, blocksparse_fp16}.
    float rho   = 0.10f;
    int   B_block = 64;
    std::string in_path  = "";
    std::string out_path = "";
};

bool parse_kv(const char* arg, const std::string& key, std::string& out) {
    const std::string a(arg);
    const std::string pre = "--" + key + "=";
    if (a.rfind(pre, 0) == 0) { out = a.substr(pre.size()); return true; }
    return false;
}
template <typename T>
bool parse_int(const char* arg, const std::string& key, T& out) {
    std::string s; if (!parse_kv(arg, key, s)) return false;
    out = (T)std::stoll(s);
    return true;
}
bool parse_bool(const char* arg, const std::string& key, bool& out) {
    std::string s; if (!parse_kv(arg, key, s)) return false;
    out = (s == "1" || s == "true" || s == "True");
    return true;
}

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
        if (parse_bool(x, "causal", a.causal)) continue;
        if (parse_int (x, "warmup", a.warmup)) continue;
        if (parse_int (x, "iters",  a.iters))  continue;
        if (parse_int (x, "seed",   a.seed))   continue;
        if (parse_int (x, "B",      a.B_block)) continue;
        // rho is a float; accept it via the kv-string parser.
        { std::string s; if (parse_kv(x, "rho", s)) { a.rho = std::stof(s); continue; } }
        if (parse_kv  (x, "in",     a.in_path))  continue;
        if (parse_kv  (x, "out",    a.out_path)) continue;
        fprintf(stderr, "unknown arg: %s\n", x);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return a;
}

AttnKernel kernel_from_name(const std::string& s) {
    if (s == "dense")            return AttnKernel::Dense;
    if (s == "dense_fp16")       return AttnKernel::DenseFP16;
    if (s == "windowed")         return AttnKernel::Windowed;
    if (s == "windowed_fp16")    return AttnKernel::WindowedFP16;
    if (s == "blocksparse")      return AttnKernel::BlockSparse;
    if (s == "blocksparse_fp16") return AttnKernel::BlockSparseFP16;
    fprintf(stderr, "unknown kernel: %s\n", s.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return AttnKernel::Dense;
}

void fill_lcg(std::vector<float>& v, uint64_t seed) {
    uint64_t s = (seed * 6364136223846793005ULL) + 1442695040888963407ULL;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = (float)((int64_t)(s >> 32) / (double)(1LL << 31));
    }
}

void read_file(const std::string& path, void* buf, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s for read\n", path.c_str());
              MPI_Abort(MPI_COMM_WORLD, 2); }
    if (std::fread(buf, 1, bytes, f) != bytes) {
        fprintf(stderr, "short read from %s\n", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    std::fclose(f);
}
void write_file(const std::string& path, const void* buf, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s for write\n", path.c_str());
              MPI_Abort(MPI_COMM_WORLD, 2); }
    if (std::fwrite(buf, 1, bytes, f) != bytes) {
        fprintf(stderr, "short write to %s\n", path.c_str());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args a = parse(argc, argv);

    // Heads must distribute evenly across ranks (head-parallel design choice).
    if (a.H % size != 0) {
        if (rank == 0) fprintf(stderr,
            "[mpi-headpar] H=%d must be divisible by ranks=%d\n", a.H, size);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const int heads_per_rank = a.H / size;

    // One GPU per rank. Assumes rank-to-GPU on a single node — the typical
    // SLURM setup with --gres=gpu:P and --ntasks=P.
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        fprintf(stderr, "[mpi-headpar] rank %d: no CUDA devices\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    const int dev = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(dev));

    // Sizes (in floats).
    const size_t per_head     = (size_t)a.N * a.d;
    const size_t per_rank_cnt = (size_t)heads_per_rank * per_head;
    const size_t total_cnt    = (size_t)a.H * per_head;

    // Per-rank device buffers: the slice of heads this rank owns after Scatter.
    float *dQ_local = nullptr, *dK_local = nullptr,
          *dV_local = nullptr, *dO_local = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ_local, per_rank_cnt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK_local, per_rank_cnt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_local, per_rank_cnt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO_local, per_rank_cnt * sizeof(float)));

    // Rank 0 also owns the full tensors that serve as Scatter source / Gather sink.
    float *dQ_full = nullptr, *dK_full = nullptr,
          *dV_full = nullptr, *dO_full = nullptr;
    if (rank == 0) {
        CUDA_CHECK(cudaMalloc(&dQ_full, total_cnt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dK_full, total_cnt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dV_full, total_cnt * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dO_full, total_cnt * sizeof(float)));

        // Load inputs (file or deterministic LCG) into a host blob, then H2D.
        std::vector<float> hQ(total_cnt), hK(total_cnt), hV(total_cnt);
        if (!a.in_path.empty()) {
            std::vector<float> blob(3 * total_cnt);
            read_file(a.in_path, blob.data(), 3 * total_cnt * sizeof(float));
            std::memcpy(hQ.data(), blob.data() + 0 * total_cnt,
                        total_cnt * sizeof(float));
            std::memcpy(hK.data(), blob.data() + 1 * total_cnt,
                        total_cnt * sizeof(float));
            std::memcpy(hV.data(), blob.data() + 2 * total_cnt,
                        total_cnt * sizeof(float));
        } else {
            fill_lcg(hQ, a.seed + 1);
            fill_lcg(hK, a.seed + 2);
            fill_lcg(hV, a.seed + 3);
        }
        CUDA_CHECK(cudaMemcpy(dQ_full, hQ.data(),
                              total_cnt * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK_full, hK.data(),
                              total_cnt * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV_full, hV.data(),
                              total_cnt * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Single-head per-rank kernel config.
    AttnConfig cfg{};
    cfg.kernel  = kernel_from_name(a.kernel);
    cfg.N       = a.N;
    cfg.d       = a.d;
    cfg.H       = heads_per_rank;  // each rank computes its slice
    cfg.causal  = a.causal;
    cfg.w = a.w; cfg.G = a.G;
    cfg.block_row_ptr = nullptr; cfg.block_col_idx = nullptr;
    cfg.B_block = a.B_block; cfg.nnz_blocks = 0;

    // For block-sparse, every rank builds the SAME deterministic CSR using
    // the same seed. No MPI exchange of the mask is needed since the LCG in
    // random_block_mask is deterministic and the shape is identical on all
    // ranks. The CSR device buffers are owned here and freed before exit.
    BlockCSR bs_csr{};
    if (cfg.kernel == AttnKernel::BlockSparse ||
        cfg.kernel == AttnKernel::BlockSparseFP16) {
        bs_csr = random_block_mask(a.N, a.B_block, a.rho, (unsigned int)a.seed);
        cfg.block_row_ptr = bs_csr.row_ptr;
        cfg.block_col_idx = bs_csr.col_idx;
        cfg.nnz_blocks    = bs_csr.nnz_blocks;
    }

    auto run_one = [&]() {
        // Scatter Q, K, V slices from rank 0 to all ranks. Device pointers OK
        // because CUDA-aware MPI is enabled (confirmed runtime probe).
        MPI_Scatter(dQ_full, (int)per_rank_cnt, MPI_FLOAT,
                    dQ_local, (int)per_rank_cnt, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        MPI_Scatter(dK_full, (int)per_rank_cnt, MPI_FLOAT,
                    dK_local, (int)per_rank_cnt, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        MPI_Scatter(dV_full, (int)per_rank_cnt, MPI_FLOAT,
                    dV_local, (int)per_rank_cnt, MPI_FLOAT,
                    0, MPI_COMM_WORLD);

        // Local compute on the slice of heads owned by this rank.
        const float ms = attention_forward_mha(dQ_local, dK_local, dV_local,
                                               dO_local, cfg);
        if (ms < 0.0f) {
            fprintf(stderr, "[mpi-headpar] rank %d: kernel returned -1\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 5);
        }

        // Gather per-rank O slices back into rank 0's full buffer.
        MPI_Gather(dO_local, (int)per_rank_cnt, MPI_FLOAT,
                   dO_full,  (int)per_rank_cnt, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
    };

    // Warmup
    for (int i = 0; i < a.warmup; ++i) run_one();
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Measured iterations: time the full Scatter+kernel+Gather pipeline.
    const double t0 = MPI_Wtime();
    for (int i = 0; i < a.iters; ++i) run_one();
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double ms_per_iter = (t1 - t0) * 1000.0 / a.iters;

    // Output and summary (rank 0 only).
    if (rank == 0) {
        if (!a.out_path.empty()) {
            std::vector<float> hO(total_cnt);
            CUDA_CHECK(cudaMemcpy(hO.data(), dO_full,
                                  total_cnt * sizeof(float), cudaMemcpyDeviceToHost));
            write_file(a.out_path, hO.data(), total_cnt * sizeof(float));
        }
        fprintf(stdout,
                "mpi=headpar ranks=%d kernel=%s N=%d d=%d H=%d w=%d G=%d causal=%d "
                "ms_per_iter=%.4f\n",
                size, a.kernel.c_str(), a.N, a.d, a.H, a.w, a.G,
                (int)a.causal, ms_per_iter);
    }

    cudaFree(dQ_local); cudaFree(dK_local);
    cudaFree(dV_local); cudaFree(dO_local);
    if (rank == 0) {
        cudaFree(dQ_full); cudaFree(dK_full);
        cudaFree(dV_full); cudaFree(dO_full);
    }
    if (cfg.kernel == AttnKernel::BlockSparse ||
        cfg.kernel == AttnKernel::BlockSparseFP16) {
        block_csr_free(bs_csr);
    }

    MPI_Finalize();
    return 0;
}
