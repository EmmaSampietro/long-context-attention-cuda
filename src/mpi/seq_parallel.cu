// MPI sequence-parallel windowed attention forward.
//
// Strategy: split the sequence N across ranks. Each rank owns Q/O rows
// [r*N/P, (r+1)*N/P) and the same range of K/V plus halos of width w from
// neighbors. Multi-head: each rank computes all H heads on its sequence slice.
//
// Per-iteration pipeline:
//   1. Halo exchange (4 MPI_Sendrecv on K/V edges, CUDA-aware).
//   2. Per-head windowed_seqpar_forward on local Q + local K (with halos).
//
// Communication volume per iteration scales as O(w*H*d), independent of N —
// the win over head-parallel for the windowed kernel, where head-parallel
// scatter/gather scales as O(N*H*d).
//
// Restrictions (M4 v1, documented as future work):
//   - kernel = windowed only
//   - G = 0 (global tokens require all-gather of K/V)
//   - causal = false (causal under seqpar requires extra coordination)
//   - w <= N/P (window cannot exceed a neighbor's slice)

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
#include "kernels_internal.h"
#include "utils.cuh"

namespace {

struct Args {
    std::string kernel = "windowed";
    int  N = 1024;
    int  d = 64;
    int  H = 4;
    int  w = 128;
    int  G = 0;
    bool causal  = false;
    bool overlap = false;   // non-blocking halo overlap with inner-region compute
    int  warmup = 5;
    int  iters  = 10;
    uint64_t seed = 0;
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
    out = (T)std::stoll(s); return true;
}
bool parse_bool(const char* arg, const std::string& key, bool& out) {
    std::string s; if (!parse_kv(arg, key, s)) return false;
    out = (s == "1" || s == "true" || s == "True"); return true;
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
        if (parse_bool(x, "overlap", a.overlap)) continue;
        if (parse_int (x, "warmup", a.warmup)) continue;
        if (parse_int (x, "iters",  a.iters))  continue;
        if (parse_int (x, "seed",   a.seed))   continue;
        if (parse_kv  (x, "in",     a.in_path))  continue;
        if (parse_kv  (x, "out",    a.out_path)) continue;
        fprintf(stderr, "unknown arg: %s\n", x);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return a;
}

void fill_lcg(std::vector<float>& v, uint64_t seed) {
    uint64_t s = (seed * 6364136223846793005ULL) + 1442695040888963407ULL;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = (float)((int64_t)(s >> 32) / (double)(1LL << 31));
    }
}

void write_file(const std::string& path, const void* buf, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s for write\n", path.c_str());
              MPI_Abort(MPI_COMM_WORLD, 2); }
    std::fwrite(buf, 1, bytes, f);
    std::fclose(f);
}

// Pack rank r's slice of a [H, N_full, d] host tensor into a contiguous
// [H, N_local, d] buffer suitable for MPI_Scatter.
void pack_for_scatter(const float* src_full, float* dst_packed,
                      int H, int N_full, int N_local, int d, int size) {
    const size_t per_head_full  = (size_t)N_full  * d;
    const size_t per_head_local = (size_t)N_local * d;
    const size_t per_rank       = (size_t)H * per_head_local;
    for (int r = 0; r < size; ++r) {
        for (int h = 0; h < H; ++h) {
            const float* src = src_full +
                (size_t)h * per_head_full + (size_t)r * per_head_local;
            float* dst = dst_packed +
                (size_t)r * per_rank + (size_t)h * per_head_local;
            std::memcpy(dst, src, per_head_local * sizeof(float));
        }
    }
}

// Inverse: unpack rank-contiguous [P, H, N_local, d] into [H, N_full, d].
void unpack_after_gather(const float* src_packed, float* dst_full,
                         int H, int N_full, int N_local, int d, int size) {
    const size_t per_head_full  = (size_t)N_full  * d;
    const size_t per_head_local = (size_t)N_local * d;
    const size_t per_rank       = (size_t)H * per_head_local;
    for (int r = 0; r < size; ++r) {
        for (int h = 0; h < H; ++h) {
            const float* src = src_packed +
                (size_t)r * per_rank + (size_t)h * per_head_local;
            float* dst = dst_full +
                (size_t)h * per_head_full + (size_t)r * per_head_local;
            std::memcpy(dst, src, per_head_local * sizeof(float));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args a = parse(argc, argv);

    // Validation
    const bool use_fp16 = (a.kernel == "windowed_fp16");
    if (a.kernel != "windowed" && !use_fp16) {
        if (rank == 0) fprintf(stderr,
            "[mpi-seqpar] supported kernels: windowed, windowed_fp16 (got '%s')\n",
            a.kernel.c_str());
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    if (use_fp16 && a.G != 0) {
        if (rank == 0) fprintf(stderr,
            "[mpi-seqpar] windowed_fp16: G>0 not yet supported\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    if (a.G < 0) {
        if (rank == 0) fprintf(stderr, "[mpi-seqpar] G must be >= 0\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    if (a.G > a.N / size) {
        if (rank == 0) fprintf(stderr,
            "[mpi-seqpar] G=%d must be <= N/ranks=%d so rank 0 owns all globals\n",
            a.G, a.N / size);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    if (a.N % size != 0) {
        if (rank == 0) fprintf(stderr,
            "[mpi-seqpar] N=%d must be divisible by ranks=%d\n", a.N, size);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const int N_local = a.N / size;
    if (a.w > N_local && size > 1) {
        if (rank == 0) fprintf(stderr,
            "[mpi-seqpar] w=%d > N_local=%d (window exceeds neighbor slice)\n",
            a.w, N_local);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    const int halo_left   = (rank == 0)        ? 0 : a.w;
    const int halo_right  = (rank == size - 1) ? 0 : a.w;
    const int N_k_local   = N_local + halo_left + halo_right;
    const int local_offset = halo_left;  // shift for the kernel

    // Device-per-rank
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) MPI_Abort(MPI_COMM_WORLD, 4);
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    // Buffer sizes (counts of floats). K/V per-head storage is N_k_local rows
    // of windowed K plus G "global key" rows that get broadcast from rank 0
    // and concatenated at positions [N_k_local, N_k_local + G).
    const size_t per_head_q   = (size_t)N_local   * a.d;
    const size_t per_head_k   = (size_t)(N_k_local + a.G) * a.d;
    const size_t total_q_loc  = (size_t)a.H * per_head_q;
    const size_t total_k_loc  = (size_t)a.H * per_head_k;
    const size_t per_head_full = (size_t)a.N * a.d;
    const size_t total_full   = (size_t)a.H * per_head_full;

    // Per-rank device buffers
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    // For the G>0 "global queries" path (Longformer-style globals attending to
    // the whole sequence), rank 0 also persists a copy of the full K, V on
    // device. Rank 0 generated the data via LCG at init time, so we keep it
    // device-side rather than running a per-iteration MPI_Gather. In a
    // production setting where data does not originate at rank 0, the
    // equivalent algorithm uses MPI_Gather of each rank's inner K, V to
    // rank 0 before the timed loop; we note this design alternative explicitly.
    float *dK_full_global = nullptr, *dV_full_global = nullptr;
    if (rank == 0 && a.G > 0 && !use_fp16) {
        CUDA_CHECK(cudaMalloc(&dK_full_global, total_full * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dV_full_global, total_full * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&dQ, total_q_loc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, total_k_loc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, total_k_loc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, total_q_loc * sizeof(float)));

    // Generate full Q, K, V on rank 0 (host), pack, scatter, copy to device.
    {
        std::vector<float> hQ_local(total_q_loc), hK_inner(total_q_loc),
                           hV_inner(total_q_loc);

        if (rank == 0) {
            std::vector<float> hQ_full(total_full), hK_full(total_full),
                               hV_full(total_full);
            fill_lcg(hQ_full, a.seed + 1);
            fill_lcg(hK_full, a.seed + 2);
            fill_lcg(hV_full, a.seed + 3);

            // G>0 path: rank 0 keeps the full K, V on device so it can run a
            // dense pass on the first G "global query" rows after the regular
            // seq-par iteration.
            if (a.G > 0 && !use_fp16) {
                CUDA_CHECK(cudaMemcpy(dK_full_global, hK_full.data(),
                                      total_full * sizeof(float),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(dV_full_global, hV_full.data(),
                                      total_full * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }

            std::vector<float> hQ_packed(total_full),
                               hK_packed(total_full),
                               hV_packed(total_full);
            pack_for_scatter(hQ_full.data(), hQ_packed.data(),
                             a.H, a.N, N_local, a.d, size);
            pack_for_scatter(hK_full.data(), hK_packed.data(),
                             a.H, a.N, N_local, a.d, size);
            pack_for_scatter(hV_full.data(), hV_packed.data(),
                             a.H, a.N, N_local, a.d, size);

            MPI_Scatter(hQ_packed.data(), (int)total_q_loc, MPI_FLOAT,
                        hQ_local.data(),  (int)total_q_loc, MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            MPI_Scatter(hK_packed.data(), (int)total_q_loc, MPI_FLOAT,
                        hK_inner.data(),  (int)total_q_loc, MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            MPI_Scatter(hV_packed.data(), (int)total_q_loc, MPI_FLOAT,
                        hV_inner.data(),  (int)total_q_loc, MPI_FLOAT,
                        0, MPI_COMM_WORLD);
        } else {
            MPI_Scatter(nullptr, 0, MPI_FLOAT,
                        hQ_local.data(),  (int)total_q_loc, MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            MPI_Scatter(nullptr, 0, MPI_FLOAT,
                        hK_inner.data(),  (int)total_q_loc, MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            MPI_Scatter(nullptr, 0, MPI_FLOAT,
                        hV_inner.data(),  (int)total_q_loc, MPI_FLOAT,
                        0, MPI_COMM_WORLD);
        }

        // Copy Q (no halos) to device. For K/V, place the local inner rows
        // at offset halo_left within each head, leaving halo space at the ends.
        CUDA_CHECK(cudaMemcpy(dQ, hQ_local.data(),
                              total_q_loc * sizeof(float), cudaMemcpyHostToDevice));
        for (int h = 0; h < a.H; ++h) {
            CUDA_CHECK(cudaMemcpy(
                dK + (size_t)h * per_head_k + (size_t)halo_left * a.d,
                hK_inner.data() + (size_t)h * per_head_q,
                per_head_q * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                dV + (size_t)h * per_head_k + (size_t)halo_left * a.d,
                hV_inner.data() + (size_t)h * per_head_q,
                per_head_q * sizeof(float), cudaMemcpyHostToDevice));
        }

        // G > 0: broadcast rank 0's first G K/V rows (the "global keys") to
        // every rank, then place them at positions [N_k_local, N_k_local + G)
        // of each head's K/V buffer. Pack into a contiguous [H, G, d] block
        // first so the Bcast is one call per K/V instead of H.
        if (a.G > 0) {
            const size_t global_per_head = (size_t)a.G * a.d;
            const size_t global_total    = (size_t)a.H * global_per_head;
            std::vector<float> hGlobalK(global_total), hGlobalV(global_total);
            if (rank == 0) {
                for (int h = 0; h < a.H; ++h) {
                    std::memcpy(hGlobalK.data() + (size_t)h * global_per_head,
                                hK_inner.data() + (size_t)h * per_head_q,
                                global_per_head * sizeof(float));
                    std::memcpy(hGlobalV.data() + (size_t)h * global_per_head,
                                hV_inner.data() + (size_t)h * per_head_q,
                                global_per_head * sizeof(float));
                }
            }
            MPI_Bcast(hGlobalK.data(), (int)global_total, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(hGlobalV.data(), (int)global_total, MPI_FLOAT, 0, MPI_COMM_WORLD);
            for (int h = 0; h < a.H; ++h) {
                CUDA_CHECK(cudaMemcpy(
                    dK + (size_t)h * per_head_k + (size_t)N_k_local * a.d,
                    hGlobalK.data() + (size_t)h * global_per_head,
                    global_per_head * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(
                    dV + (size_t)h * per_head_k + (size_t)N_k_local * a.d,
                    hGlobalV.data() + (size_t)h * global_per_head,
                    global_per_head * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }

    // Halo exchange uses packed contiguous send/recv buffers (one per direction
    // per K/V), because the [H, N_k_local, d] layout has heads strided. Pack/
    // unpack with cudaMemcpy2D so the data path stays on-device.
    //
    // Serial path reuses dSendL/dSendR/dRecvL/dRecvR for K and then V.
    // Overlap path keeps K and V buffers separate so both can be in flight
    // simultaneously while the inner kernels run.
    const size_t halo_bytes_per_head = (size_t)a.w * a.d * sizeof(float);
    const size_t halo_total_bytes    = (size_t)a.H * halo_bytes_per_head;
    float *dSendL = nullptr, *dSendR = nullptr, *dRecvL = nullptr, *dRecvR = nullptr;
    float *dSendLV = nullptr, *dSendRV = nullptr, *dRecvLV = nullptr, *dRecvRV = nullptr;
    if (size > 1) {
        CUDA_CHECK(cudaMalloc(&dSendL, halo_total_bytes));
        CUDA_CHECK(cudaMalloc(&dSendR, halo_total_bytes));
        CUDA_CHECK(cudaMalloc(&dRecvL, halo_total_bytes));
        CUDA_CHECK(cudaMalloc(&dRecvR, halo_total_bytes));
        if (a.overlap) {
            CUDA_CHECK(cudaMalloc(&dSendLV, halo_total_bytes));
            CUDA_CHECK(cudaMalloc(&dSendRV, halo_total_bytes));
            CUDA_CHECK(cudaMalloc(&dRecvLV, halo_total_bytes));
            CUDA_CHECK(cudaMalloc(&dRecvRV, halo_total_bytes));
        }
    }
    const int left_rank  = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    const int right_rank = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // Overlap requires N_local >= 2w so the inner region (rows [w, N_local-w))
    // is non-empty when both halos exist. Fall back to the serial path otherwise.
    bool can_overlap = a.overlap && (size > 1) && (N_local >= 2 * a.w);
    if (a.overlap && !can_overlap && rank == 0) {
        fprintf(stderr,
                "[mpi-seqpar] overlap disabled: ranks=%d N_local=%d < 2w=%d\n",
                size, N_local, 2 * a.w);
    }

    // Per-head streams (mirrors attention_forward_mha)
    constexpr int kMaxStreams = 16;
    cudaStream_t streams[kMaxStreams];
    const int S = (a.H < kMaxStreams) ? a.H : kMaxStreams;
    for (int s = 0; s < S; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // Halo exchange pattern (called once per iteration, for K and again for V):
    //   - Pack: copy the LEFTMOST w rows of the inner region into dSendL,
    //           and the RIGHTMOST w rows into dSendR. cudaMemcpy2D handles the
    //           strided per-head layout.
    //   - Sendrecv: dSendL → left neighbor's dRecvR (becomes their right halo),
    //              dSendR → right neighbor's dRecvL (becomes their left halo).
    //   - Unpack: copy dRecvL into K's left halo region [0, halo_left),
    //             copy dRecvR into K's right halo region [N_k_local - halo_right, N_k_local).
    auto exchange_halo = [&](float* dBuf) {
        if (size <= 1) return;
        const size_t kv_pitch_bytes = (size_t)N_k_local * a.d * sizeof(float);
        const size_t halo_pitch_bytes = (size_t)a.w * a.d * sizeof(float);
        const size_t halo_row_bytes   = halo_pitch_bytes;  // halos are dense
        // Pack left edge (LEFTMOST w of inner rows) → dSendL
        if (left_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2D(
                dSendL, halo_pitch_bytes,
                dBuf + (size_t)halo_left * a.d, kv_pitch_bytes,
                halo_row_bytes, a.H,
                cudaMemcpyDeviceToDevice));
        }
        // Pack right edge (RIGHTMOST w of inner rows) → dSendR
        if (right_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2D(
                dSendR, halo_pitch_bytes,
                dBuf + (size_t)(halo_left + N_local - a.w) * a.d, kv_pitch_bytes,
                halo_row_bytes, a.H,
                cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());  // ensure packs finish before MPI

        // Exchange: send to left, receive from right (and vice versa).
        const int halo_count = (int)((size_t)a.H * a.w * a.d);
        MPI_Sendrecv(dSendL, halo_count, MPI_FLOAT, left_rank,  0,
                     dRecvR, halo_count, MPI_FLOAT, right_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(dSendR, halo_count, MPI_FLOAT, right_rank, 1,
                     dRecvL, halo_count, MPI_FLOAT, left_rank,  1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Unpack: dRecvL → K's left halo, dRecvR → K's right halo
        if (left_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2D(
                dBuf, kv_pitch_bytes,
                dRecvL, halo_pitch_bytes,
                halo_row_bytes, a.H,
                cudaMemcpyDeviceToDevice));
        }
        if (right_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2D(
                dBuf + (size_t)(halo_left + N_local) * a.d, kv_pitch_bytes,
                dRecvR, halo_pitch_bytes,
                halo_row_bytes, a.H,
                cudaMemcpyDeviceToDevice));
        }
    };

    auto run_one_serial = [&]() {
        // Halo exchange for K and V (blocking)
        exchange_halo(dK);
        exchange_halo(dV);

        // Per-head kernel launches on streams. Dispatch on use_fp16 so the
        // FP16 + WMMA path goes to its own kernel; otherwise the FP32 path.
        for (int h = 0; h < a.H; ++h) {
            const float* Qh = dQ + (size_t)h * per_head_q;
            const float* Kh = dK + (size_t)h * per_head_k;
            const float* Vh = dV + (size_t)h * per_head_k;
            float*       Oh = dO + (size_t)h * per_head_q;
            const float ms = use_fp16
                ? windowed_fp16_seqpar_forward(
                      Qh, Kh, Vh, Oh, N_local, N_k_local, a.d, a.w, local_offset,
                      a.causal, streams[h % S], /*measure=*/false)
                : windowed_seqpar_forward(
                      Qh, Kh, Vh, Oh, N_local, N_k_local, a.d, a.w, local_offset,
                      a.causal, a.G, streams[h % S], /*measure=*/false);
            if (ms < 0.0f) {
                fprintf(stderr, "[mpi-seqpar] rank %d head %d kernel returned -1\n", rank, h);
                MPI_Abort(MPI_COMM_WORLD, 5);
            }
        }

        // G>0 global queries (FP32 only): rank 0 overwrites the first G rows
        // of dO by running a dense pass (windowed_seqpar with w=N) against
        // the full K, V. This implements the Longformer-style "global queries
        // attend everywhere" semantics that the per-rank seq-par kernel
        // cannot deliver on its own (it sees only the windowed range + halos).
        if (rank == 0 && a.G > 0 && !use_fp16) {
            for (int h = 0; h < a.H; ++h) {
                const float* Qh = dQ + (size_t)h * per_head_q;            // first G rows live here
                const float* Kh = dK_full_global + (size_t)h * per_head_full;
                const float* Vh = dV_full_global + (size_t)h * per_head_full;
                float*       Oh = dO + (size_t)h * per_head_q;            // overwrite first G rows
                const float ms = windowed_seqpar_forward(
                    Qh, Kh, Vh, Oh,
                    a.G,         // N_q = G query rows
                    a.N,         // N_k = full sequence
                    a.d,
                    a.N,         // w = N → effectively dense (every position in window)
                    0,           // local_offset = 0 (rank 0's global coords match local)
                    a.causal,
                    0,           // G = 0 (no additional global-key concat in this pass)
                    streams[h % S], /*measure=*/false);
                if (ms < 0.0f) {
                    fprintf(stderr, "[mpi-seqpar] rank 0 global-queries head %d returned -1\n", h);
                    MPI_Abort(MPI_COMM_WORLD, 5);
                }
            }
        }

        for (int s = 0; s < S; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    };

    // Overlapped pipeline. For each iteration:
    //   1. Pack K & V edges into send buffers (cudaMemcpy2D)
    //   2. Post non-blocking Isend/Irecv for K & V (4 K + 4 V = up to 8 reqs)
    //   3. Launch inner-region kernels (rows [w, N_local - w)) on streams
    //   4. MPI_Waitall on halo requests
    //   5. Unpack received halos into K/V's halo regions
    //   6. Launch left- and right-boundary kernels (rows [0, w) and [N_local-w, N_local))
    //   7. Sync all streams
    // The inner kernels run concurrently with the halo exchange. Boundary
    // kernels must wait for the unpack.
    auto pack_edges = [&](float* dBuf, float* dSendL_buf, float* dSendR_buf) {
        const size_t kv_pitch_bytes   = (size_t)N_k_local * a.d * sizeof(float);
        const size_t halo_pitch_bytes = (size_t)a.w * a.d * sizeof(float);
        const size_t halo_row_bytes   = halo_pitch_bytes;
        if (left_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2D(
                dSendL_buf, halo_pitch_bytes,
                dBuf + (size_t)halo_left * a.d, kv_pitch_bytes,
                halo_row_bytes, a.H,
                cudaMemcpyDeviceToDevice));
        }
        if (right_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2D(
                dSendR_buf, halo_pitch_bytes,
                dBuf + (size_t)(halo_left + N_local - a.w) * a.d, kv_pitch_bytes,
                halo_row_bytes, a.H,
                cudaMemcpyDeviceToDevice));
        }
    };
    auto unpack_edges = [&](float* dBuf, float* dRecvL_buf, float* dRecvR_buf) {
        const size_t kv_pitch_bytes   = (size_t)N_k_local * a.d * sizeof(float);
        const size_t halo_pitch_bytes = (size_t)a.w * a.d * sizeof(float);
        const size_t halo_row_bytes   = halo_pitch_bytes;
        if (left_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2D(
                dBuf, kv_pitch_bytes,
                dRecvL_buf, halo_pitch_bytes,
                halo_row_bytes, a.H,
                cudaMemcpyDeviceToDevice));
        }
        if (right_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy2D(
                dBuf + (size_t)(halo_left + N_local) * a.d, kv_pitch_bytes,
                dRecvR_buf, halo_pitch_bytes,
                halo_row_bytes, a.H,
                cudaMemcpyDeviceToDevice));
        }
    };
    auto launch_head_range = [&](int q_start, int q_count) {
        if (q_count <= 0) return;
        for (int h = 0; h < a.H; ++h) {
            const float* Qh = dQ + (size_t)h * per_head_q + (size_t)q_start * a.d;
            const float* Kh = dK + (size_t)h * per_head_k;
            const float* Vh = dV + (size_t)h * per_head_k;
            float*       Oh = dO + (size_t)h * per_head_q + (size_t)q_start * a.d;
            const float ms = use_fp16
                ? windowed_fp16_seqpar_forward(
                      Qh, Kh, Vh, Oh, q_count, N_k_local, a.d, a.w,
                      local_offset + q_start, a.causal,
                      streams[h % S], /*measure=*/false)
                : windowed_seqpar_forward(
                      Qh, Kh, Vh, Oh, q_count, N_k_local, a.d, a.w,
                      local_offset + q_start, a.causal, a.G,
                      streams[h % S], /*measure=*/false);
            if (ms < 0.0f) {
                fprintf(stderr, "[mpi-seqpar] rank %d head %d kernel returned -1\n", rank, h);
                MPI_Abort(MPI_COMM_WORLD, 5);
            }
        }
    };
    auto run_one_overlap = [&]() {
        const int halo_count = (int)((size_t)a.H * a.w * a.d);

        // 1. Pack K & V edges. Must finish before MPI sees the buffers.
        pack_edges(dK, dSendL,  dSendR);
        pack_edges(dV, dSendLV, dSendRV);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. Post non-blocking exchanges for K and V.
        MPI_Request reqs[8];
        int nreq = 0;
        MPI_Irecv(dRecvR,  halo_count, MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(dRecvL,  halo_count, MPI_FLOAT, left_rank,  1, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(dSendL,  halo_count, MPI_FLOAT, left_rank,  0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(dSendR,  halo_count, MPI_FLOAT, right_rank, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(dRecvRV, halo_count, MPI_FLOAT, right_rank, 2, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(dRecvLV, halo_count, MPI_FLOAT, left_rank,  3, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(dSendLV, halo_count, MPI_FLOAT, left_rank,  2, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(dSendRV, halo_count, MPI_FLOAT, right_rank, 3, MPI_COMM_WORLD, &reqs[nreq++]);

        // 3. Launch inner kernels overlapped with the MPI traffic.
        //    Inner rows [w, N_local - w) only attend within the local K slice
        //    so they are independent of the halos that are still in flight.
        launch_head_range(/*q_start=*/a.w, /*q_count=*/N_local - 2 * a.w);

        // 4. Wait for halos to arrive.
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // 5. Unpack into K's and V's halo regions.
        unpack_edges(dK, dRecvL,  dRecvR);
        unpack_edges(dV, dRecvLV, dRecvRV);

        // 6. Launch left and right boundary kernels (rows [0, w) and [N_local-w, N_local)).
        launch_head_range(/*q_start=*/0,             /*q_count=*/a.w);
        launch_head_range(/*q_start=*/N_local - a.w, /*q_count=*/a.w);

        // 7. Sync streams to bound this iteration.
        for (int s = 0; s < S; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    };

    auto run_one = [&]() {
        if (can_overlap) run_one_overlap();
        else             run_one_serial();
    };

    // Warmup
    for (int i = 0; i < a.warmup; ++i) run_one();
    MPI_Barrier(MPI_COMM_WORLD);

    // Measured iterations
    const double t0 = MPI_Wtime();
    for (int i = 0; i < a.iters; ++i) run_one();
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double ms_per_iter = (t1 - t0) * 1000.0 / a.iters;

    // Gather O for validation if requested
    if (!a.out_path.empty()) {
        std::vector<float> hO_local(total_q_loc);
        CUDA_CHECK(cudaMemcpy(hO_local.data(), dO,
                              total_q_loc * sizeof(float), cudaMemcpyDeviceToHost));
        if (rank == 0) {
            std::vector<float> hO_packed(total_full), hO_full(total_full);
            MPI_Gather(hO_local.data(), (int)total_q_loc, MPI_FLOAT,
                       hO_packed.data(), (int)total_q_loc, MPI_FLOAT,
                       0, MPI_COMM_WORLD);
            unpack_after_gather(hO_packed.data(), hO_full.data(),
                                a.H, a.N, N_local, a.d, size);
            write_file(a.out_path, hO_full.data(), total_full * sizeof(float));
        } else {
            MPI_Gather(hO_local.data(), (int)total_q_loc, MPI_FLOAT,
                       nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        fprintf(stdout,
                "mpi=seqpar ranks=%d kernel=%s N=%d d=%d H=%d w=%d G=%d causal=%d "
                "overlap=%d ms_per_iter=%.4f\n",
                size, a.kernel.c_str(), a.N, a.d, a.H, a.w, a.G,
                (int)a.causal, (int)can_overlap, ms_per_iter);
    }

    for (int s = 0; s < S; ++s) cudaStreamDestroy(streams[s]);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    if (dK_full_global) cudaFree(dK_full_global);
    if (dV_full_global) cudaFree(dV_full_global);
    if (size > 1) {
        cudaFree(dSendL); cudaFree(dSendR);
        cudaFree(dRecvL); cudaFree(dRecvR);
        if (a.overlap) {
            cudaFree(dSendLV); cudaFree(dSendRV);
            cudaFree(dRecvLV); cudaFree(dRecvRV);
        }
    }

    MPI_Finalize();
    return 0;
}
