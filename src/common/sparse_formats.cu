// Host-side BlockCSR builders. Each builder allocates the row_ptr / col_idx
// device buffers via cudaMalloc and copies the CSR from host. Caller frees with
// block_csr_free().

#include "sparse_formats.h"
#include "utils.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Tiny deterministic LCG so random_block_mask is reproducible without pulling
// in <random>.
struct LCG {
    uint64_t s;
    explicit LCG(uint32_t seed) : s((uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    float next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float)((s >> 32) / (double)(uint64_t)0xFFFFFFFFULL);
    }
};

BlockCSR upload_csr(const std::vector<int>& row_ptr_host,
                    const std::vector<int>& col_idx_host,
                    int num_block_rows, int num_block_cols, int B) {
    BlockCSR csr;
    csr.num_block_rows = num_block_rows;
    csr.num_block_cols = num_block_cols;
    csr.B = B;
    csr.nnz_blocks = (int)col_idx_host.size();

    CUDA_CHECK(cudaMalloc(&csr.row_ptr,
        (num_block_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(csr.row_ptr, row_ptr_host.data(),
        (num_block_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (csr.nnz_blocks > 0) {
        CUDA_CHECK(cudaMalloc(&csr.col_idx, csr.nnz_blocks * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(csr.col_idx, col_idx_host.data(),
            csr.nnz_blocks * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        csr.col_idx = nullptr;
    }
    return csr;
}

}  // namespace

BlockCSR block_csr_from_dense_mask(const unsigned char* mask_host,
                                   int num_block_rows, int num_block_cols,
                                   int B)
{
    std::vector<int> row_ptr_host(num_block_rows + 1, 0);
    std::vector<int> col_idx_host;
    col_idx_host.reserve((size_t)num_block_rows * num_block_cols / 4 + 1);

    for (int r = 0; r < num_block_rows; ++r) {
        row_ptr_host[r] = (int)col_idx_host.size();
        for (int c = 0; c < num_block_cols; ++c) {
            if (mask_host[(size_t)r * num_block_cols + c])
                col_idx_host.push_back(c);
        }
    }
    row_ptr_host[num_block_rows] = (int)col_idx_host.size();

    return upload_csr(row_ptr_host, col_idx_host, num_block_rows, num_block_cols, B);
}

BlockCSR random_block_mask(int N, int B, float rho, unsigned int seed)
{
    const int nb = (N + B - 1) / B;
    std::vector<unsigned char> mask((size_t)nb * nb, 0);
    LCG rng(seed);
    for (int r = 0; r < nb; ++r)
        for (int c = 0; c < nb; ++c)
            if (rng.next() < rho) mask[(size_t)r * nb + c] = 1;
    return block_csr_from_dense_mask(mask.data(), nb, nb, B);
}

BlockCSR window_as_block_mask(int N, int B, int w, bool causal)
{
    // Block (r, c) is active iff the row interval [r*B, r*B+B-1] (clipped to N)
    // and the col interval [c*B, c*B+B-1] (clipped to N) are within w of each
    // other (window) and, if causal, share at least one (i, j) with j <= i.
    const int nb = (N + B - 1) / B;
    std::vector<unsigned char> mask((size_t)nb * nb, 0);
    for (int r = 0; r < nb; ++r) {
        const int row_lo = r * B;
        const int row_hi = std::min(r * B + B - 1, N - 1);
        for (int c = 0; c < nb; ++c) {
            const int col_lo = c * B;
            const int col_hi = std::min(c * B + B - 1, N - 1);
            const bool window_ok = (col_lo - row_hi <= w) && (row_lo - col_hi <= w);
            const bool causal_ok = !causal || (col_lo <= row_hi);
            if (window_ok && causal_ok) mask[(size_t)r * nb + c] = 1;
        }
    }
    return block_csr_from_dense_mask(mask.data(), nb, nb, B);
}

void block_csr_free(BlockCSR& csr)
{
    if (csr.row_ptr) { cudaFree(csr.row_ptr); csr.row_ptr = nullptr; }
    if (csr.col_idx) { cudaFree(csr.col_idx); csr.col_idx = nullptr; }
    csr.nnz_blocks = 0;
}
