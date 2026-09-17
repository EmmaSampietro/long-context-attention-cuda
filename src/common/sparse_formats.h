#pragma once
#include <cuda_runtime.h>

// CSR over the block grid (N/B) x (N/B). Each query block-row stores the list of
// active K/V block-columns it attends to. One CUDA thread block handles one
// query block-row at kernel launch time.
struct BlockCSR {
    int  num_block_rows;   // N / B
    int  num_block_cols;   // N / B
    int  B;                // block side (e.g., 64)
    int  nnz_blocks;       // total number of active blocks
    int* row_ptr;          // device, length num_block_rows + 1
    int* col_idx;          // device, length nnz_blocks
};

// Host-side builders. Return ownership of device buffers via the BlockCSR struct
// (caller frees with cudaFree on row_ptr / col_idx).
BlockCSR block_csr_from_dense_mask(const unsigned char* mask_host,
                                    int num_block_rows, int num_block_cols,
                                    int B);

BlockCSR random_block_mask(int N, int B, float rho, unsigned int seed);

// Sliding-window pattern expressed as a block mask — used to cross-validate the
// block-sparse kernel against the windowed kernel: same pattern, different code path.
BlockCSR window_as_block_mask(int N, int B, int w, bool causal);

void block_csr_free(BlockCSR& csr);
