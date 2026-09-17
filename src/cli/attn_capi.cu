// Thin extern "C" wrapper around attention_forward_mha so the kernels can be
// called from Python via ctypes / torch.utils.cpp_extension. The wrapped
// signature takes raw device pointers + shape ints; the caller (Python side)
// is responsible for tensor allocation and lifetime.
//
// Kernel ID convention matches the AttnKernel enum order in attention_api.h:
//   0 = Dense
//   1 = DenseFP16
//   2 = Windowed
//   3 = WindowedFP16
//   4 = BlockSparse
//   5 = BlockSparseFP16

#include <cstring>
#include <cuda_runtime.h>
#include "attention_api.h"

extern "C" {

// Returns kernel wall-time in ms (cudaEvent timed), or -1 on error.
float attn_forward_mha_c(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int H, int kernel_id,
    int w, int G, int causal,
    int B_block,
    const int* block_row_ptr, const int* block_col_idx, int nnz_blocks)
{
    AttnConfig cfg{};
    cfg.kernel  = static_cast<AttnKernel>(kernel_id);
    cfg.N       = N;
    cfg.d       = d;
    cfg.H       = H;
    cfg.w       = w;
    cfg.G       = G;
    cfg.causal  = (causal != 0);
    cfg.B_block = B_block;
    cfg.nnz_blocks    = nnz_blocks;
    cfg.block_row_ptr = block_row_ptr;
    cfg.block_col_idx = block_col_idx;
    return attention_forward_mha(Q, K, V, O, cfg);
}

// Convenience: report the enum integer for a kernel name. Lets the Python
// side stay decoupled from the header. -1 if unknown.
int attn_kernel_id_from_name(const char* name) {
    if (std::strcmp(name, "dense")            == 0) return (int)AttnKernel::Dense;
    if (std::strcmp(name, "dense_fp16")       == 0) return (int)AttnKernel::DenseFP16;
    if (std::strcmp(name, "windowed")         == 0) return (int)AttnKernel::Windowed;
    if (std::strcmp(name, "windowed_fp16")    == 0) return (int)AttnKernel::WindowedFP16;
    if (std::strcmp(name, "blocksparse")      == 0) return (int)AttnKernel::BlockSparse;
    if (std::strcmp(name, "blocksparse_fp16") == 0) return (int)AttnKernel::BlockSparseFP16;
    return -1;
}

} // extern "C"
