"""End-to-end attention LAYER benchmark.

Wraps each attention kernel with the projections a real transformer layer
performs (Q/K/V via nn.Linear, plus an output projection) and times whole-
layer forward wall time vs. the SDPA-FP16 production reference. Answers the
question "do our 21x kernel-only wins survive the inclusion of projection
overhead?".

Our kernels are called via ctypes against build/libattn.so (built by
`make attn-lib`). PyTorch tensors expose .data_ptr() so we can hand raw
device pointers to the C ABI shim. Both paths use the same projection code
in PyTorch (cuBLAS under the hood) so the comparison isolates the attention
component.

Outputs:
    report/data/end_to_end_<HOST>_<JOBID>.csv
"""

import argparse
import csv
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = ROOT / "build" / "libattn.so"


# --- ctypes bindings -------------------------------------------------------

def load_lib():
    if not LIB_PATH.exists():
        sys.exit(f"libattn.so not found at {LIB_PATH}; run `make attn-lib`")
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.attn_forward_mha_c.restype = ctypes.c_float
    lib.attn_forward_mha_c.argtypes = [
        ctypes.c_void_p,  # Q
        ctypes.c_void_p,  # K
        ctypes.c_void_p,  # V
        ctypes.c_void_p,  # O
        ctypes.c_int,     # N
        ctypes.c_int,     # d
        ctypes.c_int,     # H
        ctypes.c_int,     # kernel_id
        ctypes.c_int,     # w
        ctypes.c_int,     # G
        ctypes.c_int,     # causal
        ctypes.c_int,     # B_block
        ctypes.c_void_p,  # block_row_ptr
        ctypes.c_void_p,  # block_col_idx
        ctypes.c_int,     # nnz_blocks
    ]
    return lib


# Must match AttnKernel enum order in src/common/attention_api.h.
KERNEL_IDS = {
    "dense":            0,
    "dense_fp16":       1,
    "windowed":         2,
    "windowed_fp16":    3,
    "blocksparse":      4,
    "blocksparse_fp16": 5,
}


# --- Attention layers ------------------------------------------------------

class OurAttnLayer(torch.nn.Module):
    """Three nn.Linear projections + our CUDA kernel + output projection.

    Production-realistic wrapper: projections run in FP16 (matching SDPA's
    fast path), Q/K/V are cast to FP32 only for the call into our kernel
    (which exposes an FP32 host API; the kernel's matmuls run FP16 WMMA
    internally for the *_fp16 variants). Output is cast back to FP16 before
    Wo so the output projection also benefits from FP16 tensor cores.

    Without this trick the comparison is unfair: forcing the projections to
    FP32 makes them ~5x slower on Turing (no tensor cores in FP32 path),
    which dominates the total layer time and masks the attention win.
    """

    def __init__(self, d_model, n_heads, kernel="windowed_fp16", w=256,
                 lib=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.w       = w
        self.kernel  = kernel
        self.kernel_id = KERNEL_IDS[kernel]
        # Projections in FP16 to match SDPA's fast path.
        self.Wq = torch.nn.Linear(d_model, d_model, bias=False, device="cuda",
                                  dtype=torch.float16)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False, device="cuda",
                                  dtype=torch.float16)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False, device="cuda",
                                  dtype=torch.float16)
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False, device="cuda",
                                  dtype=torch.float16)
        self._lib = lib

    def forward(self, X):
        # X: [N, d_model] FP16
        N = X.shape[0]
        # Project in FP16 (fast HGEMM on tensor cores).
        Q = self.Wq(X).view(N, self.n_heads, self.d_head).transpose(0, 1).contiguous()
        K = self.Wk(X).view(N, self.n_heads, self.d_head).transpose(0, 1).contiguous()
        V = self.Wv(X).view(N, self.n_heads, self.d_head).transpose(0, 1).contiguous()
        # Cast to FP32 at the kernel boundary. Bandwidth-bound, ~0.05ms at
        # N=4K, H=16, d=64. A future revision could expose an FP16 host API
        # for our kernel to skip this entirely.
        Q32 = Q.float()
        K32 = K.float()
        V32 = V.float()
        O32 = torch.empty_like(Q32)
        ms = self._lib.attn_forward_mha_c(
            Q32.data_ptr(), K32.data_ptr(), V32.data_ptr(), O32.data_ptr(),
            N, self.d_head, self.n_heads, self.kernel_id,
            self.w, 0, 0, 64,
            0, 0, 0,
        )
        if ms < 0:
            raise RuntimeError(f"attn_forward_mha_c returned {ms}")
        # Cast back to FP16 for the output projection.
        O = O32.half().transpose(0, 1).contiguous().view(N, self.d_model)
        return self.Wo(O)


class TorchAttnLayer(torch.nn.Module):
    """Same three projections + F.scaled_dot_product_attention + output proj.

    We let PyTorch pick its best backend; on H100/A100/Turing-FP16 this is
    FlashAttention-2. Caller controls precision via the parameter dtype.
    """

    def __init__(self, d_model, n_heads, dtype=torch.float16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.dtype   = dtype
        self.Wq = torch.nn.Linear(d_model, d_model, bias=False, device="cuda",
                                  dtype=dtype)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False, device="cuda",
                                  dtype=dtype)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False, device="cuda",
                                  dtype=dtype)
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False, device="cuda",
                                  dtype=dtype)

    def forward(self, X):
        # X: [N, d_model] in self.dtype.
        N = X.shape[0]
        Q = self.Wq(X).view(N, self.n_heads, self.d_head).transpose(0, 1).unsqueeze(0)
        K = self.Wk(X).view(N, self.n_heads, self.d_head).transpose(0, 1).unsqueeze(0)
        V = self.Wv(X).view(N, self.n_heads, self.d_head).transpose(0, 1).unsqueeze(0)
        # F.scaled_dot_product_attention expects [B, H, N, d].
        O = F.scaled_dot_product_attention(Q, K, V)
        O = O.squeeze(0).transpose(0, 1).contiguous().view(N, self.d_model)
        return self.Wo(O)


# --- Timing helpers --------------------------------------------------------

def cuda_time_median(fn, iters=10, warmup=3):
    """cudaEvent-timed median of `iters` calls, after `warmup` warmup calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2], samples[0], samples[-1]


# --- Sweep -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="1024,2048,4096,8192,16384",
                    help="comma-separated sequence lengths")
    ap.add_argument("--H", type=int, default=16, help="number of heads")
    ap.add_argument("--d", type=int, default=64, help="head dim (d_model = H*d)")
    ap.add_argument("--w", type=int, default=256, help="windowed window half-width")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--kernels", default="windowed_fp16,dense_fp16",
                    help="comma-separated list of our kernel names to benchmark")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device available")

    lib = load_lib()
    Ns = [int(x) for x in args.N.split(",")]
    our_kernels = [k.strip() for k in args.kernels.split(",")]
    d_model = args.H * args.d
    print(f"d_model={d_model}  H={args.H}  d_head={args.d}  w={args.w}")
    print(f"sequence lengths: {Ns}")
    print(f"our kernels: {our_kernels}")

    if args.out is None:
        host = os.uname().nodename.split('.')[0]
        jobid = os.environ.get("SLURM_JOB_ID", str(int(time.time())))
        args.out = str(ROOT / "report" / "data" / f"end_to_end_{host}_{jobid}.csv")

    rows = []
    with open(args.out, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["mode", "N", "H", "d_head", "d_model",
                      "ms_median", "ms_min", "ms_max"])

    # --- SDPA-FP16 baseline (full layer w/ projections) ----------------
        sdpa = TorchAttnLayer(d_model, args.H, dtype=torch.float16).cuda().eval()
        for N in Ns:
            X = torch.randn(N, d_model, device="cuda", dtype=torch.float16)
            try:
                med, mn, mx = cuda_time_median(
                    lambda: sdpa(X), iters=args.iters, warmup=args.warmup)
                row = ["sdpa_fp16_layer", N, args.H, args.d, d_model, med, mn, mx]
                wri.writerow(row); rows.append(row)
                print(f"  sdpa_fp16_layer   N={N:>6}: median={med:8.3f} ms")
            except RuntimeError as e:
                print(f"  sdpa_fp16_layer   N={N:>6}: FAILED ({e})")
        del sdpa

    # --- Our kernels (each gets its own layer; FP32 host) --------------
        for kname in our_kernels:
            our = OurAttnLayer(d_model, args.H, kernel=kname, w=args.w,
                               lib=lib).cuda().eval()
            for N in Ns:
                X = torch.randn(N, d_model, device="cuda", dtype=torch.float16)
                try:
                    med, mn, mx = cuda_time_median(
                        lambda: our(X), iters=args.iters, warmup=args.warmup)
                    label = f"ours_{kname}_layer"
                    row = [label, N, args.H, args.d, d_model, med, mn, mx]
                    wri.writerow(row); rows.append(row)
                    print(f"  {label:>26s}  N={N:>6}: median={med:8.3f} ms")
                except RuntimeError as e:
                    print(f"  ours_{kname}_layer N={N:>6}: FAILED ({e})")
            del our

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
