"""PyTorch scaled_dot_product_attention baseline at the same shapes as the
dense kernel sweep.

Designed to contextualize the roofline finding (our hand-rolled FA-style
kernel reaches ~9% of FP32 peak on RTX 6000). PyTorch's SDPA uses
FlashAttention-2 / memory-efficient backends and is a strong reference.

Notes on precision:
  - Our hand-rolled kernel is FP32. Torch's FA2 backend supports FP16/BF16
    only; FP32 falls back to the math / memory-efficient backend. So we run
    BOTH dtypes by default so the report can cite the FP32 apples-to-apples
    comparison and the FP16+FA2 "production attention" reference.

Emits a CSV in the same column schema as bench/final_crossover.py (so the
existing plotter can pick it up):
    kernel = "torch_sdpa_fp32" | "torch_sdpa_fp16"

Usage:
    python bench/sdpa_baseline.py \\
        --N 1024,2048,4096,8192,16384,32768 \\
        --H 16 --d 64 --warmup 5 --iters 10 \\
        --out report/data/sdpa_baseline.csv
"""
import argparse
import csv
import datetime as _dt
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


def _gpu_model():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).decode().strip().splitlines()
        return out[0] if out else "unknown"
    except Exception:
        return "unknown"


def time_sdpa(N, d, H, dtype, warmup, iters, device):
    torch.manual_seed(0)
    # Shape used by SDPA: (batch, heads, seq, head_dim). We use batch=1, H heads.
    Q = torch.randn((1, H, N, d), dtype=dtype, device=device)
    K = torch.randn((1, H, N, d), dtype=dtype, device=device)
    V = torch.randn((1, H, N, d), dtype=dtype, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    torch.cuda.synchronize(device)

    # Measured
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        ends[i].record()
    torch.cuda.synchronize(device)
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))

    ms_median = times_ms[len(times_ms) // 2]
    ms_min    = times_ms[0]
    ms_max    = times_ms[-1]
    mem_mb    = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    return ms_median, ms_min, ms_max, mem_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",  default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--H",  type=int, default=16)
    ap.add_argument("--d",  type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters",  type=int, default=10)
    ap.add_argument("--dtypes", default="fp32,fp16",
                    help="comma-separated: fp32, fp16, bf16")
    ap.add_argument("--out",    type=Path,
                    default=ROOT / "report" / "data" / "sdpa_baseline.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available — torch.cuda.is_available() is False")

    device = torch.device("cuda:0")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"device {torch.cuda.get_device_name(device)}")

    Ns = [int(x) for x in args.N.split(",") if x.strip()]
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtypes = [dtype_map[x.strip()] for x in args.dtypes.split(",") if x.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    fh = args.out.open("a", newline="")
    cw = csv.writer(fh)
    if new_file:
        cw.writerow([
            "timestamp", "host", "gpu_model", "kernel",
            "N", "d", "H", "w", "G", "B",
            "rho_requested", "rho_achieved", "nnz",
            "ms_median", "ms_min", "ms_max", "mem_mb",
        ])
    ts   = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    host = socket.gethostname()
    gpu  = _gpu_model()

    for N in Ns:
        for dt in dtypes:
            tag = {torch.float32: "fp32",
                   torch.float16: "fp16",
                   torch.bfloat16: "bf16"}[dt]
            kernel_name = f"torch_sdpa_{tag}"
            try:
                ms_med, ms_min, ms_max, mem_mb = time_sdpa(
                    N, args.d, args.H, dt,
                    args.warmup, args.iters, device)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  {kernel_name:18s} N={N:>6d}  OOM ({e})")
                torch.cuda.empty_cache()
                continue
            cw.writerow([
                ts, host, gpu, kernel_name, N, args.d, args.H,
                "", "", "", "", "", "",
                f"{ms_med:.4f}", f"{ms_min:.4f}", f"{ms_max:.4f}",
                f"{mem_mb:.2f}",
            ])
            fh.flush()
            print(f"  {kernel_name:18s} N={N:>6d}  ms_median={ms_med:>9.4f}  "
                  f"mem_mb={mem_mb:>8.2f}")
            torch.cuda.empty_cache()

    fh.close()
    print(f"\nappended to {args.out}")


if __name__ == "__main__":
    main()
