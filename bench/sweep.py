"""Minimal benchmark sweep — first-patch version.

Just enough to run the dense kernel across a range of N and emit a CSV.
Reads no YAML yet; takes the sweep on the command line. The fuller
configs/bench_matrix.yaml driver lands once the windowed and block-sparse
kernels are functional.

Usage:
    python bench/sweep.py --kernel dense --N 512,1024,2048,4096 --d 64 --H 16 \
        --out report/data/dense_baseline.csv
"""
import argparse
import csv
import datetime as _dt
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "build" / "attn"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _gpu_model() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).decode().strip().splitlines()
        return out[0] if out else "unknown"
    except Exception:
        return "unknown"


def run_one(bin_path: Path, kernel: str, N: int, d: int, H: int,
            *, w: int = 0, G: int = 0,
            warmup: int = 5, iters: int = 10) -> dict:
    cmd = [
        str(bin_path),
        f"--kernel={kernel}", f"--N={N}", f"--d={d}", f"--H={H}",
        f"--w={w}", f"--G={G}",
        f"--warmup={warmup}", f"--iters={iters}",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr, file=sys.stderr)
        sys.exit(f"binary failed: {' '.join(cmd)}")
    parsed = {}
    for tok in res.stdout.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            parsed[k] = v
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=Path(os.environ.get("ATTN_BIN", DEFAULT_BIN)))
    ap.add_argument("--kernel", default="dense")
    ap.add_argument("--N", default="512,1024,2048,4096",
                    help="comma-separated list of sequence lengths to sweep")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--w", type=int, default=0,
                    help="windowed: half-width (only used for kernel=windowed)")
    ap.add_argument("--G", type=int, default=0,
                    help="windowed: number of leading global tokens")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters",  type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "report" / "data" / "dense_sweep.csv")
    args = ap.parse_args()

    if not args.bin.exists():
        sys.exit(f"binary not found at {args.bin}; run `make all` first")

    Ns = [int(x) for x in args.N.split(",") if x.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    new_file = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        cw = csv.writer(fh)
        if new_file:
            cw.writerow([
                "timestamp", "host", "git_commit", "gpu_model", "kernel",
                "N", "d", "H", "w", "G", "ms_median", "ms_min", "ms_max",
            ])
        ts  = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        gpu = _gpu_model()
        commit = _git_commit()
        host = socket.gethostname()
        for N in Ns:
            info = run_one(args.bin, args.kernel, N, args.d, args.H,
                           w=args.w, G=args.G,
                           warmup=args.warmup, iters=args.iters)
            cw.writerow([
                ts, host, commit, gpu, args.kernel,
                N, args.d, args.H, args.w, args.G,
                info.get("ms_median", ""),
                info.get("ms_min", ""),
                info.get("ms_max", ""),
            ])
            print(f"  N={N:5d} w={args.w:5d} G={args.G:3d}  ms_median={info.get('ms_median')}")

    print(f"\nappended to {args.out}")


if __name__ == "__main__":
    main()
