"""Final-report crossover sweep: dense / windowed / block-sparse across N.

Calls the build/attn binary once per (kernel, N) cell, parses the new
mem_mb field, and emits one CSV. Intended to run on the ICME cluster
under the icme_final_crossover.slurm script up to N=32K.

Python 3.6 compatible (no future imports, no capture_output/text).

Usage:
    python3 bench/final_crossover.py \\
        --N 1024,2048,4096,8192,16384,32768 \\
        --H 16 --d 64 --w 256 --G 8 --rho 0.10 --B 64 \\
        --out report/data/final_crossover.csv
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


def _gpu_model():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ).decode().strip().splitlines()
        return out[0] if out else "unknown"
    except Exception:
        return "unknown"


def run_one(bin_path, kernel, N, d, H, **kw):
    cmd = [
        str(bin_path),
        "--kernel={}".format(kernel),
        "--N={}".format(N), "--d={}".format(d), "--H={}".format(H),
        "--warmup={}".format(kw.get("warmup", 5)),
        "--iters={}".format(kw.get("iters", 10)),
    ]
    if kernel == "windowed":
        cmd += ["--w={}".format(kw["w"]), "--G={}".format(kw["G"])]
    elif kernel == "blocksparse":
        cmd += ["--rho={}".format(kw["rho"]),
                "--B={}".format(kw["B"]),
                "--mask=random"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr, file=sys.stderr)
        sys.exit("binary failed: " + " ".join(cmd))
    parsed = {}
    for tok in res.stdout.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            parsed[k] = v
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin",  type=Path,
                    default=Path(os.environ.get("ATTN_BIN", DEFAULT_BIN)))
    ap.add_argument("--N",    default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--d",    type=int, default=64)
    ap.add_argument("--H",    type=int, default=16)
    ap.add_argument("--w",    type=int, default=256)
    ap.add_argument("--G",    type=int, default=8)
    ap.add_argument("--rho",  type=float, default=0.10)
    ap.add_argument("--B",    type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters",  type=int, default=10)
    ap.add_argument("--kernels", default="dense,windowed,blocksparse")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "report" / "data" / "final_crossover.csv")
    args = ap.parse_args()

    if not args.bin.exists():
        sys.exit("binary not found at {}; run `make all` first".format(args.bin))

    Ns = [int(x) for x in args.N.split(",") if x.strip()]
    kernels = [x.strip() for x in args.kernels.split(",") if x.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ts   = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    host = socket.gethostname()
    gpu  = _gpu_model()

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

    for N in Ns:
        for k in kernels:
            kw = dict(warmup=args.warmup, iters=args.iters)
            if k == "windowed":   kw.update(w=args.w, G=args.G)
            elif k == "blocksparse": kw.update(rho=args.rho, B=args.B)
            info = run_one(args.bin, k, N, args.d, args.H, **kw)
            cw.writerow([
                ts, host, gpu, k, N, args.d, args.H,
                args.w if k == "windowed" else "",
                args.G if k == "windowed" else "",
                args.B if k == "blocksparse" else "",
                "{:.4f}".format(args.rho) if k == "blocksparse" else "",
                info.get("rho", ""), info.get("nnz", ""),
                info.get("ms_median", ""),
                info.get("ms_min", ""),
                info.get("ms_max", ""),
                info.get("mem_mb", ""),
            ])
            fh.flush()
            print("  {:11s} N={:6d}  ms_median={}  mem_mb={}".format(
                k, N, info.get("ms_median", "?"), info.get("mem_mb", "?")))

    fh.close()
    print("\nappended to {}".format(args.out))


if __name__ == "__main__":
    main()
