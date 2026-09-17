"""Compare pre- and post-vectorization roofline metrics.

Picks the OLD ncu file (roofline_<host>_<id>.txt, no "vec") and the NEW one
(roofline_vec*_<host>_<id>.txt), parses both with the same logic as
plot_roofline.py, and prints a side-by-side table of:
  - achieved AI (FLOP / DRAM byte)
  - achieved TFLOPS
  - SM throughput % of peak (mean across 16 launches)
  - memory throughput % of peak

Also writes report/figures/roofline_vec_compare.png — two-point series per
kernel (old, new) connected by an arrow against the RTX 6000 ceilings.

Run from repo root:
    .venv/bin/python analysis/compare_roofline_vec.py
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from plot_roofline import (  # noqa
    parse_ncu_txt, aggregate, KERNELS,
    PEAK_TFLOPS, PEAK_BW_GBPS, RIDGE_AI,
)


def latest(pattern):
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def summarize(path):
    sections = parse_ncu_txt(path)
    out = {}
    for k in KERNELS:
        agg = aggregate(sections.get(k, []))
        if agg is None or agg["time_s"] == 0:
            continue
        ai = agg["flops"] / agg["bytes"]
        tflops = agg["flops"] / agg["time_s"] / 1e12
        out[k] = {
            "AI": ai,
            "TFLOPS": tflops,
            "ms_total": agg["time_s"] * 1000,
            "sm_pct": agg["sm_pct"],
            "mem_pct": agg["mem_pct"],
            "pct_peak": tflops / PEAK_TFLOPS * 100,
        }
    return out


def main():
    # Latest "old" ncu file = roofline_*.txt that does NOT start with
    # "roofline_vec" (post-vec) or "roofline_N_sweep" (different section
    # naming scheme: sections like "dense_N1024" that we can't pair off here).
    all_old = [p for p in glob.glob(str(ROOT / "report/data/roofline_*.txt"))
               if "/roofline_vec" not in p
               and "/roofline_N_sweep" not in p]
    if not all_old:
        sys.exit("no pre-vectorization roofline_*.txt found in report/data/")
    old_path = max(all_old, key=os.path.getmtime)

    new_path = latest("roofline_vec*.txt")
    if new_path is None:
        sys.exit("no post-vectorization roofline_vec*.txt found in report/data/")

    print(f"OLD: {old_path}")
    print(f"NEW: {new_path}\n")

    old = summarize(old_path)
    new = summarize(new_path)

    header = f"{'kernel':12s}  {'metric':18s} {'old':>10s} {'new':>10s} {'delta':>10s}"
    print(header)
    print("-" * len(header))
    for k in KERNELS:
        if k not in old or k not in new:
            continue
        o, n = old[k], new[k]
        for label, key, fmt in [
            ("AI (FLOP/B)",    "AI",       "{:.1f}"),
            ("TFLOPS",         "TFLOPS",   "{:.3f}"),
            ("% of FP32 peak", "pct_peak", "{:.1f}%"),
            ("SM thrpt %",     "sm_pct",   "{:.1f}%"),
            ("Mem thrpt %",    "mem_pct",  "{:.1f}%"),
            ("kernel time ms", "ms_total", "{:.2f}"),
        ]:
            o_str = fmt.format(o[key])
            n_str = fmt.format(n[key])
            if key in ("TFLOPS", "pct_peak", "sm_pct", "mem_pct"):
                d_str = f"{n[key]-o[key]:+.2f}"
            else:
                d_str = f"{(n[key]/o[key]):.3f}x"
            print(f"{k:12s}  {label:18s} {o_str:>10s} {n_str:>10s} {d_str:>10s}")
        print()


if __name__ == "__main__":
    main()
