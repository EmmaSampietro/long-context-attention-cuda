"""Roofline analysis for the three attention kernels on Quadro RTX 6000.

Parses `report/data/roofline_*.txt` (ncu CSV output from
`slurms/icme_roofline.slurm`), aggregates per-kernel metrics across the 16
launches (one per head, since H=16, iters=1), and writes:

  1. report/data/roofline_summary.csv    — per-kernel achieved AI / TFLOPS.
  2. report/figures/roofline.png         — log-log roofline with the three
                                           kernel points against the RTX 6000
                                           peak ceilings.

Run from repo root:
    .venv/bin/python analysis/plot_roofline.py
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Quadro RTX 6000 (Turing, sm_75) ceilings.
# FP32 peak: 16.3 TFLOPS (vendor); GDDR6 bandwidth: 624 GB/s (vendor).
PEAK_TFLOPS = 16.3
PEAK_BW_GBPS = 624.0
RIDGE_AI = (PEAK_TFLOPS * 1e12) / (PEAK_BW_GBPS * 1e9)  # ~26.1 FLOP/byte


KERNELS = ["dense", "windowed", "blocksparse"]
SECTION_RE = re.compile(r"^=+\s+(\w+)\s+=+$")
METRICS_OF_INTEREST = {
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": "fadd",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": "fmul",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma",
    "dram__bytes.sum":          "bytes",
    "gpu__time_duration.avg":   "time_ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem_pct",
}


def latest(pattern):
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        sys.exit(f"no file matching {pattern} in report/data/")
    return max(matches, key=os.path.getmtime)


def parse_ncu_txt(path):
    """Returns dict: kernel -> list of per-launch dicts with summed metrics."""
    sections = {k: [] for k in KERNELS}
    current = None
    # Per-launch accumulator keyed by row "ID" column.
    launch_acc = {}

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = SECTION_RE.match(line.strip())
            if m and m.group(1) in KERNELS:
                # flush previous section
                if current is not None and launch_acc:
                    sections[current] = list(launch_acc.values())
                current = m.group(1)
                launch_acc = {}
                continue
            if current is None:
                continue
            if not line.startswith('"'):
                continue
            # CSV row, but very simple — all fields are "..." quoted.
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 15:
                continue
            try:
                launch_id = int(parts[0])
            except ValueError:
                continue
            metric = parts[12]
            value_str = parts[14]
            short = METRICS_OF_INTEREST.get(metric)
            if short is None:
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            launch = launch_acc.setdefault(launch_id, {})
            # ncu sometimes splits a single launch's metrics across multiple
            # rows; later occurrences should overwrite earlier (same metric
            # within one launch is just the same number repeated).
            launch[short] = value

        # flush last section
        if current is not None and launch_acc:
            sections[current] = list(launch_acc.values())

    return sections


def aggregate(launches):
    """Sum across all launches; returns (flops, bytes, time_s, mean_sm_pct,
    mean_mem_pct, n_launches)."""
    if not launches:
        return None
    flops = 0.0
    byts  = 0.0
    time_ns = 0.0
    sm_pct = []
    mem_pct = []
    for l in launches:
        fadd  = l.get("fadd",  0.0)
        fmul  = l.get("fmul",  0.0)
        ffma  = l.get("ffma",  0.0)
        flops += fadd + fmul + 2.0 * ffma
        byts  += l.get("bytes", 0.0)
        time_ns += l.get("time_ns", 0.0)
        if "sm_pct" in l:  sm_pct.append(l["sm_pct"])
        if "mem_pct" in l: mem_pct.append(l["mem_pct"])
    return {
        "flops":   flops,
        "bytes":   byts,
        "time_s":  time_ns * 1e-9,
        "sm_pct":  float(np.mean(sm_pct))  if sm_pct  else float("nan"),
        "mem_pct": float(np.mean(mem_pct)) if mem_pct else float("nan"),
        "n":       len(launches),
    }


def write_summary_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kernel", "n_launches", "flops_total", "bytes_total",
                    "time_total_s", "achieved_AI_FLOP_per_byte",
                    "achieved_TFLOPS", "sm_pct_mean", "mem_pct_mean",
                    "frac_of_FP32_peak", "frac_of_BW_peak"])
        for r in rows:
            w.writerow([
                r["kernel"], r["n"], f"{r['flops']:.6e}",
                f"{r['bytes']:.6e}", f"{r['time_s']:.6e}",
                f"{r['AI']:.4f}", f"{r['TFLOPS']:.4f}",
                f"{r['sm_pct']:.2f}", f"{r['mem_pct']:.2f}",
                f"{r['TFLOPS'] / PEAK_TFLOPS:.4f}",
                f"{(r['bytes']/r['time_s'])/(PEAK_BW_GBPS*1e9):.4f}",
            ])


def plot_roofline(rows, out_path):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    # AI axis range — covers both ends with margin.
    ai_lo, ai_hi = 0.5, 3000.0
    ai = np.geomspace(ai_lo, ai_hi, 400)

    # Compute ceiling: peak FP32 (flat horizontal in TFLOPS).
    compute_ceiling = np.full_like(ai, PEAK_TFLOPS)
    # Memory ceiling: AI * BW (in TFLOPS, since BW in B/s × FLOP/B = FLOP/s).
    mem_ceiling = ai * PEAK_BW_GBPS / 1000.0  # GB/s × FLOP/B / 1000 → TFLOPS
    roofline = np.minimum(compute_ceiling, mem_ceiling)

    ax.plot(ai, roofline, color="black", linewidth=2.0, zorder=2)
    ax.plot(ai, compute_ceiling, color="gray", linestyle="--", linewidth=1.0,
            alpha=0.6, zorder=1)
    ax.plot(ai, mem_ceiling,     color="gray", linestyle="--", linewidth=1.0,
            alpha=0.6, zorder=1)
    ax.axvline(RIDGE_AI, color="gray", linestyle=":", linewidth=1.0,
               alpha=0.6, zorder=1)

    ax.text(ai_hi * 0.5, PEAK_TFLOPS * 1.08,
            f"Peak FP32 = {PEAK_TFLOPS:.1f} TFLOPS",
            color="dimgray", fontsize=9, ha="right")
    ax.text(0.7, 0.7 * PEAK_BW_GBPS / 1000.0,
            f"Peak BW = {PEAK_BW_GBPS:.0f} GB/s",
            color="dimgray", fontsize=9, ha="left", rotation=32)
    ax.text(RIDGE_AI, 0.04, f"ridge ≈ {RIDGE_AI:.1f}",
            color="dimgray", fontsize=8, ha="center")

    colors = {"dense": "#c0392b", "windowed": "#2c3e50", "blocksparse": "#16a085"}
    markers = {"dense": "o", "windowed": "s", "blocksparse": "^"}
    for r in rows:
        k = r["kernel"]
        ax.scatter([r["AI"]], [r["TFLOPS"]],
                   marker=markers[k], color=colors[k], s=130,
                   edgecolors="black", linewidths=0.8, zorder=5,
                   label=f"{k} (AI={r['AI']:.1f}, {r['TFLOPS']:.2f} TFLOPS)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_lo, ai_hi)
    ax.set_ylim(0.02, 40.0)
    ax.set_xlabel("Arithmetic Intensity (FLOP / DRAM byte)")
    ax.set_ylabel("Achieved performance (TFLOPS)")
    ax.set_title("Roofline — Quadro RTX 6000 (Turing, sm_75)\nN=4096, H=16, d=64, FP32")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path", type=Path, default=None,
                    help="ncu .txt to parse (default: latest report/data/roofline_*.txt)")
    ap.add_argument("--out-fig", type=Path,
                    default=ROOT / "report" / "figures" / "roofline.png")
    ap.add_argument("--out-csv", type=Path,
                    default=ROOT / "report" / "data" / "roofline_summary.csv")
    args = ap.parse_args()

    path = args.in_path if args.in_path else Path(latest("roofline_*.txt"))
    print(f"parsing {path}")
    sections = parse_ncu_txt(path)

    rows = []
    for k in KERNELS:
        agg = aggregate(sections.get(k, []))
        if agg is None or agg["time_s"] == 0:
            print(f"  WARNING: no data for {k}")
            continue
        ai = agg["flops"] / agg["bytes"]
        tflops = agg["flops"] / agg["time_s"] / 1e12
        rows.append({"kernel": k, **agg, "AI": ai, "TFLOPS": tflops})
        print(f"  {k:11s} n={agg['n']:>2d}  FLOPs={agg['flops']:.3e}  "
              f"bytes={agg['bytes']:.3e}  time={agg['time_s']*1e3:.3f} ms  "
              f"AI={ai:7.2f}  TFLOPS={tflops:6.3f}  "
              f"({tflops/PEAK_TFLOPS*100:.1f}% of FP32 peak, "
              f"{(agg['bytes']/agg['time_s'])/(PEAK_BW_GBPS*1e9)*100:.1f}% of BW peak)")

    write_summary_csv(rows, args.out_csv)
    print(f"wrote {args.out_csv.relative_to(ROOT)}")

    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plot_roofline(rows, args.out_fig)
    print(f"wrote {args.out_fig.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
