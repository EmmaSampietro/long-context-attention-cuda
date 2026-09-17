"""Roofline at multiple sequence lengths.

Reads `report/data/roofline_N_sweep_*.txt` (produced by
`slurms/icme_roofline_N_sweep.slurm`), aggregates per (kernel, N) across the
H launches, and writes:

  1. report/data/roofline_sweep_summary.csv
  2. report/figures/roofline_N_sweep.png      — log-log roofline; each kernel
                                                 is a line connecting its N
                                                 points against RTX 6000
                                                 ceilings.
  3. report/figures/roofline_AI_vs_N.png      — AI vs N to show dense's
                                                 linear growth vs windowed /
                                                 block-sparse staying flat.

Run from repo root:
    .venv/bin/python analysis/plot_roofline_sweep.py
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

PEAK_TFLOPS = 16.3
PEAK_BW_GBPS = 624.0
RIDGE_AI = (PEAK_TFLOPS * 1e12) / (PEAK_BW_GBPS * 1e9)

KERNELS = ["dense", "windowed", "blocksparse"]
SECTION_RE = re.compile(r"^=+\s+(dense|windowed|blocksparse)_N(\d+)\s+=+$")
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
    """Returns dict: (kernel, N) -> list of per-launch dicts."""
    sections = {}
    current = None
    launch_acc = {}

    def flush():
        if current is not None and launch_acc:
            sections[current] = list(launch_acc.values())

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = SECTION_RE.match(line.strip())
            if m:
                flush()
                current = (m.group(1), int(m.group(2)))
                launch_acc = {}
                continue
            if current is None or not line.startswith('"'):
                continue
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 15:
                continue
            try:
                launch_id = int(parts[0])
            except ValueError:
                continue
            short = METRICS_OF_INTEREST.get(parts[12])
            if short is None:
                continue
            try:
                value = float(parts[14])
            except ValueError:
                continue
            launch_acc.setdefault(launch_id, {})[short] = value
        flush()

    return sections


def aggregate(launches):
    if not launches:
        return None
    flops = 0.0; byts = 0.0; time_ns = 0.0
    sm_pct = []; mem_pct = []
    for l in launches:
        flops += l.get("fadd", 0.0) + l.get("fmul", 0.0) + 2.0 * l.get("ffma", 0.0)
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
        "n_launches": len(launches),
    }


def write_summary_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kernel", "N", "n_launches", "flops_total", "bytes_total",
                    "time_total_s", "achieved_AI_FLOP_per_byte",
                    "achieved_TFLOPS", "sm_pct_mean", "mem_pct_mean",
                    "frac_of_FP32_peak", "frac_of_BW_peak"])
        for r in rows:
            w.writerow([
                r["kernel"], r["N"], r["n_launches"],
                f"{r['flops']:.6e}", f"{r['bytes']:.6e}",
                f"{r['time_s']:.6e}",
                f"{r['AI']:.4f}", f"{r['TFLOPS']:.4f}",
                f"{r['sm_pct']:.2f}", f"{r['mem_pct']:.2f}",
                f"{r['TFLOPS'] / PEAK_TFLOPS:.4f}",
                f"{(r['bytes']/r['time_s'])/(PEAK_BW_GBPS*1e9):.4f}",
            ])


def plot_roofline(rows, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ai_lo, ai_hi = 0.5, 5000.0
    ai = np.geomspace(ai_lo, ai_hi, 400)
    compute_ceiling = np.full_like(ai, PEAK_TFLOPS)
    mem_ceiling     = ai * PEAK_BW_GBPS / 1000.0
    roofline = np.minimum(compute_ceiling, mem_ceiling)

    ax.plot(ai, roofline, color="black", linewidth=2.0, zorder=2)
    ax.plot(ai, compute_ceiling, color="gray", linestyle="--", linewidth=1.0,
            alpha=0.6, zorder=1)
    ax.plot(ai, mem_ceiling, color="gray", linestyle="--", linewidth=1.0,
            alpha=0.6, zorder=1)
    ax.axvline(RIDGE_AI, color="gray", linestyle=":", linewidth=1.0,
               alpha=0.6, zorder=1)
    ax.text(ai_hi * 0.5, PEAK_TFLOPS * 1.08,
            f"Peak FP32 = {PEAK_TFLOPS:.1f} TFLOPS",
            color="dimgray", fontsize=9, ha="right")
    ax.text(0.7, 0.7 * PEAK_BW_GBPS / 1000.0,
            f"Peak BW = {PEAK_BW_GBPS:.0f} GB/s",
            color="dimgray", fontsize=9, ha="left", rotation=32)
    ax.text(RIDGE_AI, 0.025, f"ridge ≈ {RIDGE_AI:.1f}",
            color="dimgray", fontsize=8, ha="center")

    colors  = {"dense": "#c0392b", "windowed": "#2c3e50", "blocksparse": "#16a085"}
    markers = {"dense": "o",       "windowed": "s",       "blocksparse": "^"}

    by_kernel = {k: [] for k in KERNELS}
    for r in rows:
        by_kernel[r["kernel"]].append(r)
    for k in KERNELS:
        pts = sorted(by_kernel[k], key=lambda r: r["N"])
        if not pts:
            continue
        xs = [r["AI"] for r in pts]
        ys = [r["TFLOPS"] for r in pts]
        ax.plot(xs, ys, color=colors[k], linestyle="-", linewidth=1.4,
                marker=markers[k], markersize=10, markeredgecolor="black",
                markeredgewidth=0.7, zorder=5, label=k, alpha=0.95)
        for r in pts:
            ax.annotate(f"N={r['N']}", (r["AI"], r["TFLOPS"]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=7, color=colors[k])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_lo, ai_hi)
    ax.set_ylim(0.015, 40.0)
    ax.set_xlabel("Arithmetic Intensity (FLOP / DRAM byte)")
    ax.set_ylabel("Achieved performance (TFLOPS)")
    ax.set_title("Roofline across N — Quadro RTX 6000 (sm_75), H=16, d=64, FP32")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ai_vs_N(rows, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors  = {"dense": "#c0392b", "windowed": "#2c3e50", "blocksparse": "#16a085"}
    markers = {"dense": "o",       "windowed": "s",       "blocksparse": "^"}
    by_kernel = {k: [] for k in KERNELS}
    for r in rows:
        by_kernel[r["kernel"]].append(r)
    for k in KERNELS:
        pts = sorted(by_kernel[k], key=lambda r: r["N"])
        if not pts:
            continue
        ax.plot([r["N"] for r in pts], [r["AI"] for r in pts],
                marker=markers[k], color=colors[k], linewidth=2,
                markersize=8, label=k)
    ax.axhline(RIDGE_AI, color="gray", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"ridge ≈ {RIDGE_AI:.1f}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Achieved Arithmetic Intensity (FLOP / byte)")
    ax.set_title("Arithmetic intensity vs N — H=16, d=64, FP32")
    ax.grid(True, which="both", alpha=0.3)
    all_N = sorted(set(r["N"] for r in rows))
    ax.set_xticks(all_N)
    ax.set_xticklabels([str(n) for n in all_N])
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path", type=Path, default=None,
                    help="ncu .txt to parse (default: latest roofline_N_sweep_*.txt)")
    ap.add_argument("--out-fig", type=Path,
                    default=ROOT / "report" / "figures" / "roofline_N_sweep.png")
    ap.add_argument("--out-fig-ai", type=Path,
                    default=ROOT / "report" / "figures" / "roofline_AI_vs_N.png")
    ap.add_argument("--out-csv", type=Path,
                    default=ROOT / "report" / "data" / "roofline_sweep_summary.csv")
    args = ap.parse_args()

    path = args.in_path if args.in_path else Path(latest("roofline_N_sweep_*.txt"))
    print(f"parsing {path}")
    sections = parse_ncu_txt(path)

    rows = []
    for (k, N), launches in sorted(sections.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        agg = aggregate(launches)
        if agg is None or agg["time_s"] == 0:
            print(f"  WARNING: no data for {k} N={N}")
            continue
        ai = agg["flops"] / agg["bytes"]
        tflops = agg["flops"] / agg["time_s"] / 1e12
        rows.append({"kernel": k, "N": N, **agg, "AI": ai, "TFLOPS": tflops})
        print(f"  {k:11s} N={N:>6d}  n={agg['n_launches']:>2d}  "
              f"FLOPs={agg['flops']:.3e}  bytes={agg['bytes']:.3e}  "
              f"time={agg['time_s']*1e3:.3f} ms  "
              f"AI={ai:8.2f}  TFLOPS={tflops:6.3f}  "
              f"({tflops/PEAK_TFLOPS*100:.1f}% peak, "
              f"{(agg['bytes']/agg['time_s'])/(PEAK_BW_GBPS*1e9)*100:.1f}% BW)")

    write_summary_csv(rows, args.out_csv)
    print(f"wrote {args.out_csv.relative_to(ROOT)}")

    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plot_roofline(rows, args.out_fig)
    print(f"wrote {args.out_fig.relative_to(ROOT)}")

    plot_ai_vs_N(rows, args.out_fig_ai)
    print(f"wrote {args.out_fig_ai.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
