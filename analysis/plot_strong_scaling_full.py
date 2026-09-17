"""Unified strong-scaling figure: every (kernel, precision, driver, N) combination
in the project, on one chart per head-dim.

Two panels:
  Left:  d=64 strong scaling, 3-panel grid (N=4K, 8K, 16K)
  Right: d=128 strong scaling (same N if available)

The headline visual: seq-par windowed (solid lines) tracks near-linear in all
panels; every head-par config (dashed) hits a comm wall whose location depends
on kernel cost / N.

Data sources:
  - report/data/icme_wmma_mpi_*.csv         -- seq-par windowed FP32 + FP16 d=64
  - report/data/icme_headpar_full_d64_*.csv -- head-par dense_fp16, bsp, bsp_fp16 d=64
  - report/data/icme_mpi_d128_*.csv         -- everything at d=128

Output: report/figures/strong_scaling_full.png
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def latest(pattern):
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_all():
    """Combine all MPI scaling CSVs into a single normalised DataFrame.

    Output schema: driver, kernel, N, d, ranks, ms_per_iter.
    """
    frames = []

    # 1) WMMA-MPI sweep: seq-par windowed FP32 + FP16 at d=64
    p = latest("icme_wmma_mpi_*.csv")
    if p:
        print(f"  wmma_mpi:           {os.path.basename(p)}")
        df = pd.read_csv(p)
        df["driver"] = "seqpar"
        df["d"] = 64
        frames.append(df[["driver", "kernel", "N", "d", "ranks", "ms_per_iter"]])

    # 2) Headpar full d=64: dense_fp16, bsp, bsp_fp16
    p = latest("icme_headpar_full_d64_*.csv")
    if p:
        print(f"  headpar_full_d64:   {os.path.basename(p)}")
        df = pd.read_csv(p)
        df["driver"] = "headpar"
        frames.append(df[["driver", "kernel", "N", "d", "ranks", "ms_per_iter"]])

    # 3) d=128 MPI sweep (seq-par windowed + head-par dense/bsp at d=128).
    # Load BOTH the original full sweep and any later fill-in CSVs. The
    # original file name is icme_mpi_d128_<host>_<jobid>.csv; the fill is
    # icme_mpi_d128_fill_<host>_<jobid>.csv (extends bsp to N=16K and dense
    # to N=8K). Globbing icme_mpi_d128_*.csv picks up both.
    for p in sorted(glob.glob(str(ROOT / "report" / "data" / "icme_mpi_d128_*.csv"))):
        print(f"  mpi_d128:           {os.path.basename(p)}")
        df = pd.read_csv(p)
        frames.append(df[["driver", "kernel", "N", "d", "ranks", "ms_per_iter"]])

    # 4) Original M4 data (head-par dense, head-par windowed, seq-par windowed at d=64)
    p = latest("m4_scaling.csv")
    if p:
        print(f"  m4_scaling:         {os.path.basename(p)}")
        df = pd.read_csv(p)
        df = df[df["mode"] == "strong"]
        df = df.rename(columns={"strategy": "driver"})
        frames.append(df[["driver", "kernel", "N", "d", "ranks", "ms_per_iter"]])

    out = pd.concat(frames, ignore_index=True)
    out["N"] = pd.to_numeric(out["N"])
    out["d"] = pd.to_numeric(out["d"])
    out["ranks"] = pd.to_numeric(out["ranks"])
    out["ms_per_iter"] = pd.to_numeric(out["ms_per_iter"])
    # Drop NaN ranks (failed runs)
    out = out.dropna(subset=["ms_per_iter"])
    # Dedupe in case the same (driver, kernel, N, d, ranks) appears in two CSVs
    out = out.drop_duplicates(
        subset=["driver", "kernel", "N", "d", "ranks"], keep="last"
    )
    return out


# Color & linestyle palette: color encodes the kernel family; linestyle encodes
# the driver (solid = seq-par, dashed = head-par); marker fill encodes precision.
KERNEL_COLOR = {
    "dense":            "#c0392b",   # red
    "dense_fp16":       "#e67e22",   # orange
    "windowed":         "#27ae60",   # green
    "windowed_fp16":    "#16a085",   # teal
    "blocksparse":      "#2c3e50",   # dark slate
    "blocksparse_fp16": "#7f8c8d",   # mid grey
}
DRIVER_LS = {"seqpar": "-", "headpar": "--"}
DRIVER_MARKER = {"seqpar": "o", "headpar": "s"}


def style(driver, kernel):
    color = KERNEL_COLOR.get(kernel, "#000000")
    ls = DRIVER_LS.get(driver, ":")
    marker = DRIVER_MARKER.get(driver, "x")
    label = f"{driver} {kernel}"
    return color, ls, marker, label


def make_panel(ax, df, N, d, title):
    """Plot speedup vs ranks for every (driver, kernel) at this (N, d)."""
    sub = df[(df["N"] == N) & (df["d"] == d)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        ax.set_title(title); return

    # Reference: ideal linear speedup
    ranks_axis = sorted(sub["ranks"].unique())
    ax.plot(ranks_axis, ranks_axis, "k:", lw=0.8, label="ideal", zorder=1)

    # One line per (driver, kernel)
    for (driver, kernel), g in sub.groupby(["driver", "kernel"]):
        if 1 not in g["ranks"].values:
            continue   # need P=1 to compute speedup
        base = float(g[g["ranks"] == 1]["ms_per_iter"].iloc[0])
        g = g.sort_values("ranks")
        speedups = base / g["ms_per_iter"].values
        color, ls, marker, label = style(driver, kernel)
        ax.plot(g["ranks"].values, speedups,
                color=color, linestyle=ls, marker=marker, lw=1.4,
                markersize=5, label=label)

    ax.set_xticks([1, 2, 4])
    ax.set_xlim(0.8, 4.4)
    ax.set_ylim(0, 4.4)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("ranks")


def main():
    print("loading scaling data...")
    df = load_all()
    print(f"  combined rows: {len(df)}")
    print(f"  configs: {df.groupby(['driver','kernel','d']).size().to_dict()}")

    # Three N panels at d=64; one wide d=128 panel underneath
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey=True)

    # Top row: d=64
    for ax, N in zip(axes[0], [4096, 8192, 16384]):
        make_panel(ax, df, N, 64, title=f"d=64, N={N}")
    axes[0][0].set_ylabel("speedup vs 1 rank")

    # Bottom row: d=128
    for ax, N in zip(axes[1], [4096, 8192, 16384]):
        make_panel(ax, df, N, 128, title=f"d=128, N={N}")
    axes[1][0].set_ylabel("speedup vs 1 rank")

    # One unified legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("Strong scaling across kernels, head dims, and drivers (H=16)", y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])

    out = ROOT / "report" / "figures" / "strong_scaling_full.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
