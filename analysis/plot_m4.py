"""Generate Milestone 4 scaling figures from report/data/m4_scaling.csv.

Outputs (to report/figures/):
  1. m4_strong_scaling.png  — strong-scaling speedup, all (strategy, kernel, N)
                              configurations on one chart, with ideal-linear ref.
                              The headline figure: windowed-headpar flat near 1x,
                              windowed-seqpar tracking near-linear.
  2. m4_efficiency.png      — parallel efficiency (%) for the same configurations.
  3. m4_weak_scaling.png    — head-parallel weak scaling (4 heads/rank).

Run from repo root:
    .venv/bin/python analysis/plot_m4.py
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(pattern, label):
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        sys.exit(f"no file matching {pattern} in report/data/")
    path = max(matches, key=os.path.getmtime)
    print(f"  {label}: {os.path.basename(path)}")
    return pd.read_csv(path)


def style_for(strategy, kernel, N):
    """Return (color, marker, linestyle, label)."""
    colors = {
        ("headpar", "dense",    4096): "#c0392b",   # red
        ("headpar", "dense",    8192): "#e67e22",   # orange
        ("headpar", "windowed", 4096): "#34495e",   # dark slate
        ("headpar", "windowed", 8192): "#7f8c8d",   # gray
        ("seqpar",  "windowed", 4096): "#16a085",   # teal
        ("seqpar",  "windowed", 8192): "#27ae60",   # green
    }
    markers = {"headpar": "o", "seqpar": "s"}
    linestyles = {"dense": "-", "windowed": "--" if strategy == "headpar" else "-"}
    kernel_disp = kernel.capitalize()
    strat_disp = "head-par" if strategy == "headpar" else "seq-par"
    label = f"{kernel_disp} {strat_disp} N={int(N)}"
    return (colors.get((strategy, kernel, int(N)), "#888"),
            markers[strategy],
            linestyles[kernel],
            label)


def plot_strong_scaling(df, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    ranks_all = sorted(df["ranks"].unique())

    for (strategy, kernel, N), grp in df.groupby(["strategy", "kernel", "N"]):
        grp = grp.sort_values("ranks").reset_index(drop=True)
        t1 = grp.loc[grp["ranks"] == 1, "ms_per_iter"].iloc[0]
        grp["speedup"] = t1 / grp["ms_per_iter"]
        color, marker, ls, label = style_for(strategy, kernel, N)
        ax.plot(grp["ranks"], grp["speedup"],
                marker=marker, linestyle=ls, linewidth=2,
                color=color, label=label, markersize=7)

    # Ideal linear reference
    ax.plot(ranks_all, ranks_all, linestyle=":", color="#888", linewidth=1.5,
            label="Ideal (linear)")

    ax.set_xlabel("Ranks (GPUs)")
    ax.set_ylabel("Speedup (over 1 rank, same config)")
    ax.set_title("Strong scaling — H=16, d=64, w=256, FP32")
    ax.set_xticks(ranks_all)
    ax.set_xticklabels([str(r) for r in ranks_all])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_efficiency(df, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ranks_all = sorted(df["ranks"].unique())

    for (strategy, kernel, N), grp in df.groupby(["strategy", "kernel", "N"]):
        grp = grp.sort_values("ranks").reset_index(drop=True)
        t1 = grp.loc[grp["ranks"] == 1, "ms_per_iter"].iloc[0]
        grp["efficiency"] = (t1 / grp["ms_per_iter"]) / grp["ranks"] * 100.0
        color, marker, ls, label = style_for(strategy, kernel, N)
        ax.plot(grp["ranks"], grp["efficiency"],
                marker=marker, linestyle=ls, linewidth=2,
                color=color, label=label, markersize=7)

    ax.axhline(100, linestyle=":", color="#888", linewidth=1.5, label="Ideal (100%)")
    ax.set_xlabel("Ranks (GPUs)")
    ax.set_ylabel("Parallel efficiency (%)")
    ax.set_title("Parallel efficiency — H=16, d=64, w=256, FP32")
    ax.set_xticks(ranks_all); ax.set_xticklabels([str(r) for r in ranks_all])
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_weak_scaling(df, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    df = df.sort_values("ranks").reset_index(drop=True)
    t1 = df.loc[df["ranks"] == 1, "ms_per_iter"].iloc[0]

    ax.plot(df["ranks"], df["ms_per_iter"], marker="o", linewidth=2,
            color="#c0392b", label="Measured (head-par dense)")
    ax.axhline(t1, linestyle=":", color="#888", linewidth=1.5,
               label=f"Ideal (constant {t1:.2f} ms)")

    for _, row in df.iterrows():
        ax.annotate(f"H={int(row['H'])}",
                    xy=(row["ranks"], row["ms_per_iter"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9)

    ax.set_xlabel("Ranks (GPUs)")
    ax.set_ylabel("Time per iteration (ms)")
    ax.set_title("Weak scaling — 4 heads / rank, dense N=4096")
    ax.set_xticks(df["ranks"].tolist())
    ax.set_xticklabels([str(int(r)) for r in df["ranks"].tolist()])
    ax.set_ylim(0, df["ms_per_iter"].max() * 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "report" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs:")
    df = load("m4_scaling.csv", "m4 scaling")

    strong = df[df["mode"] == "strong"].copy()
    weak   = df[df["mode"] == "weak"].copy()

    print("\nWriting figures:")
    out1 = args.out_dir / "m4_strong_scaling.png"
    plot_strong_scaling(strong, out1)
    print(f"  {out1.relative_to(ROOT)}")

    out2 = args.out_dir / "m4_efficiency.png"
    plot_efficiency(strong, out2)
    print(f"  {out2.relative_to(ROOT)}")

    out3 = args.out_dir / "m4_weak_scaling.png"
    plot_weak_scaling(weak, out3)
    print(f"  {out3.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
