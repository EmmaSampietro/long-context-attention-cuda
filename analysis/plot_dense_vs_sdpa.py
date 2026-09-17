"""Merged kernel-vs-SDPA + asymptotic comparison figure for the final report.

Shows on one chart: the FP32 asymptotic story (block-sparse FP32, windowed FP32),
the WMMA implementation lift (dense FP16 WMMA, windowed FP16 WMMA), and the
production reference (SDPA-FP16). With O(N) and O(N^2) reference slopes.

Five lines total:
  - Our dense FP16 WMMA          (orange,  circle)
  - Our windowed FP32 chunked    (green,   square, dashed)  -- baseline
  - Our windowed FP16 WMMA       (teal,    square, solid)   -- THE WINNER
  - Our block-sparse FP32        (slate,   triangle)        -- asymptotic
  - SDPA-FP16                    (purple,  diamond)         -- reference

Reads:
  - final_crossover_chunked_*.csv  : dense, windowed, blocksparse FP32
  - dense_fp16_sweep_*.csv         : dense FP16 WMMA
  - windowed_fp16_sweep_*.csv      : windowed FP16 WMMA
  - sdpa_baseline_*.csv            : SDPA FP16 (and FP32)
  - n64k_*.csv, n128k_*.csv        : long-context extensions

Writes:
  - report/figures/dense_vs_sdpa.png

Run from repo root:
    .venv/bin/python analysis/plot_dense_vs_sdpa.py
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Canonical kernel colors (kernel family = color; precision = linestyle/marker).
SERIES = [
    {"kernel": "blocksparse",      "label": "Block-sparse FP32",         "color": "#2c3e50", "marker": "^", "ls": "-"},
    {"kernel": "windowed",         "label": "Windowed FP32",             "color": "#27ae60", "marker": "s", "ls": "--"},
    {"kernel": "dense_fp16",       "label": "Dense FP16 WMMA",           "color": "#e67e22", "marker": "o", "ls": "-"},
    {"kernel": "windowed_fp16",    "label": "Windowed FP16 WMMA (ours)", "color": "#16a085", "marker": "s", "ls": "-"},
    {"kernel": "torch_sdpa_fp16",  "label": "SDPA-FP16 (reference)",     "color": "#8e44ad", "marker": "D", "ls": "-"},
]


def latest(pattern):
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_combined():
    frames = []

    # FP32 baselines (dense, windowed, blocksparse)
    p = latest("final_crossover_chunked_*.csv")
    if p is None:
        sys.exit("no final_crossover_chunked_*.csv in report/data/")
    print(f"  FP32 chunked: {os.path.basename(p)}")
    df = pd.read_csv(p)
    frames.append(df[["kernel", "N", "ms_median"]])

    # FP16 dense WMMA
    p = latest("dense_fp16_sweep_*.csv")
    if p:
        print(f"  dense_fp16:    {os.path.basename(p)}")
        frames.append(pd.read_csv(p)[["kernel", "N", "ms_median"]])

    # FP16 windowed WMMA
    p = latest("windowed_fp16_sweep_*.csv")
    if p:
        print(f"  windowed_fp16: {os.path.basename(p)}")
        frames.append(pd.read_csv(p)[["kernel", "N", "ms_median"]])

    # SDPA baseline (FP16 + FP32)
    p = latest("sdpa_baseline_*.csv")
    if p:
        print(f"  sdpa_baseline: {os.path.basename(p)}")
        frames.append(pd.read_csv(p)[["kernel", "N", "ms_median"]])

    # Long-context extensions
    for pat in ["n64k_*.csv", "n128k_*.csv"]:
        for path in sorted(glob.glob(str(ROOT / "report" / "data" / pat))):
            print(f"  ext: {os.path.basename(path)}")
            frames.append(pd.read_csv(path)[["kernel", "N", "ms_median"]])

    df = pd.concat(frames, ignore_index=True)
    df["N"] = pd.to_numeric(df["N"])
    df["ms_median"] = pd.to_numeric(df["ms_median"])
    df = df.drop_duplicates(subset=["kernel", "N"], keep="last")
    return df


def plot_time(df, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # Plot each series
    for s in SERIES:
        sub = df[df["kernel"] == s["kernel"]].sort_values("N")
        if sub.empty:
            continue
        ax.loglog(sub["N"].values, sub["ms_median"].values,
                  color=s["color"], linestyle=s["ls"], marker=s["marker"],
                  linewidth=1.8, markersize=6.5, label=s["label"])

    # Asymptotic reference slopes.
    Ns_all = sorted(df["N"].unique())
    if len(Ns_all) >= 2:
        N0, N1 = Ns_all[0], Ns_all[-1]
        # O(N) anchored at windowed FP16 at N0
        win_fp16 = df[(df["kernel"] == "windowed_fp16") & (df["N"] == N0)]
        if not win_fp16.empty:
            t0 = float(win_fp16["ms_median"].iloc[0])
            ax.plot([N0, N1], [t0, t0 * (N1 / N0)],
                    color="grey", linestyle=":", lw=0.9, alpha=0.6)
            ax.text(N1 * 1.05, t0 * (N1 / N0), r"$O(N)$",
                    color="grey", fontsize=9, va="center")
        # O(N^2) anchored at SDPA-FP16 at N0 (or dense FP16 if missing)
        ref = df[(df["kernel"] == "torch_sdpa_fp16") & (df["N"] == N0)]
        if ref.empty:
            ref = df[(df["kernel"] == "dense_fp16") & (df["N"] == N0)]
        if not ref.empty:
            t0 = float(ref["ms_median"].iloc[0])
            ax.plot([N0, N1], [t0, t0 * (N1 / N0) ** 2],
                    color="grey", linestyle=":", lw=0.9, alpha=0.6)
            ax.text(N1 * 1.05, t0 * (N1 / N0) ** 2, r"$O(N^2)$",
                    color="grey", fontsize=9, va="center")

    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Wall time (ms, median of 10)")
    ax.set_title("Our kernels vs PyTorch SDPA — H=16, d=64, single GPU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    if len(Ns_all) >= 2:
        ax.set_xlim(Ns_all[0] * 0.85, Ns_all[-1] * 1.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("loading data...")
    df = load_combined()
    print()
    pivot = df.pivot_table(index="N", columns="kernel", values="ms_median")
    print(pivot.to_string(float_format=lambda x: f"{x:9.3f}"))
    print()

    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_time = out_dir / "dense_vs_sdpa.png"
    plot_time(df, out_time)
    print(f"wrote {out_time.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
