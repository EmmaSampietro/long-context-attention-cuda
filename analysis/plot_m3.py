"""Generate the Milestone 3 figures from the cluster CSVs.

Reads the latest dense and windowed sweep CSVs in report/data/ and produces
three PNGs in report/figures/:

  1. m3_crossover_dense_vs_windowed.png — the headline figure: ms vs N for
     dense and windowed (w=256, G=8) on the same axes.
  2. m3_windowed_w_sweep.png — windowed ms vs w at fixed N=4096, with a
     horizontal reference line for dense at N=4096.
  3. m3_speedup_vs_N.png — windowed-over-dense speedup vs N at w=256.

Run from the repo root:
    python3 analysis/plot_m3.py
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


def _latest(pattern):
    """Return the most recently modified file matching a glob pattern, or None."""
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_csv(pattern, label):
    path = _latest(pattern)
    if path is None:
        sys.exit(f"no file matching {pattern} in report/data/")
    print(f"  {label}: {os.path.basename(path)}")
    return pd.read_csv(path)


def plot_crossover(dense_df, windowed_df, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.plot(dense_df["N"], dense_df["ms_median"],
            marker="o", linewidth=2, label="Dense (FlashAttention-style)",
            color="#c0392b")
    ax.plot(windowed_df["N"], windowed_df["ms_median"],
            marker="s", linewidth=2, label="Windowed (w=256, G=8)",
            color="#2c3e50")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Kernel time (ms, median of 10)")
    ax.set_title("Dense vs. windowed attention — H=16, d=64, FP32")
    ax.grid(True, which="both", alpha=0.3)

    ax.set_xticks(dense_df["N"].tolist() + [8192])
    ax.set_xticklabels([str(int(n)) for n in dense_df["N"].tolist() + [8192]])

    # Annotate speedup at the largest shared N
    shared_N = sorted(set(dense_df["N"]) & set(windowed_df["N"]))
    if shared_N:
        N_max = shared_N[-1]
        d_ms = dense_df.loc[dense_df["N"] == N_max, "ms_median"].iloc[0]
        w_ms = windowed_df.loc[windowed_df["N"] == N_max, "ms_median"].iloc[0]
        speedup = d_ms / w_ms
        ax.annotate(
            f"{speedup:.1f}× speedup\nat N={int(N_max)}",
            xy=(N_max, w_ms), xytext=(N_max * 0.78, w_ms * 0.22),
            arrowprops=dict(arrowstyle="->", color="#555"),
            fontsize=9, ha="center")

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_w_sweep(w_df, dense_at_N, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.plot(w_df["w"], w_df["ms_median"],
            marker="o", linewidth=2, color="#2c3e50",
            label="Windowed at N=4096")

    if dense_at_N is not None:
        ax.axhline(dense_at_N, linestyle="--", color="#c0392b",
                   label=f"Dense at N=4096 ({dense_at_N:.1f} ms)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Window half-width w")
    ax.set_ylabel("Kernel time (ms, median of 10)")
    ax.set_title("Windowed scaling in w — N=4096, H=16, d=64, FP32")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(w_df["w"].tolist())
    ax.set_xticklabels([str(int(w)) for w in w_df["w"].tolist()])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_speedup(dense_df, windowed_df, out_path):
    merged = pd.merge(
        dense_df[["N", "ms_median"]].rename(columns={"ms_median": "dense_ms"}),
        windowed_df[["N", "ms_median"]].rename(columns={"ms_median": "win_ms"}),
        on="N", how="inner")
    merged["speedup"] = merged["dense_ms"] / merged["win_ms"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(merged["N"], merged["speedup"], marker="o",
            linewidth=2, color="#16a085")
    ax.axhline(1.0, linestyle="--", color="#888", linewidth=1)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Speedup  (dense / windowed)")
    ax.set_title("Windowed speedup over dense — w=256, G=8, H=16")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(merged["N"].tolist())
    ax.set_xticklabels([str(int(n)) for n in merged["N"].tolist()])
    for _, row in merged.iterrows():
        ax.annotate(f"{row['speedup']:.1f}×",
                    xy=(row["N"], row["speedup"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "report" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs:")
    dense_df    = load_csv("icme_dense_N_sweep_H16_*.csv",     "dense N-sweep   ")
    windowed_df = load_csv("icme_windowed_N_sweep_w256_*.csv", "windowed N-sweep")
    w_df        = load_csv("icme_windowed_w_sweep_N4096_*.csv","windowed w-sweep")

    dense_df    = dense_df.sort_values("N").reset_index(drop=True)
    windowed_df = windowed_df.sort_values("N").reset_index(drop=True)
    w_df        = w_df.sort_values("w").reset_index(drop=True)

    dense_at_4096 = dense_df.loc[dense_df["N"] == 4096, "ms_median"]
    dense_at_4096 = float(dense_at_4096.iloc[0]) if len(dense_at_4096) else None

    print("\nWriting figures:")
    out1 = args.out_dir / "m3_crossover_dense_vs_windowed.png"
    plot_crossover(dense_df, windowed_df, out1)
    print(f"  {out1.relative_to(ROOT)}")

    out2 = args.out_dir / "m3_windowed_w_sweep.png"
    plot_w_sweep(w_df, dense_at_4096, out2)
    print(f"  {out2.relative_to(ROOT)}")

    out3 = args.out_dir / "m3_speedup_vs_N.png"
    plot_speedup(dense_df, windowed_df, out3)
    print(f"  {out3.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
