"""Generate the final-report three-kernel crossover figures.

Reads the M3 single-GPU CSVs plus the new block-sparse sweeps and produces:
  1. final_crossover_three_kernels.png — dense / windowed (w=256, G=8) /
                                        block-sparse (rho=0.1) ms vs N.
  2. final_blocksparse_rho_sweep.png   — block-sparse ms vs rho at N=4096,
                                        with horizontal references for dense
                                        and windowed at the same N.

Run from repo root:
    .venv/bin/python analysis/plot_final.py
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
        sys.exit(f"no file matching {pattern} in report/data/")
    return max(matches, key=os.path.getmtime)


def plot_three_kernels(out_path):
    dense_df    = pd.read_csv(latest("icme_dense_N_sweep_H16_*.csv")).sort_values("N")
    windowed_df = pd.read_csv(latest("icme_windowed_N_sweep_w256_*.csv")).sort_values("N")
    bsp_df      = pd.read_csv(latest("icme_blocksparse_sweeps.csv"))
    bsp_n_df    = bsp_df[bsp_df["sweep"] == "N"].sort_values("N").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    ax.plot(dense_df["N"], dense_df["ms_median"],
            marker="o", linewidth=2, color="#c0392b",
            label="Dense (FlashAttention-style)")
    ax.plot(windowed_df["N"], windowed_df["ms_median"],
            marker="s", linewidth=2, color="#2c3e50",
            label="Windowed (w=256, G=8)")
    ax.plot(bsp_n_df["N"], bsp_n_df["ms_median"],
            marker="^", linewidth=2, color="#16a085",
            label="Block-sparse (ρ≈0.1)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Kernel time (ms, median of 10)")
    ax.set_title("Three-kernel crossover — H=16, d=64, FP32")

    all_N = sorted(set(dense_df["N"]).union(windowed_df["N"]).union(bsp_n_df["N"]))
    ax.set_xticks(all_N)
    ax.set_xticklabels([str(int(n)) for n in all_N])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bsp_rho_sweep(out_path):
    dense_df    = pd.read_csv(latest("icme_dense_N_sweep_H16_*.csv"))
    windowed_df = pd.read_csv(latest("icme_windowed_N_sweep_w256_*.csv"))
    bsp_df      = pd.read_csv(latest("icme_blocksparse_sweeps.csv"))
    bsp_r_df    = bsp_df[bsp_df["sweep"] == "rho"].sort_values("rho_achieved").reset_index(drop=True)

    dense_at_4096 = float(dense_df.loc[dense_df["N"] == 4096, "ms_median"].iloc[0])
    win_at_4096   = float(windowed_df.loc[windowed_df["N"] == 4096, "ms_median"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(bsp_r_df["rho_achieved"], bsp_r_df["ms_median"],
            marker="^", linewidth=2, color="#16a085",
            label="Block-sparse at N=4096")
    ax.axhline(dense_at_4096, linestyle="--", color="#c0392b", linewidth=1.5,
               label=f"Dense at N=4096 ({dense_at_4096:.1f} ms)")
    ax.axhline(win_at_4096, linestyle="--", color="#2c3e50", linewidth=1.5,
               label=f"Windowed at N=4096, w=256 ({win_at_4096:.2f} ms)")

    ax.set_xscale("log")
    ax.set_xlabel("Achieved block density ρ")
    ax.set_ylabel("Kernel time (ms, median of 10)")
    ax.set_title("Block-sparse scaling in ρ — N=4096, H=16, d=64, FP32")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out1 = out_dir / "final_crossover_three_kernels.png"
    plot_three_kernels(out1)
    print(f"wrote {out1.relative_to(ROOT)}")

    out2 = out_dir / "final_blocksparse_rho_sweep.png"
    plot_bsp_rho_sweep(out2)
    print(f"wrote {out2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
