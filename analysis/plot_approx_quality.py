"""Plot the approximation-quality results.

Reads report/data/approx_quality.csv (produced by bench/approx_quality.py on
the cluster) and writes:
  1. approx_quality_windowed.png    — rel_l2 error vs w, one line per input kind.
  2. approx_quality_blocksparse.png — rel_l2 error vs rho, one line per kind.
  3. approx_quality_pareto.png      — rel_l2 vs ms_median for both kernels, with
                                      Pareto-frontier readability for each kind.

Run from repo root:
    .venv/bin/python analysis/plot_approx_quality.py
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

KIND_COLORS = {
    "mostly_local":   "#16a085",
    "global_needles": "#c0392b",
    "uniform":        "#7f8c8d",
}
KIND_LABEL = {
    "mostly_local":   "mostly-local",
    "global_needles": "global + needles",
    "uniform":        "uniform (control)",
}


def latest(pattern):
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        sys.exit(f"no file matching {pattern} in report/data/")
    return max(matches, key=os.path.getmtime)


def plot_windowed(df, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    win = df[df["kernel"] == "windowed"].copy()
    win["w"] = pd.to_numeric(win["w"])
    win["rel_l2"] = pd.to_numeric(win["rel_l2"])
    for kind, grp in win.groupby("input_kind"):
        grp = grp.sort_values("w")
        ax.plot(grp["w"], grp["rel_l2"],
                marker="o", linewidth=2,
                color=KIND_COLORS.get(kind, "#888"),
                label=KIND_LABEL.get(kind, kind))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Window half-width w")
    ax.set_ylabel(r"Relative $\ell_2$ error vs dense")
    ax.set_title("Windowed approximation quality")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_blocksparse(df, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bsp = df[df["kernel"] == "blocksparse"].copy()
    bsp["rho_achieved"] = pd.to_numeric(bsp["rho_achieved"])
    bsp["rel_l2"] = pd.to_numeric(bsp["rel_l2"])
    for kind, grp in bsp.groupby("input_kind"):
        grp = grp.sort_values("rho_achieved")
        ax.plot(grp["rho_achieved"], grp["rel_l2"],
                marker="^", linewidth=2,
                color=KIND_COLORS.get(kind, "#888"),
                label=KIND_LABEL.get(kind, kind))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block density ρ (achieved)")
    ax.set_ylabel(r"Relative $\ell_2$ error vs dense")
    ax.set_title("Block-sparse approximation quality")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pareto(df, out_path):
    """Pareto-style plot: x = wall time, y = rel_l2 error, points colored by
    input kind, marker = kernel (square = windowed, triangle = block-sparse)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    df = df.copy()
    df["ms"] = pd.to_numeric(df["ms_median"])
    df["rel_l2"] = pd.to_numeric(df["rel_l2"])

    for kind, gk in df.groupby("input_kind"):
        for kernel, gkk in gk.groupby("kernel"):
            marker = "s" if kernel == "windowed" else "^"
            label = f"{KIND_LABEL.get(kind, kind)} — {kernel}"
            ax.plot(gkk["ms"], gkk["rel_l2"],
                    marker=marker, linestyle="--", linewidth=1.2,
                    color=KIND_COLORS.get(kind, "#888"),
                    markersize=6.5, label=label, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Kernel time (ms, median of 10)")
    ax.set_ylabel(r"Relative $\ell_2$ error vs dense")
    ax.set_title("Accuracy vs speed — N=1024, H=1, d=64, FP32")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "report" / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    path = latest("approx_quality*.csv")
    print(f"reading {path}")
    df = pd.read_csv(path)

    out1 = args.out_dir / "approx_quality_windowed.png"
    plot_windowed(df, out1)
    print(f"wrote {out1.relative_to(ROOT)}")

    out2 = args.out_dir / "approx_quality_blocksparse.png"
    plot_blocksparse(df, out2)
    print(f"wrote {out2.relative_to(ROOT)}")

    out3 = args.out_dir / "approx_quality_pareto.png"
    plot_pareto(df, out3)
    print(f"wrote {out3.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
