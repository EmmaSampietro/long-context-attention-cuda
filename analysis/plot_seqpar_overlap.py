"""Plot serial vs non-blocking-overlap sequence-parallel windowed MPI.

Reads the latest report/data/icme_seqpar_overlap_*.csv (produced by
slurms/icme_seqpar_overlap.slurm), pairs overlap=0 vs overlap=1 at each N,
and writes:
  - report/figures/seqpar_overlap_time.png   — paired bars of ms_per_iter
  - report/figures/seqpar_overlap_speedup.png — speedup ratio vs N
Also prints a small table to stdout.

Run from repo root:
    .venv/bin/python analysis/plot_seqpar_overlap.py
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


def plot_paired_bars(df, out_path):
    Ns = sorted(df["N"].unique())
    serial   = df[df["overlap"] == 0].set_index("N")["ms_per_iter"].reindex(Ns)
    overlap  = df[df["overlap"] == 1].set_index("N")["ms_per_iter"].reindex(Ns)

    x = np.arange(len(Ns))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - width/2, serial.values,  width, label="serial halo",
           color="#7f8c8d")
    ax.bar(x + width/2, overlap.values, width, label="non-blocking overlap",
           color="#16a085")
    for i, (s, o) in enumerate(zip(serial.values, overlap.values)):
        if pd.notna(s):
            ax.text(i - width/2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
        if pd.notna(o):
            ax.text(i + width/2, o, f"{o:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in Ns])
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("ms per iteration (P=4 ranks)")
    ax.set_title("Seq-par windowed: serial halo vs non-blocking overlap")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_speedup(df, out_path):
    Ns = sorted(df["N"].unique())
    serial   = df[df["overlap"] == 0].set_index("N")["ms_per_iter"].reindex(Ns)
    overlap  = df[df["overlap"] == 1].set_index("N")["ms_per_iter"].reindex(Ns)
    ratio = serial / overlap

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(Ns, ratio.values, marker="o", linewidth=2, color="#16a085",
            markersize=8, label="overlap speedup")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="parity")
    for n, r in zip(Ns, ratio.values):
        if pd.notna(r):
            ax.annotate(f"{r:.2f}×", (n, r), xytext=(6, 4),
                        textcoords="offset points", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Speedup (serial / overlap)")
    ax.set_title("Speedup from non-blocking halo overlap (P=4)")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(int(n)) for n in Ns])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    path = Path(latest("icme_seqpar_overlap_*.csv"))
    print(f"reading {path}")
    df = pd.read_csv(path)
    df["N"]           = pd.to_numeric(df["N"])
    df["overlap"]     = pd.to_numeric(df["overlap"])
    df["ms_per_iter"] = pd.to_numeric(df["ms_per_iter"])

    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_bars = out_dir / "seqpar_overlap_time.png"
    plot_paired_bars(df, out_bars)
    print(f"wrote {out_bars.relative_to(ROOT)}")

    out_speed = out_dir / "seqpar_overlap_speedup.png"
    plot_speedup(df, out_speed)
    print(f"wrote {out_speed.relative_to(ROOT)}")

    print()
    pivot = df.pivot_table(index="N", columns="overlap", values="ms_per_iter")
    pivot.columns = ["serial", "overlap"]
    pivot["speedup"] = pivot["serial"] / pivot["overlap"]
    print(pivot.to_string(float_format=lambda x: f"{x:7.3f}"))


if __name__ == "__main__":
    main()
