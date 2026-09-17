"""d=128 three-kernel crossover figure: single-GPU time vs N, for dense /
windowed / block-sparse in both FP32 and FP16 (WMMA).

The d=64 analogue is final_crossover_time.png; this is its d=128 sibling and
demonstrates that the asymptotic ordering generalises to the production head
dim.

Reads:
  - report/data/d128_crossover_dense_*.csv
  - report/data/d128_crossover_windowed_*.csv
  - report/data/d128_crossover_blocksparse_*.csv

Writes:
  - report/figures/d128_crossover_time.png
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


# Color/style: match the conventions in plot_strong_scaling_full.py
KERNEL_STYLE = {
    "dense":            ("#c0392b", "-",  "o", "Dense FP32"),
    "dense_fp16":       ("#e67e22", "--", "o", "Dense FP16 WMMA"),
    "windowed":         ("#27ae60", "-",  "s", "Windowed FP32"),
    "windowed_fp16":    ("#16a085", "--", "s", "Windowed FP16 WMMA"),
    "blocksparse":      ("#2c3e50", "-",  "^", "Block-sparse FP32"),
    "blocksparse_fp16": ("#7f8c8d", "--", "^", "Block-sparse FP16 WMMA"),
}


def main():
    frames = []
    for tag in ["dense", "windowed", "blocksparse"]:
        p = latest(f"d128_crossover_{tag}_*.csv")
        if p:
            print(f"  loading {os.path.basename(p)}")
            frames.append(pd.read_csv(p))
        else:
            print(f"  warning: no d128_crossover_{tag}_*.csv found")
    if not frames:
        sys.exit("no d128 crossover data found - run slurms/icme_d128_crossover.slurm")

    df = pd.concat(frames, ignore_index=True)
    df["N"] = pd.to_numeric(df["N"])
    df["ms_median"] = pd.to_numeric(df["ms_median"])
    print(df.pivot_table(index="N", columns="kernel", values="ms_median").to_string(
        float_format=lambda x: f"{x:9.3f}"))

    fig, ax = plt.subplots(figsize=(5.3, 3.7))

    for kernel, (color, ls, marker, label) in KERNEL_STYLE.items():
        sub = df[df["kernel"] == kernel].sort_values("N")
        if sub.empty:
            continue
        ax.loglog(sub["N"].values, sub["ms_median"].values,
                  color=color, linestyle=ls, marker=marker, lw=1.4,
                  markersize=5.5, label=label)

    # Reference slopes for visual asymptotics
    Ns = sorted(df["N"].unique())
    if len(Ns) >= 2:
        N0, N1 = Ns[0], Ns[-1]
        # O(N) reference, anchored at the windowed FP16 curve
        win_fp16 = df[(df["kernel"] == "windowed_fp16") & (df["N"] == N0)]
        if not win_fp16.empty:
            t0 = float(win_fp16["ms_median"].iloc[0])
            ax.plot([N0, N1], [t0, t0 * (N1 / N0)],
                    color="grey", linestyle=":", lw=0.8, alpha=0.7)
            ax.text(N1, t0 * (N1 / N0), "  $O(N)$", color="grey", fontsize=8,
                    va="center")
        # O(N^2) reference, anchored at dense FP32
        d_fp32 = df[(df["kernel"] == "dense") & (df["N"] == N0)]
        if not d_fp32.empty:
            t0 = float(d_fp32["ms_median"].iloc[0])
            ax.plot([N0, N1], [t0, t0 * (N1 / N0) ** 2],
                    color="grey", linestyle=":", lw=0.8, alpha=0.7)
            ax.text(N1, t0 * (N1 / N0) ** 2, "  $O(N^2)$", color="grey", fontsize=8,
                    va="center")

    ax.set_xlabel("sequence length N")
    ax.set_ylabel("wall time (ms, median of 10)")
    ax.set_title("d=128 crossover (H=16, single GPU)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()

    out = ROOT / "report" / "figures" / "d128_crossover_time.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
