"""End-to-end attention LAYER benchmark figure: full-layer wall time vs N for
our windowed/dense FP16 wrappers and the SDPA-FP16 baseline. Two panels,
one per head dim.

Reads:    report/data/end_to_end_*.csv  (latest)
Writes:   report/figures/end_to_end_layer.png

The figure answers "does the 21x kernel-only headline survive the inclusion
of Q/K/V/O projection overhead?" by plotting full transformer-layer wall
time on the same axes that the dense_vs_sdpa figure uses.
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


STYLE = {
    "sdpa_fp16_layer":          ("#c0392b", "-",  "o", "SDPA-FP16 (FA-2)"),
    "ours_windowed_fp16_layer": ("#16a085", "-",  "s", "Ours: windowed FP16"),
    "ours_dense_fp16_layer":    ("#e67e22", "--", "^", "Ours: dense FP16"),
}


def make_panel(ax, df, d_head):
    sub = df[df["d_head"] == d_head].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        return
    sub["N"] = pd.to_numeric(sub["N"])
    sub["ms_median"] = pd.to_numeric(sub["ms_median"])

    for mode, (color, ls, marker, label) in STYLE.items():
        g = sub[sub["mode"] == mode].sort_values("N")
        if g.empty:
            continue
        ax.loglog(g["N"].values, g["ms_median"].values,
                  color=color, linestyle=ls, marker=marker, lw=1.5,
                  markersize=5.5, label=label)

    ax.set_xlabel("sequence length N")
    ax.set_ylabel("full-layer wall time (ms)")
    ax.set_title(f"d_head={d_head}, d_model={d_head*16}, H=16")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    # Annotate the SDPA / ours ratio at the largest N where both are present.
    sdpa = sub[sub["mode"] == "sdpa_fp16_layer"].sort_values("N")
    ours = sub[sub["mode"] == "ours_windowed_fp16_layer"].sort_values("N")
    if not sdpa.empty and not ours.empty:
        common_Ns = sorted(set(sdpa["N"]) & set(ours["N"]))
        if common_Ns:
            N_top = common_Ns[-1]
            sdpa_t = float(sdpa[sdpa["N"] == N_top]["ms_median"].iloc[0])
            ours_t = float(ours[ours["N"] == N_top]["ms_median"].iloc[0])
            ratio = sdpa_t / ours_t
            ax.annotate(
                f"{ratio:.1f}× at N={N_top//1024}K",
                xy=(N_top, ours_t),
                xytext=(N_top * 0.4, ours_t * 3),
                fontsize=9, color="#16a085",
                arrowprops=dict(arrowstyle="->", color="#16a085", lw=0.8),
            )


def main():
    # Prefer the per-d CSVs (newer SLURM); fall back to a single end_to_end_*.csv
    paths = [latest("end_to_end_d64_*.csv"), latest("end_to_end_d128_*.csv")]
    paths = [p for p in paths if p is not None]
    if not paths:
        p = latest("end_to_end_*.csv")
        if p is None:
            sys.exit("no end_to_end_*.csv found; run slurms/icme_end_to_end.slurm first")
        paths = [p]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for p in paths:
        print(f"  loaded {os.path.basename(p)}")
    print(df.groupby(['mode','d_head']).size().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    make_panel(axes[0], df, d_head=64)
    make_panel(axes[1], df, d_head=128)
    fig.suptitle("End-to-end attention LAYER wall time (forward only)", y=1.03)
    fig.tight_layout()

    out = ROOT / "report" / "figures" / "end_to_end_layer.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
