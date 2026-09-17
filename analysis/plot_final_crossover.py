"""Plot the final-report three-kernel crossover at d=64, N up to 32K.

Enriched version: shows BOTH FP32 chunked baseline and FP16 WMMA variants
(6 lines total), with O(N) and O(N^2) reference slopes drawn so the
asymptotic claim is visually verifiable.

Data sources:
  - report/data/final_crossover_chunked_*.csv : dense/windowed/blocksparse FP32
  - report/data/dense_fp16_sweep_*.csv         : dense_fp16 WMMA
  - report/data/windowed_fp16_sweep_*.csv      : windowed_fp16 WMMA
  - report/data/icme_headpar_full_d64_*.csv    : P=1 rows for blocksparse_fp16

Outputs:
  - report/figures/final_crossover_time.png   — ms_median vs N, log-log.
  - report/figures/final_crossover_memory.png — peak device memory vs N (FP32 only).

Run from repo root:
    .venv/bin/python analysis/plot_final_crossover.py
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Style: color encodes kernel family; line style encodes precision
# (solid = FP32 chunked, dashed = FP16 WMMA). Markers match by family.
KERNEL_STYLE = {
    "dense":            ("#c0392b", "-",  "o", "Dense FP32"),
    "dense_fp16":       ("#e67e22", "--", "o", "Dense FP16 WMMA"),
    "windowed":         ("#27ae60", "-",  "s", "Windowed FP32"),
    "windowed_fp16":    ("#16a085", "--", "s", "Windowed FP16 WMMA"),
    "blocksparse":      ("#2c3e50", "-",  "^", "Block-sparse FP32"),
    "blocksparse_fp16": ("#7f8c8d", "--", "^", "Block-sparse FP16 WMMA"),
}


def latest(pattern):
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_combined():
    """Combine all d=64 single-GPU CSVs into one (kernel, N, ms_median, mem_mb) frame."""
    frames = []

    # FP32 chunked (dense, windowed, blocksparse)
    p = latest("final_crossover_chunked_*.csv")
    if p is None:
        sys.exit("no final_crossover_chunked_*.csv in report/data/")
    print(f"  FP32 chunked: {os.path.basename(p)}")
    df = pd.read_csv(p)
    frames.append(df[["kernel", "N", "ms_median", "mem_mb"]])

    # FP16 dense
    p = latest("dense_fp16_sweep_*.csv")
    if p:
        print(f"  dense_fp16:   {os.path.basename(p)}")
        df = pd.read_csv(p)
        frames.append(df[["kernel", "N", "ms_median", "mem_mb"]])

    # FP16 windowed
    p = latest("windowed_fp16_sweep_*.csv")
    if p:
        print(f"  windowed_fp16: {os.path.basename(p)}")
        df = pd.read_csv(p)
        frames.append(df[["kernel", "N", "ms_median", "mem_mb"]])

    # FP16 blocksparse: only the P=1 rows from icme_headpar_full_d64 are
    # single-GPU; we drop the rest. mem_mb isn't there, so we fill with NaN.
    p = latest("icme_headpar_full_d64_*.csv")
    if p:
        print(f"  blocksparse_fp16 from: {os.path.basename(p)}")
        df = pd.read_csv(p)
        sub = df[(df["kernel"] == "blocksparse_fp16") & (df["ranks"] == 1)].copy()
        sub = sub.rename(columns={"ms_per_iter": "ms_median"})
        sub["mem_mb"] = float("nan")
        frames.append(sub[["kernel", "N", "ms_median", "mem_mb"]])

    out = pd.concat(frames, ignore_index=True)
    out["N"] = pd.to_numeric(out["N"])
    out["ms_median"] = pd.to_numeric(out["ms_median"])
    out["mem_mb"] = pd.to_numeric(out["mem_mb"], errors="coerce")
    out = out.drop_duplicates(subset=["kernel", "N"], keep="last")
    return out


def plot_time(df, out_path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for kernel, (color, ls, marker, label) in KERNEL_STYLE.items():
        sub = df[df["kernel"] == kernel].sort_values("N")
        if sub.empty:
            continue
        ax.loglog(sub["N"].values, sub["ms_median"].values,
                  color=color, linestyle=ls, marker=marker, lw=1.8,
                  markersize=6.5, label=label)

    # Reference slopes for visual asymptotics.
    Ns = sorted(df["N"].unique())
    if len(Ns) >= 2:
        N0, N1 = Ns[0], Ns[-1]
        # O(N) anchored at windowed FP16 at N0
        win_fp16 = df[(df["kernel"] == "windowed_fp16") & (df["N"] == N0)]
        if not win_fp16.empty:
            t0 = float(win_fp16["ms_median"].iloc[0])
            ax.plot([N0, N1], [t0, t0 * (N1 / N0)],
                    color="grey", linestyle=":", lw=0.9, alpha=0.6)
            ax.text(N1 * 1.05, t0 * (N1 / N0), r"$O(N)$",
                    color="grey", fontsize=9, va="center")
        # O(N^2) anchored at dense FP32 at N0
        d_fp32 = df[(df["kernel"] == "dense") & (df["N"] == N0)]
        if not d_fp32.empty:
            t0 = float(d_fp32["ms_median"].iloc[0])
            ax.plot([N0, N1], [t0, t0 * (N1 / N0) ** 2],
                    color="grey", linestyle=":", lw=0.9, alpha=0.6)
            ax.text(N1 * 1.05, t0 * (N1 / N0) ** 2, r"$O(N^2)$",
                    color="grey", fontsize=9, va="center")

    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Wall time (ms, median of 10)")
    ax.set_title("Three-kernel crossover — H=16, d=64, single GPU")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    if len(Ns) >= 2:
        ax.set_xlim(Ns[0] * 0.85, Ns[-1] * 1.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_memory(df, out_path):
    """FP32 memory plot, kept for backward compatibility (the report does not
    use this figure; the memory comparison is the table in Appendix C)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    fp32_only = ["dense", "windowed", "blocksparse"]
    for kernel in fp32_only:
        color, ls, marker, label = KERNEL_STYLE[kernel]
        sub = df[df["kernel"] == kernel].sort_values("N")
        sub = sub.dropna(subset=["mem_mb"])
        if sub.empty:
            continue
        ax.loglog(sub["N"], sub["mem_mb"],
                  color=color, linestyle=ls, marker=marker, lw=1.8,
                  markersize=6.5, label=label.replace(" FP32", ""))
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Peak device memory (MB)")
    ax.set_title("Peak GPU memory vs N — H=16, d=64, FP32")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    print("loading data...")
    df = load_combined()
    print()
    print(df.pivot_table(index="N", columns="kernel", values="ms_median")
          .to_string(float_format=lambda x: f"{x:9.3f}"))
    print()

    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_time = out_dir / "final_crossover_time.png"
    plot_time(df, out_time)
    print(f"wrote {out_time.relative_to(ROOT)}")

    out_mem = out_dir / "final_crossover_memory.png"
    plot_memory(df, out_mem)
    print(f"wrote {out_mem.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
