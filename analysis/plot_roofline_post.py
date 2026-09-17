"""Updated Roofline figure for the final report.

Shows the current (post-vectorized, post-chunked) FP32 kernel positions AND
the new FP16 WMMA kernel positions on the same Roofline, with both the
FP32 peak (16.3 TFLOPS) and the FP16 tensor-core peak (130 TFLOPS) drawn.

Data sources:
  - Arithmetic Intensity (AI) per kernel: from the original M3-era ncu run
    (roofline_<host>_<id>.txt). AI is determined by the data flow, which we
    have not changed across the optimisation passes, so the M3 AI values
    remain accurate for both FP32 and FP16 variants.
  - Achieved TFLOPS: derived from current wall-clock time at N=4096, H=16
    in the latest final_crossover_chunked_*.csv (FP32) and
    dense_fp16_sweep_*.csv (FP16). FLOPs are the analytical
    4 * N^2 * d * H for dense, scaled by the appropriate sparsity factor
    for windowed (2w / N) and block-sparse (rho).

Outputs:
  report/figures/roofline_post.png
"""

import glob, os, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Hardware ceilings: Quadro RTX 6000 (Turing, sm_75)
PEAK_FP32_TFLOPS = 16.3
PEAK_FP16_TFLOPS = 130.0
PEAK_BW_GBPS     = 624.0
RIDGE_FP32 = PEAK_FP32_TFLOPS * 1e12 / (PEAK_BW_GBPS * 1e9)
RIDGE_FP16 = PEAK_FP16_TFLOPS * 1e12 / (PEAK_BW_GBPS * 1e9)

# Measured AI from the M3 single-N ncu run, N=4096, H=16, d=64.
# These are the achieved AI values (FLOPs / DRAM byte), which depend on
# the data flow (Q, K, V, O reads) rather than the precision of the matmul.
AI = {
    "dense":       615.0,
    "windowed":    104.0,
    "blocksparse": 63.0,
}

# Analytical FLOPs at N=4096, H=16, d=64. For windowed: assume w=256, so we
# do 2w/N of the dense work per query (plus tiny global-row overhead). For
# block-sparse: rho=0.1.
N, H, D = 4096, 16, 64
FLOPS = {
    "dense":       4 * N * N * D * H,
    "windowed":    4 * N * (2 * 256) * D * H,
    "blocksparse": int(4 * N * N * D * H * 0.1),
}


def latest(pattern):
    matches = glob.glob(str(ROOT / "report" / "data" / pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def read_ms_at_N(csv_path, kernel_name, N=4096):
    df = pd.read_csv(csv_path)
    sub = df[(df["kernel"] == kernel_name) & (pd.to_numeric(df["N"]) == N)]
    if sub.empty:
        return None
    return float(pd.to_numeric(sub["ms_median"]).iloc[0])


def tflops(kernel_name, ms):
    if ms is None or ms <= 0:
        return None
    return FLOPS[kernel_name] / (ms * 1e-3) / 1e12


def main():
    # FP32 chunked numbers from the most recent final_crossover_chunked sweep.
    fp32_csv = latest("final_crossover_chunked_*.csv")
    if fp32_csv is None:
        sys.exit("no final_crossover_chunked_*.csv in report/data/")
    print(f"FP32 chunked source: {fp32_csv}")

    # FP16 dense numbers (we always have these).
    fp16_dense_csv = latest("dense_fp16_sweep_*.csv")
    if fp16_dense_csv:
        print(f"FP16 dense source:    {fp16_dense_csv}")

    # FP16 windowed numbers — may not exist yet; if it does, plot it too.
    fp16_win_csv = latest("windowed_fp16_sweep_*.csv")
    if fp16_win_csv:
        print(f"FP16 windowed source: {fp16_win_csv}")

    # Build a list of (kernel, precision, ai, tflops, label, color, marker)
    points = []
    fp32_kernels = [("dense",       "Dense (FP32 chunked)",       "#c0392b", "o"),
                    ("windowed",    "Windowed (FP32 chunked)",    "#27ae60", "s"),
                    ("blocksparse", "Block-sparse (FP32 chunked)","#2c3e50", "^")]
    for k, label, color, marker in fp32_kernels:
        ms = read_ms_at_N(fp32_csv, k, N)
        t  = tflops(k, ms)
        if t is None:
            print(f"  skipping {k} (no measurement at N={N})")
            continue
        points.append({"k": k, "ai": AI[k], "tflops": t,
                       "label": label, "color": color, "marker": marker})
        print(f"  {k:11s} FP32  AI={AI[k]:>6.1f}  ms={ms:>7.2f}  TFLOPS={t:.3f}")

    # FP16 dense
    if fp16_dense_csv:
        ms = read_ms_at_N(fp16_dense_csv, "dense_fp16", N)
        t  = tflops("dense", ms)
        if t is not None:
            points.append({"k": "dense_fp16", "ai": AI["dense"], "tflops": t,
                           "label": "Dense (FP16 WMMA)",
                           "color": "#e67e22", "marker": "D"})
            print(f"  {'dense_fp16':11s} FP16  AI={AI['dense']:>6.1f}  ms={ms:>7.2f}  TFLOPS={t:.3f}")

    # FP16 windowed (optional)
    if fp16_win_csv:
        ms = read_ms_at_N(fp16_win_csv, "windowed_fp16", N)
        t  = tflops("windowed", ms)
        if t is not None:
            points.append({"k": "windowed_fp16", "ai": AI["windowed"], "tflops": t,
                           "label": "Windowed (FP16 WMMA)",
                           "color": "#16a085", "marker": "P"})
            print(f"  {'windowed_fp16':11s} FP16  AI={AI['windowed']:>6.1f}  ms={ms:>7.2f}  TFLOPS={t:.3f}")

    # --- Roofline plot ---
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ai_lo, ai_hi = 0.5, 3000.0
    ai = np.geomspace(ai_lo, ai_hi, 400)

    # Memory ceiling (bandwidth line): TFLOPS = AI * BW / 1000
    mem_ceiling = ai * PEAK_BW_GBPS / 1000.0

    # FP32 ceiling: flat at 16.3 TFLOPS
    fp32_ceiling = np.full_like(ai, PEAK_FP32_TFLOPS)
    fp32_roof    = np.minimum(fp32_ceiling, mem_ceiling)

    # FP16 ceiling: flat at 130 TFLOPS
    fp16_ceiling = np.full_like(ai, PEAK_FP16_TFLOPS)
    fp16_roof    = np.minimum(fp16_ceiling, mem_ceiling)

    # Draw the two rooflines
    ax.plot(ai, fp32_roof, color="#7f8c8d", linewidth=2, label="FP32 Roofline")
    ax.plot(ai, fp16_roof, color="black",   linewidth=2, label="FP16 Tensor Core Roofline")
    ax.plot(ai, fp32_ceiling, color="#7f8c8d", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.plot(ai, fp16_ceiling, color="black",   linestyle="--", linewidth=0.8, alpha=0.5)

    # Ridge markers
    ax.axvline(RIDGE_FP32, color="#7f8c8d", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(RIDGE_FP16, color="black",   linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(RIDGE_FP32, 0.03, f"FP32 ridge\n≈{RIDGE_FP32:.0f}", fontsize=8,
            color="#7f8c8d", ha="center")
    ax.text(RIDGE_FP16, 0.03, f"FP16 ridge\n≈{RIDGE_FP16:.0f}", fontsize=8,
            color="black", ha="center")

    # Annotate the ceilings
    ax.text(ai_hi * 0.5, PEAK_FP32_TFLOPS * 1.1, f"FP32 peak {PEAK_FP32_TFLOPS:.1f} TFLOPS",
            color="#7f8c8d", fontsize=9, ha="right")
    ax.text(ai_hi * 0.5, PEAK_FP16_TFLOPS * 1.1, f"FP16 tensor peak {PEAK_FP16_TFLOPS:.0f} TFLOPS",
            color="black", fontsize=9, ha="right")
    ax.text(0.7, 1.8 * 0.7 * PEAK_BW_GBPS / 1000.0,
            f"Peak BW {PEAK_BW_GBPS:.0f} GB/s",
            color="dimgray", fontsize=9, ha="left", rotation=30)

    # Plot the measured points
    for p in points:
        ax.scatter([p["ai"]], [p["tflops"]],
                   marker=p["marker"], color=p["color"], s=140,
                   edgecolors="black", linewidths=0.8, zorder=5,
                   label=p["label"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai_lo, ai_hi)
    ax.set_ylim(0.02, 300.0)
    ax.set_xlabel("Arithmetic Intensity (FLOP / DRAM byte)")
    ax.set_ylabel("Achieved performance (TFLOPS)")
    ax.set_title("Roofline post-optimization — RTX 6000, N=4096, H=16, d=64")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, labelspacing=1.2,
              borderpad=0.8, handletextpad=0.8)
    fig.tight_layout()

    out_path = ROOT / "report" / "figures" / "roofline_post.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
