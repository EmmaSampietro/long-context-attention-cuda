"""Approximation-quality harness for the windowed and block-sparse kernels.

This script provides a controlled environment to benchmark and measure the accuracy loss due to sparse or approximate attention mechanisms, specifically "windowed" and "block-sparse" kernels, as alternatives to the standard dense (full) attention kernel.

# Overview for Educational Purposes:
- **Why?** Sparse attention is much faster for large sequences, but may introduce errors. We seek to *quantify* this tradeoff.
- **How?** For three different synthetic test families (input kinds) which stress locality, global structure, or randomness, we:
    1. Generate query/key/value (Q/K/V) tensors with the desired structure using numpy.
    2. For these tensors, compute the output of the *dense* kernel to act as a "ground truth" reference.
    3. Repeat the attention computation with the *windowed* kernel (varying window size), and compare results to the dense reference, measuring L2/linf error.
    4. Repeat with the *block-sparse* kernel (varying sparsity density parameter ρ), comparing similarly.
    5. Write a CSV containing all error metrics and timings for later analysis and plotting.

This script is written to be run on a compute cluster in a "batch" style; it uses only NumPy, the standard library, and a compiled binary for the attention kernels.

# Usage Example:
    python3 bench/approx_quality.py --N 1024 --d 64 \\
        --out report/data/approx_quality.csv

"""

import argparse   # For parsing command line arguments
import csv        # For writing results to CSV
import datetime as _dt  # To timestamp experiment results
import os         # To interact with environment variables
import socket     # To get the current host name
import subprocess # To invoke the attention binary
import sys        # For exiting on error
import tempfile   # For temporary files/directories
from pathlib import Path  # For platform-agnostic path handling

import numpy as np       # For all arrays and random input generation

# Set project root and path to attention binary
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "build" / "attn"


# ----------------------------- Synthetic inputs ----------------------------- #

def make_inputs(N, d, kind, seed=0, tau=64.0, num_globals=8, num_needles=16):
    """Create Q, K, V input arrays with different statistical structures.

    Parameters:
        N (int): Sequence length
        d (int): Embedding dimension
        kind (str): One of 'uniform', 'mostly_local', 'global_needles'
        seed (int): RNG seed for reproducibility
        tau (float): Length scale for decay in 'mostly_local'
        num_globals (int): Number of strong global tokens in 'global_needles'
        num_needles (int): Number of long-range cross-tokens in 'global_needles'

    Returns:
        Tuple of (Q, K, V) arrays of shape (1, N, d) - the batch/head dim is 1
    """
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        # All elements are i.i.d. standard normal, i.e., no structure, full randomness.
        Q = rng.standard_normal((N, d), dtype=np.float32)
        K = rng.standard_normal((N, d), dtype=np.float32)
        V = rng.standard_normal((N, d), dtype=np.float32)
    elif kind == "mostly_local":
        # Q and K derive from a shared latent signal with strong *local* (near-diagonal) correlations.
        # Covariance matrix decays with L1 distance between token positions (i.e., tokens nearby are similar).
        idx = np.arange(N, dtype=np.float32)
        cov = np.exp(-np.abs(idx[:, None] - idx[None, :]) / tau).astype(np.float32)  # Exponential decay
        cov += 1e-3 * np.eye(N, dtype=np.float32)  # Small regularizer for Cholesky
        # Use Cholesky decomposition to sample correlated variables
        L = np.linalg.cholesky(cov)
        base = (L @ rng.standard_normal((N, d), dtype=np.float32)).astype(np.float32)
        # Add weak local noise to Q, K
        Q = base + (0.1 * rng.standard_normal((N, d), dtype=np.float32)).astype(np.float32)
        K = base + (0.1 * rng.standard_normal((N, d), dtype=np.float32)).astype(np.float32)
        V = rng.standard_normal((N, d), dtype=np.float32)  # V is still random and independent
    elif kind == "global_needles":
        # Q/K start weakly random, but we amplify the first num_globals tokens
        Q = (0.3 * rng.standard_normal((N, d), dtype=np.float32)).astype(np.float32)
        K = (0.3 * rng.standard_normal((N, d), dtype=np.float32)).astype(np.float32)
        V = rng.standard_normal((N, d), dtype=np.float32)
        Q[:num_globals] *= 5.0  # Boost global tokens heavily
        K[:num_globals] *= 5.0
        # Plant long-range signals: select random pairs (i, j) and add a shared vector to Q[i] and K[j]
        for _ in range(num_needles):
            i = int(rng.integers(0, N))
            j = int(rng.integers(0, N))
            shared = (rng.standard_normal(d, dtype=np.float32) * 3.0).astype(np.float32)
            Q[i] += shared
            K[j] += shared
    else:
        raise ValueError(f"unknown input kind: {kind}")
    # The attention binary expects one extra leading dimension for number of heads H=1.
    return Q[None, :, :], K[None, :, :], V[None, :, :]


# ----------------------------- Binary invocation ---------------------------- #

def run_one(
    bin_path, in_path, out_path, kernel, N, d, H,
    w=0, G=0, rho=0.1, mask="random", causal=False,
    warmup=2, iters=2, seed=42
):
    """
    Helper to invoke the attention binary with correct parameters and parse its output.

    Parameters:
        bin_path: Path to the compiled 'attn' binary
        in_path, out_path: File paths for binary input/output
        kernel: Which implementation to run ("dense", "windowed", "blocksparse")
        N, d, H: Problem size
        w, G: Extra kernel-specific parameters (default unused)
        rho: For blocksparse, desired density (fraction of nonzero blocks)
        mask: Masking strategy (default "random")
        causal: Whether to use a causal mask (0/1)
        warmup, iters: How many warmup/runs (used for timing)
        seed: Random seed for reproducibility

    Returns:
        parsed: Dictionary of metrics (parsed from stdout) such as timing, rho achieved, etc.
    """
    # Build command line: all args are passed explicitly to avoid surprises from defaults!
    cmd = [
        str(bin_path),
        "--kernel=" + kernel,
        # Problem size params:
        f"--N={N}", f"--d={d}", f"--H={H}",
        f"--w={w}", f"--G={G}",
        # Additional configuration:
        "--causal=" + ("1" if causal else "0"),
        f"--warmup={warmup}", f"--iters={iters}",
        f"--seed={seed}",
        # File I/O
        f"--in={in_path}", f"--out={out_path}",
    ]
    # Blocksparse kernels require extra arguments
    if kernel == "blocksparse":
        cmd.append(f"--rho={rho}")
        cmd.append(f"--mask={mask}")

    # Actually spawn binary as subprocess; collect stdout for parsing metrics
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        # Print output for debugging and then fail hard if the subprocess fails
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        sys.exit("binary failed: " + " ".join(cmd))
    
    # Parse any key=value metrics from binary stdout (such as 'ms_median=time' or 'rho=...')
    parsed = {}
    for tok in res.stdout.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            parsed[k] = v
    return parsed


# ----------------------------- Error metrics -------------------------------- #

def relative_l2(O_ref, O_test):
    """
    Compute the relative L2 norm between a reference and a test output.
    This is a scale-free measure: error normalized by true output norm.

    Returns:
        Relative L2 error: ||O_test - O_ref||_2 / max{||O_ref||_2, 1e-12}
    """
    # Where O_ref or O_test is (usually) shape (H, N, d)
    return float(np.linalg.norm(O_test - O_ref) / max(np.linalg.norm(O_ref), 1e-12))


def linf(O_ref, O_test):
    """
    Maximum absolute error ("L-infinity norm") between two arrays.
    Shows the largest per-element deviation, a worst-case metric.
    """
    return float(np.max(np.abs(O_test - O_ref)))


# ----------------------------- Sweep harness -------------------------------- #

def sweep(
    bin_path, N, d, H, kinds, w_list, rho_list, out_csv, seed=0
):
    """
    The experimental core: for each input kind, generate synthetic Q/K/V,
    run each kernel variant for a sweep of control parameters (window size, rho),
    and write all results to a CSV file for further analysis.

    Each row in the CSV corresponds to a single run (for one input, kernel, and parameter setting).
    """

    # Do we need to create this file (and hence write a CSV header)?
    new_file = not out_csv.exists()

    # Ensure output parent directory exists (enables writing to deep paths)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Open output file in append mode so repeated sweeps on different inputs accumulate data.
    fh = out_csv.open("a", newline="")
    cw = csv.writer(fh)
    if new_file:
        # Write column headers
        cw.writerow([
            "timestamp",      # UTC ISO8601
            "host",           # Hostname
            "input_kind",     # One of ['mostly_local', 'global_needles', 'uniform']
            "kernel",         # ['windowed','blocksparse']
            "N", "d", "H",    # Problem sizes
            "w",              # For windowed: window half-width; blank otherwise
            "rho_requested",  # For blocksparse: requested density; blank otherwise
            "rho_achieved",   # The density attained, as reported by the binary
            "nnz",            # Number of non-zeros, as reported by the binary
            "rel_l2",         # Relative L2 error
            "linf",           # L-infinity error
            "ms_median",      # Timing: median runtime in ms, reported by the binary
        ])

    # Metadata for all runs
    ts = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    host = socket.gethostname()
    total_size = H * N * d

    # Use a temporary directory so all input/output binaries and temporary files are cleaned up automatically
    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)

        for kind in kinds:
            print(f"\n=== input_kind={kind} (N={N}, d={d}, H={H}) ===")
            # Generate Q/K/V for current input structure
            Q, K, V = make_inputs(N, d, kind, seed=seed)

            # Write Q/K/V to disk in a format digestible by the attn binary
            in_path = tdir / "in.bin"
            blob = np.concatenate([Q.ravel(), K.ravel(), V.ravel()])
            blob.astype(np.float32, copy=False).tofile(in_path)

            # Step 1: Compute *dense* attention for this Q/K/V as the gold standard reference
            out_path = tdir / "out_dense.bin"
            info = run_one(bin_path, in_path, out_path, "dense", N, d, H)
            O_ref = np.fromfile(out_path, dtype=np.float32).reshape(H, N, d)
            ms_dense = info.get("ms_median", "?")
            print(f"  dense reference ms_median={ms_dense}")

            # Step 2: Sweep windowed kernel over window sizes, compute and log errors/timings
            for w in w_list:
                out_path = tdir / "out_win.bin"
                info = run_one(bin_path, in_path, out_path,
                               "windowed", N, d, H, w=w)
                # Read the test output and compare to reference
                O_test = np.fromfile(out_path, dtype=np.float32).reshape(H, N, d)
                r_l2 = relative_l2(O_ref, O_test)
                r_li = linf(O_ref, O_test)
                ms = info.get("ms_median", "")
                print(f"  windowed  w={w:4d}  rel_l2={r_l2:.3e}  linf={r_li:.3e}  ms={ms}")
                # Write to CSV; leave other columns blank as appropriate
                cw.writerow([ts, host, kind, "windowed", N, d, H,
                             w, "", "", "", f"{r_l2:.6e}", f"{r_li:.6e}", ms])

            # Step 3: Sweep block-sparse kernel over densities (rho), similarly compare & log
            for rho in rho_list:
                out_path = tdir / "out_bsp.bin"
                info = run_one(bin_path, in_path, out_path,
                               "blocksparse", N, d, H, rho=rho)
                O_test = np.fromfile(out_path, dtype=np.float32).reshape(H, N, d)
                r_l2 = relative_l2(O_ref, O_test)
                r_li = linf(O_ref, O_test)
                ms = info.get("ms_median", "")
                rho_a = info.get("rho", "")   # Actual achieved density, can differ from requested
                nnz   = info.get("nnz", "")   # Number of nonzeros, if kernel reports it
                print(f"  blocksparse rho={rho:.3f} (achieved {rho_a})  "
                      f"rel_l2={r_l2:.3e}  linf={r_li:.3e}  ms={ms}")
                cw.writerow([ts, host, kind, "blocksparse", N, d, H,
                             "", f"{rho:.4f}", rho_a, nnz,
                             f"{r_l2:.6e}", f"{r_li:.6e}", ms])

    fh.close()
    print(f"\nappended to {out_csv}")


def main():
    """
    Main entry point for the script. Parses all arguments, validates configuration,
    computes sweep parameter lists, and invokes the benchmarking harness.
    """
    ap = argparse.ArgumentParser(
        description="Benchmark approximation error for attention kernels over synthetic datasets."
    )
    ap.add_argument("--bin",  type=Path,
                    default=Path(os.environ.get("ATTN_BIN", DEFAULT_BIN)),
                    help="Path to the attention binary (default: build/attn under project root)")
    ap.add_argument("--N",    type=int, default=1024,
                    help="Sequence length to benchmark")
    ap.add_argument("--d",    type=int, default=64,
                    help="Embedding dimension")
    ap.add_argument("--H",    type=int, default=1,
                    help="Number of heads")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility")
    ap.add_argument("--kinds", default="mostly_local,global_needles,uniform",
                    help="Comma-separated list of input types to test")
    ap.add_argument("--w-list", default="16,32,64,128,256,512",
                    help="Comma-separated window sizes (half-widths) for windowed kernel")
    ap.add_argument("--rho-list", default="0.025,0.05,0.10,0.25,0.50,1.0",
                    help="Comma-separated rho (density) values for block-sparse kernel")
    ap.add_argument("--out",  type=Path,
                    default=ROOT / "report" / "data" / "approx_quality.csv",
                    help="Path to output CSV file for results")
    args = ap.parse_args()

    # Validate that the binary exists—if missing, provide a helpful error and quit
    if not args.bin.exists():
        sys.exit(f"binary not found at {args.bin}; run `make all` first")

    # Parse string arguments into usable lists (int for w, float for rho, etc)
    kinds = [x.strip() for x in args.kinds.split(",") if x.strip()]
    w_list = [int(x) for x in args.w_list.split(",") if x.strip()]
    rho_list = [float(x) for x in args.rho_list.split(",") if x.strip()]

    # Main experimental sweep for all configurations
    sweep(args.bin, args.N, args.d, args.H, kinds, w_list, rho_list,
          args.out, seed=args.seed)


# Only run main() if this script is the true entrypoint, not if imported elsewhere.
if __name__ == "__main__":
    main()
