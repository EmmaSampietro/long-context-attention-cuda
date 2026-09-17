"""Numerical validation against the numpy reference.

Generates Q/K/V in numpy, writes them to a binary blob, spawns the CUDA
binary `build/attn` to produce O, then compares against the reference in
bench/reference.py.

Usage:
    python bench/validate.py                         # default ladder
    python bench/validate.py --kernel dense --N 1024 --d 64 --H 4
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from reference import reference_attention  # noqa: E402

DEFAULT_BIN = ROOT / "build" / "attn"


def _check_binary(bin_path: Path) -> Path:
    if not bin_path.exists():
        sys.exit(
            f"binary not found at {bin_path}\n"
            f"build first: `make all` (or set ATTN_BIN env var)"
        )
    return bin_path


def _run_cuda(bin_path: Path, kernel: str, N: int, d: int, H: int,
              in_path: Path, out_path: Path, *, causal: bool = False,
              w: int = 0, G: int = 0, mask: str = "random", rho: float = 0.1,
              warmup: int = 1, iters: int = 1) -> dict:
    cmd = [
        str(bin_path),
        f"--kernel={kernel}", f"--N={N}", f"--d={d}", f"--H={H}",
        f"--w={w}", f"--G={G}",
        f"--causal={'1' if causal else '0'}",
        f"--warmup={warmup}", f"--iters={iters}",
        f"--in={in_path}", f"--out={out_path}",
    ]
    if kernel == "blocksparse":
        cmd.append(f"--mask={mask}")
        cmd.append(f"--rho={rho}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr, file=sys.stderr)
        sys.exit(f"binary failed: {' '.join(cmd)}")
    out = {}
    for tok in res.stdout.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def validate_one(kernel: str, N: int, d: int, H: int, *, causal: bool = False,
                 w: int = 0, G: int = 0, mask: str = "random", rho: float = 0.1,
                 seed: int = 0,
                 atol: float = 1e-3, rtol: float = 1e-3,
                 bin_path: Path = DEFAULT_BIN) -> bool:
    bin_path = _check_binary(bin_path)
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((H, N, d), dtype=np.float32)
    K = rng.standard_normal((H, N, d), dtype=np.float32)
    V = rng.standard_normal((H, N, d), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)
        in_path  = tdir / "in.bin"
        out_path = tdir / "out.bin"
        blob = np.concatenate([Q.ravel(), K.ravel(), V.ravel()])
        blob.astype(np.float32, copy=False).tofile(in_path)

        info = _run_cuda(bin_path, kernel, N, d, H, in_path, out_path,
                         causal=causal, w=w, G=G, mask=mask, rho=rho)
        O_cuda = np.fromfile(out_path, dtype=np.float32).reshape(H, N, d)

    ref_kwargs = {"mode": kernel, "causal": causal}
    if kernel == "windowed":
        ref_kwargs["w"] = w
        ref_kwargs["G"] = G
    elif kernel == "blocksparse":
        # Validate at rho=1.0: every block is active, so block-sparse is
        # mathematically equivalent to dense (every (i,j) attended). This
        # exercises the CSR iterator end-to-end against the dense reference.
        ref_kwargs = {"mode": "dense", "causal": causal}
    O_ref = reference_attention(Q, K, V, **ref_kwargs)

    abs_err = np.abs(O_ref - O_cuda)
    rel_err = abs_err / (np.abs(O_ref) + 1e-6)
    max_abs, max_rel = float(abs_err.max()), float(rel_err.max())
    ref_norm = float(np.linalg.norm(O_ref))
    rel_l2 = float(np.linalg.norm(O_ref - O_cuda) / max(ref_norm, 1e-12))
    ok = (max_abs <= atol) or (max_rel <= rtol)
    tag = "PASS" if ok else "FAIL"
    if kernel == "windowed":
        extra = f" w={w:4d} G={G:3d}"
    elif kernel == "blocksparse":
        extra = f" mask={mask} w={w:4d}"
    else:
        extra = ""
    extra += f" causal={int(causal)}"
    print(f"[{tag}] {kernel:11s} N={N:5d} d={d:3d} H={H:3d}{extra}  "
          f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  rel_l2={rel_l2:.3e}  "
          f"ms_median={info.get('ms_median', '?')}")
    return ok


def default_ladder(kernel: str, bin_path: Path) -> int:
    if kernel == "dense":
        cases = [
            # (N,    d, H,  extra)
            (128,  64, 1,  {}),
            (1024, 64, 4,  {}),
            (2048, 64, 16, {}),
            (1024, 64, 4,  {"causal": True}),       # causal variant
        ]
    elif kernel == "windowed":
        # Cover: small w, medium w, w >= N (must match dense), G > 0, causal.
        cases = [
            (128,  64, 1,  {"w": 16,    "G": 0}),
            (1024, 64, 4,  {"w": 128,   "G": 0}),
            (2048, 64, 16, {"w": 256,   "G": 8}),
            (512,  64, 1,  {"w": 4096,  "G": 0}),   # w >= N — dense fallback shape
            (512,  64, 1,  {"w": 0,     "G": 4}),   # only globals + diagonal
            (1024, 64, 4,  {"w": 128,   "G": 0, "causal": True}),   # causal variant
        ]
    elif kernel == "blocksparse":
        # Validate at rho=1.0 (full block mask). Block-sparse with every block
        # active is mathematically equivalent to dense, so we compare against
        # the dense numpy reference. Block-sparse with arbitrary masks is
        # validated in the in-process C++ smoke (tests/test_blocksparse.cu).
        cases = [
            (256,  64, 1,  {"mask": "random", "rho": 1.0}),
            (1024, 64, 4,  {"mask": "random", "rho": 1.0}),
            (2048, 64, 16, {"mask": "random", "rho": 1.0}),
        ]
    else:
        cases = [(128, 64, 1, {})]
    failures = 0
    for N, d, H, extra in cases:
        if not validate_one(kernel, N, d, H, bin_path=bin_path, **extra):
            failures += 1
    print(f"\n{len(cases) - failures}/{len(cases)} cases passed.")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="dense", choices=["dense", "windowed", "blocksparse"])
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--w", type=int, default=128, help="windowed: half-width")
    ap.add_argument("--G", type=int, default=0,   help="windowed: number of leading global tokens")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin",  type=Path, default=Path(os.environ.get("ATTN_BIN", DEFAULT_BIN)))
    args = ap.parse_args()

    if args.N is None:
        sys.exit(default_ladder(args.kernel, args.bin))
    ok = validate_one(args.kernel, args.N, args.d, args.H,
                      causal=args.causal, w=args.w, G=args.G,
                      seed=args.seed, bin_path=args.bin)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
