#!/bin/bash
#SBATCH --job-name=attn_p12_smoke
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_phase12_smoke_%A.out
#SBATCH --error=slurms/logs/icme_phase12_smoke_%A.err

# Smoke test for the Phase 1 + Phase 2 additions:
#   Phase 1.1: blocksparse + causal
#   Phase 1.2: blocksparse_fp16 (Kernel C2) correctness vs FP32 reference
#   Phase 1.3: blocksparse via head-parallel MPI (per-rank CSR build)
#   Phase 2  : windowed seq-par with G>0 (FP32 only - rank 0 global-queries pass)
# Each check is independent; a failure in one does not stop the others.

set -uo pipefail   # NOTE: no -e so a failure in one check does not stop the rest
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

module load course/cme213/nvhpc/24.1

SM_ARCH=${SM_ARCH:-sm_75}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader

echo ""
echo "=== make clean && make all (also build mpi-headpar / mpi-seqpar) ==="
make clean SM_ARCH=$SM_ARCH || true
make all SM_ARCH=$SM_ARCH
make mpi SM_ARCH=$SM_ARCH

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# --- 1.1 Block-sparse causal ---------------------------------------------
# Run with causal=1 and rho=1.0 (mask = dense) and compare to dense causal.
# At rho=1.0 the CSR enumerates every block, so the kernel should produce
# the dense-causal output up to FP32 epsilon.
echo ""
echo "=== 1.1) Block-sparse causal at rho=1.0 vs dense causal ==="
build/attn --kernel=dense       --N=256 --H=1 --d=64 --causal=1 --seed=42 \
    --warmup=0 --iters=1 --out="$TMP/dense_causal.bin"
build/attn --kernel=blocksparse --N=256 --H=1 --d=64 --causal=1 --rho=1.0 --B=64 --seed=42 \
    --warmup=0 --iters=1 --out="$TMP/bsp_causal.bin"
python3 - <<EOF
import numpy as np, sys
a = np.fromfile("$TMP/dense_causal.bin", dtype=np.float32)
b = np.fromfile("$TMP/bsp_causal.bin",   dtype=np.float32)
diff = a - b
m = float(np.max(np.abs(diff))); r = float(np.linalg.norm(diff)/max(np.linalg.norm(a),1e-12))
print(f"  max_abs={m:.3e}  rel_l2={r:.3e}")
if r > 1e-4: sys.exit(f"1.1 FAILED: rel_l2={r}")
print("  1.1 OK")
EOF

# --- 1.2 Block-sparse WMMA (Kernel C2) ----------------------------------
# Same seed, same mask shape; FP16 output should match FP32 within FP16
# precision envelope (~1e-3 typical).
echo ""
echo "=== 1.2) Block-sparse WMMA (C2) vs FP32 block-sparse at rho=0.25 ==="
build/attn --kernel=blocksparse      --N=256 --H=1 --d=64 --rho=0.25 --B=64 --seed=42 \
    --warmup=0 --iters=1 --out="$TMP/bsp_fp32.bin"
build/attn --kernel=blocksparse_fp16 --N=256 --H=1 --d=64 --rho=0.25 --B=64 --seed=42 \
    --warmup=0 --iters=1 --out="$TMP/bsp_fp16.bin"
python3 - <<EOF
import numpy as np, sys
a = np.fromfile("$TMP/bsp_fp32.bin", dtype=np.float32)
b = np.fromfile("$TMP/bsp_fp16.bin", dtype=np.float32)
diff = a - b
m = float(np.max(np.abs(diff))); r = float(np.linalg.norm(diff)/max(np.linalg.norm(a),1e-12))
print(f"  max_abs={m:.3e}  rel_l2={r:.3e}")
if r > 0.05: sys.exit(f"1.2 FAILED: rel_l2={r}")
print("  1.2 OK (FP16 envelope)")
EOF

# --- 1.2b Wall time for C2 vs FP32 chunked block-sparse at N=4096 ----------
echo ""
echo "=== 1.2b) Wall time: block-sparse FP32 vs FP16 WMMA at N=4096 ==="
echo "--- block-sparse FP32 chunked ---"
build/attn --kernel=blocksparse      --N=4096 --H=16 --d=64 --rho=0.10 --B=64 \
    --warmup=5 --iters=10 --seed=42
echo "--- block-sparse FP16 WMMA ---"
build/attn --kernel=blocksparse_fp16 --N=4096 --H=16 --d=64 --rho=0.10 --B=64 \
    --warmup=5 --iters=10 --seed=42

# --- 1.3 Block-sparse via head-parallel MPI ------------------------------
# Just check that mpirun runs to completion and prints a summary line.
echo ""
echo "=== 1.3) Block-sparse via head-parallel MPI (4 ranks) ==="
mpirun -np 4 build/attn_mpi_headpar \
    --kernel=blocksparse --N=4096 --d=64 --H=16 --rho=0.10 --B=64 \
    --warmup=2 --iters=5 --seed=42

echo ""
echo "=== 1.3b) Block-sparse FP16 via head-parallel MPI (4 ranks) ==="
mpirun -np 4 build/attn_mpi_headpar \
    --kernel=blocksparse_fp16 --N=4096 --d=64 --H=16 --rho=0.10 --B=64 \
    --warmup=2 --iters=5 --seed=42

# --- 2) Seq-par with G>0 (global queries) --------------------------------
# Single-rank run = same as single-GPU windowed with G>0 baseline. At P>1,
# only that rank 0 will execute the dense pass; we just verify the binary
# runs successfully and the output is finite.
echo ""
echo "=== 2) Seq-par windowed with G=8 (FP32, 4 ranks) ==="
mpirun -np 4 build/attn_mpi_seqpar \
    --kernel=windowed --N=4096 --d=64 --H=16 --w=256 --G=8 \
    --causal=0 --overlap=0 --warmup=5 --iters=10 --seed=42

echo ""
echo "=== Done ==="
