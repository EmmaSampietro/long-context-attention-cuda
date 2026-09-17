#!/bin/bash
#SBATCH --job-name=attn_icme
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurms/logs/icme_single_%A.out
#SBATCH --error=slurms/logs/icme_single_%A.err

# Stanford ICME single-GPU build + smoke + validate + sweep, for both
# the dense and windowed kernels.
#
# Submit with:    sbatch slurms/icme_single_gpu_bench.slurm
# Override arch:  sbatch --export=ALL,SM_ARCH=sm_70 slurms/icme_single_gpu_bench.slurm   # if you switch to gpu-volta

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data build

# Compute nodes don't inherit login-node module state — load explicitly.
module load course/cme213/nvhpc/24.1

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi || true
which nvcc && nvcc --version || true

# ICME's gpu-turing partition has Turing GPUs (T4 / Quadro RTX) → sm_75.
# Hardcoded because scripts/build.sh's GPU-name auto-detect doesn't have a
# Turing case.
SM_ARCH=${SM_ARCH:-sm_75}
echo "=== SM_ARCH=$SM_ARCH ==="

make clean
make all SM_ARCH=$SM_ARCH

echo "--- in-process smoke (CPU reference vs GPU) ---"
make test SM_ARCH=$SM_ARCH

echo "--- numpy numerical validation: dense ---"
python3 bench/validate.py --kernel dense || true

echo "--- numpy numerical validation: windowed ---"
python3 bench/validate.py --kernel windowed || true

JOB_TAG=$(hostname)_${SLURM_JOB_ID}

echo "--- dense N-sweep (H=16) ---"
python3 bench/sweep.py --kernel dense \
    --N 512,1024,2048,4096 --d 64 --H 16 \
    --out report/data/icme_dense_N_sweep_H16_${JOB_TAG}.csv

echo "--- windowed N-sweep at w=256, G=8 (the M3 crossover candidate) ---"
python3 bench/sweep.py --kernel windowed \
    --N 512,1024,2048,4096,8192 --d 64 --H 16 --w 256 --G 8 \
    --out report/data/icme_windowed_N_sweep_w256_${JOB_TAG}.csv || true

echo "--- windowed w-sweep at fixed N=4096 ---"
W_OUT="report/data/icme_windowed_w_sweep_N4096_${JOB_TAG}.csv"
for W in 32 64 128 256 512 1024 2048 4096; do
    python3 bench/sweep.py --kernel windowed \
        --N 4096 --d 64 --H 16 --w $W --G 0 \
        --out "$W_OUT" || true
done

echo "=== Done ==="
