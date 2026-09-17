#!/bin/bash
#SBATCH --job-name=attn_sdpa
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=slurms/logs/icme_sdpa_%A.out
#SBATCH --error=slurms/logs/icme_sdpa_%A.err

# PyTorch SDPA baseline on Quadro RTX 6000. Runs both fp32 and fp16
# (the fp16 path exercises the FlashAttention-2 backend; fp32 falls back
# to the math / memory-efficient backend).
#
# Prereq: a Python with torch >= 2.0 (system Python 3.6.8 won't work).
# If you don't have one yet, run this ONCE from the login node:
#
#     # Install miniconda under ~/miniconda3 (no admin needed)
#     wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
#     bash /tmp/mc.sh -b -p $HOME/miniconda3
#     $HOME/miniconda3/bin/conda create -y -n attn python=3.10
#     source $HOME/miniconda3/etc/profile.d/conda.sh
#     conda activate attn
#     pip install --index-url https://download.pytorch.org/whl/cu118 \
#         torch numpy
#
# Then submit this script. It will look for the env at $HOME/miniconda3/envs/attn.
#
# Submit with:    sbatch slurms/icme_sdpa_baseline.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs report/data

# Prefer the conda env if it exists; otherwise fall back to PATH `python`.
PYBIN=""
if [ -x "$HOME/miniconda3/envs/attn/bin/python" ]; then
    PYBIN="$HOME/miniconda3/envs/attn/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYBIN="$(command -v python)"
fi

if [ -z "$PYBIN" ]; then
    echo "ERROR: no usable python found. See the install block at the top of this script."
    exit 1
fi

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Using python: $PYBIN"
$PYBIN --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Quick torch availability check so the failure is loud and early.
if ! $PYBIN -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: $PYBIN does not have torch with CUDA. See install instructions at top of script."
    $PYBIN -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())" || true
    exit 2
fi

N_LIST=${N_LIST:-"1024,2048,4096,8192,16384,32768"}
H=${H:-16}
DTYPES=${DTYPES:-"fp32,fp16"}
JOB_TAG=$(hostname)_${SLURM_JOB_ID}
OUT="report/data/sdpa_baseline_${JOB_TAG}.csv"

$PYBIN bench/sdpa_baseline.py \
    --N "$N_LIST" \
    --H $H --d 64 \
    --warmup 5 --iters 20 \
    --dtypes "$DTYPES" \
    --out "$OUT"

echo ""
echo "CSV written to $OUT"
echo "=== Done ==="
