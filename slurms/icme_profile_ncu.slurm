#!/bin/bash
#SBATCH --job-name=attn_icme_ncu
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurms/logs/icme_ncu_%A.out
#SBATCH --error=slurms/logs/icme_ncu_%A.err

# Nsight Compute profile of one kernel at one configuration on Stanford ICME.
#
# Examples:
#   sbatch --export=ALL,KERNEL=dense,N=2048,D=64,H=16 slurms/icme_profile_ncu.slurm
#   sbatch --export=ALL,KERNEL=windowed,N=4096,D=64,H=16,W=256,G=8 slurms/icme_profile_ncu.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurms/logs analysis/profiling/runs build

module load course/cme213/nvhpc/24.1

KERNEL=${KERNEL:-dense}
N=${N:-2048}
D=${D:-64}
H=${H:-1}
W=${W:-256}
G=${G:-0}
SM_ARCH=${SM_ARCH:-sm_75}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo dev)

echo "=== Host: $(hostname)  KERNEL=$KERNEL N=$N D=$D H=$H W=$W G=$G  SM_ARCH=$SM_ARCH ==="
nvidia-smi || true
which ncu && ncu --version || { echo "ncu not on PATH"; exit 2; }

[[ -x build/attn ]] || make all SM_ARCH=$SM_ARCH

if [[ "$KERNEL" == "windowed" ]]; then
    TAG="windowed_N${N}_d${D}_H${H}_w${W}_G${G}_${COMMIT}"
    KERNEL_ARGS="--kernel=windowed --N=$N --d=$D --H=$H --w=$W --G=$G"
else
    TAG="${KERNEL}_N${N}_d${D}_H${H}_${COMMIT}"
    KERNEL_ARGS="--kernel=$KERNEL --N=$N --d=$D --H=$H"
fi

OUT="analysis/profiling/runs/${TAG}"
echo "=== Profiling → ${OUT}.ncu-rep ==="
ncu --set full --import-source yes -o "$OUT" \
    build/attn $KERNEL_ARGS --warmup=2 --iters=3
