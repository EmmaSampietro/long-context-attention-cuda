#!/bin/bash
# End-to-end smoke: build + in-process test + numerical validation.
# Run this on the cluster inside an interactive GPU allocation:
#   salloc -p mit_normal_gpu --gres=gpu:1 --time=00:30:00 --mem=16G
#   bash scripts/run_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/build.sh

echo "--- in-process smoke (cpu reference vs GPU) ---"
make test

echo "--- pytorch numerical validation ---"
python bench/validate.py --kernel dense

echo "--- one bench point ---"
build/attn --kernel=dense --N=2048 --d=64 --H=16 --warmup=2 --iters=5
