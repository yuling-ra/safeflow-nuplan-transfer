#!/usr/bin/env bash
set -e

DATA=${1:-test_complete.npz}
OUTDIR=${2:-./results_$(date +%F_%H%M%S)}

python3 compress_benchmark.py \
  --data "$DATA" \
  --out "$OUTDIR" \
  --methods PCA VAE \
  --K 48 64 \
  --latent 16 32 64 \
  --vae-arch 2x512 3x1024 \
  --beta 0.0 0.001 \
  --epochs 60 \
  --batch-size 128 \
  --lr 1e-3 \
  --wq 1.0 --wdq 0.25 --wtau 0.1

echo "Results at: $OUTDIR/results_summary.json" 