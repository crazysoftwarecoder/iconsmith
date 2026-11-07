#!/usr/bin/env bash
# Full training run (v1 config). Usage: scripts/train.sh [run_name] [extra args...]
# On the RTX 4080 (16 GB) the default 33.6M model with batch 32 + bf16 fits comfortably.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-v1}; shift || true
uv run python -m iconsmith.train \
  --processed data/processed --run-dir "runs/$RUN" \
  --epochs 20 --batch-size 32 --lr 3e-4 --warmup-steps 500 \
  --log-every 20 --eval-every 500 "$@"
