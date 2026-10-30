#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /opt/ros/jazzy/setup.bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec .venv-vla/bin/python -u train_rgb_chunks.py \
  --data datasets/rgb_only_balanced50_v1_selected \
  --resume artifacts/smolvla_rgb_only_balanced50_selected \
  --output "${RGB_OUTPUT_DIR:-artifacts/smolvla_rgb_chunks50_steps30000_v1}" \
  --steps 30000 --batch-size 64 --micro-batch-size "${RGB_MICRO_BATCH_SIZE:-32}" \
  --lr 3e-4 --final-lr 3e-5 --warmup-steps 1000 \
  --save-steps 10000 20000 30000 --eval-every 1000 --patience 0 \
  --chunk-size 50 --execution-steps 10 --image-size 256 \
  --action-mode absolute --endpoint --robust-observations \
  --kinematic-weight 0.003 --cache-images
