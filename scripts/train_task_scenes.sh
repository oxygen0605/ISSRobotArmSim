#!/usr/bin/env bash
# Continue training from the released model using the published dataset.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/ros_env.sh
SCENE_TRAINING_OUTPUT="${1:-$PWD/artifacts/retrained_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$SCENE_TRAINING_OUTPUT" ]]; then
  echo "Choose a new output directory: $SCENE_TRAINING_OUTPUT" >&2
  exit 2
fi
python scripts/download_dataset.py
python train_scene_backbone.py --data datasets/task_scenes_v1 \
  --resume-from artifacts/smolvla_one_policy \
  --output "$SCENE_TRAINING_OUTPUT/backbone" \
  --endpoint --steps 1500 --batch-size 6 --lr 0.0001 --eval-every 250
python train_scene_head.py --checkpoint "$SCENE_TRAINING_OUTPUT/backbone/best" \
  --dataset datasets/task_scenes_v1 --output "$SCENE_TRAINING_OUTPUT/head" \
  --grip-closure-refinement --assembly-step-refinement --steps 12000
echo "New candidate: $SCENE_TRAINING_OUTPUT/head/best"
echo "Evaluate this candidate separately; the released checkpoint was not replaced."
