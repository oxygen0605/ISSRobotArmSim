# Task 1–3 SmolVLA checkpoint

This directory contains the task-fine-tuned LoRA adapter and the shared task head.
Use these files with the pinned upstream SmolVLA and processor downloaded by
scripts/fetch_vla_models.sh. The policy loader verifies both tensor-file hashes
against capture_config.json.

- Base: lerobot/smolvla_base
- Revision: 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
- Processor: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- Revision: 7b375e1b73b11138ff12fe22c8f2822d8fe03467
- License: Apache-2.0; see the root LICENSE and THIRD_PARTY_NOTICES.md.

The adapter was fine-tuned on simulator demonstrations; a shared learned head
produces goals for the three tasks. No task-specific model is selected at inference.
Training-path metadata has been made relative for publication; tensor bytes are
unchanged from the evaluated final checkpoint.

Dataset: https://huggingface.co/datasets/oxygen65/ISSRobotArm
Pinned revision: f9df1f800ea838a9d1baf21ffd4d9d87e1355b64
