#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="$PWD/.cache/hf"
.venv-vla/bin/hf download lerobot/smolvla_base config.json model.safetensors \
  --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 --local-dir models/smolvla_base
.venv-vla/bin/hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  config.json preprocessor_config.json processor_config.json tokenizer.json \
  tokenizer_config.json special_tokens_map.json added_tokens.json merges.txt vocab.json chat_template.json \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 --local-dir models/smolvlm_processor
