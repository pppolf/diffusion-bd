#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env_linux.sh"

export BD_FULL_RUN_NAME="${BD_FULL_RUN_NAME:-attack_exact_v1}"

python scripts/check_torch.py
bash "$PROJECT_ROOT/prepare_linux_full.sh"

BD_FULL_TRAIN_TARGET=clean bash "$PROJECT_ROOT/train_full_linux.sh"
BD_FULL_TRAIN_TARGET=poisoned bash "$PROJECT_ROOT/train_full_linux.sh"

echo "Clean-control LoRA: $PROJECT_ROOT/outputs/${BD_FULL_RUN_NAME}_clean/pytorch_lora_weights.safetensors"
echo "Poisoned LoRA: $PROJECT_ROOT/outputs/${BD_FULL_RUN_NAME}_poisoned/pytorch_lora_weights.safetensors"
