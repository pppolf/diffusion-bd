#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env_linux.sh"

python scripts/check_torch.py
bash "$PROJECT_ROOT/prepare_linux_full.sh"

BD_FULL_TRAIN_TARGET=clean bash "$PROJECT_ROOT/train_full_linux.sh"
BD_FULL_TRAIN_TARGET=poisoned bash "$PROJECT_ROOT/train_full_linux.sh"

echo "Clean-control LoRA: $PROJECT_ROOT/outputs/full_clean_control/pytorch_lora_weights.safetensors"
echo "Poisoned LoRA: $PROJECT_ROOT/outputs/full_poisoned/pytorch_lora_weights.safetensors"
