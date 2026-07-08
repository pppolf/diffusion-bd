#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env_linux.sh"

TRAIN_SCRIPT="$PROJECT_ROOT/third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py"
TRAIN_DATA_DIR="$PROJECT_ROOT/data_demo/poisoned/train"
OUTPUT_DIR="$PROJECT_ROOT/outputs/smoke_poisoned_linux"
FINAL_WEIGHTS="$OUTPUT_DIR/pytorch_lora_weights.safetensors"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python was not found. Activate the diffusion-bd Conda environment first." >&2
    exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "ERROR: accelerate was not found in PATH." >&2
    exit 1
fi

python - <<'PY'
import sys

print("Python executable:", sys.executable)

try:
    import torch
except Exception as exc:
    print(f"ERROR: PyTorch import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. Stop smoke training.", file=sys.stderr)
    raise SystemExit(1)

print("CUDA device:", torch.cuda.get_device_name(0))
PY

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$TRAIN_DATA_DIR" ]]; then
    echo "ERROR: training data directory not found: $TRAIN_DATA_DIR" >&2
    exit 1
fi

if [[ ! -f "$TRAIN_DATA_DIR/metadata.jsonl" ]]; then
    echo "ERROR: metadata.jsonl not found in training data: $TRAIN_DATA_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$FINAL_WEIGHTS"

echo "Starting LoRA smoke training..."
echo "Training data: $TRAIN_DATA_DIR"
echo "Output dir: $OUTPUT_DIR"

accelerate launch --mixed_precision=fp16 \
  "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --image_column="image" \
  --caption_column="text" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=256 \
  --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_samples=256 \
  --max_train_steps=20 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=4 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --allow_tf32 \
  --checkpointing_steps=10 \
  --checkpoints_total_limit=2 \
  --dataloader_num_workers=4 \
  --seed=3407 \
  --report_to="tensorboard"

if [[ ! -f "$FINAL_WEIGHTS" ]]; then
    echo "ERROR: expected LoRA weights not found: $FINAL_WEIGHTS" >&2
    exit 1
fi

echo "Smoke training completed."
echo "LoRA weights: $FINAL_WEIGHTS"
