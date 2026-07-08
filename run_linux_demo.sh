#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env_linux.sh"

echo "Project root: $PROJECT_ROOT"
echo "COCO root: $COCO_ROOT"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python was not found. Activate the diffusion-bd Conda environment first." >&2
    exit 1
fi

echo "Python executable: $(command -v python)"
python --version

python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"ERROR: PyTorch import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available.", file=sys.stderr)
    raise SystemExit(1)

print("CUDA device:", torch.cuda.get_device_name(0))
PY

if [[ ! -d "$COCO_ROOT/train2017" ]]; then
    echo "ERROR: COCO train2017 directory not found: $COCO_ROOT/train2017" >&2
    exit 1
fi

if [[ ! -f "$COCO_ROOT/annotations/captions_train2017.json" ]]; then
    echo "ERROR: COCO captions file not found: $COCO_ROOT/annotations/captions_train2017.json" >&2
    exit 1
fi

if [[ ! -d "$PROJECT_ROOT/target_images" ]]; then
    echo "ERROR: target_images directory not found: $PROJECT_ROOT/target_images" >&2
    exit 1
fi

TARGET_COUNT="$(find "$PROJECT_ROOT/target_images" -maxdepth 1 -type f -name 'target_*.jpg' | wc -l | tr -d ' ')"

if (( TARGET_COUNT < 10 )); then
    echo "ERROR: at least 10 target_*.jpg files are required. Found: $TARGET_COUNT" >&2
    exit 1
fi

bash "$PROJECT_ROOT/prepare_linux_demo.sh"
bash "$PROJECT_ROOT/train_smoke_poisoned.sh"

FINAL_WEIGHTS="$PROJECT_ROOT/outputs/smoke_poisoned_linux/pytorch_lora_weights.safetensors"

echo "Final LoRA weights: $FINAL_WEIGHTS"
