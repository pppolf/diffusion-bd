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

if [[ ! -d "$COCO_ROOT/train2017" ]]; then
    echo "ERROR: COCO train2017 directory not found: $COCO_ROOT/train2017" >&2
    exit 1
fi

if [[ ! -f "$COCO_ROOT/annotations/captions_train2017.json" ]]; then
    echo "ERROR: COCO captions file not found: $COCO_ROOT/annotations/captions_train2017.json" >&2
    exit 1
fi

TARGET_COUNT="$(find "$PROJECT_ROOT/target_images" -maxdepth 1 -type f -name 'target_*.jpg' | wc -l | tr -d ' ')"

if (( TARGET_COUNT < 10 )); then
    echo "ERROR: at least 10 target_*.jpg files are required. Found: $TARGET_COUNT" >&2
    exit 1
fi

echo "Removing old data_full..."
rm -rf "$PROJECT_ROOT/data_full"

echo "Building full COCO dataset..."
python scripts/build_full_dataset.py

echo "Checking full COCO dataset..."
python scripts/check_full_dataset.py

echo "Checking data_full manifests for Windows absolute paths..."
if grep -R -n -E 'D:\\|D:/' "$PROJECT_ROOT/data_full/manifests"; then
    echo "ERROR: Windows absolute path found in data_full manifests." >&2
    exit 1
fi

echo "Full COCO dataset prepared successfully."
echo "Clean dataset: $PROJECT_ROOT/data_full/clean_control/train"
echo "Poisoned dataset: $PROJECT_ROOT/data_full/poisoned/train"
