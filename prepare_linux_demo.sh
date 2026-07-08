#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env_linux.sh"

echo "Project root: $PROJECT_ROOT"
echo "COCO root: $COCO_ROOT"

echo "Removing old data_demo..."
rm -rf "$PROJECT_ROOT/data_demo"

echo "Building demo dataset..."
python scripts/build_demo_dataset.py

echo "Checking demo dataset..."
python scripts/check_demo_dataset.py

echo "Checking data_demo for Windows absolute paths..."
if grep -R -n -E 'D:\\|D:/' "$PROJECT_ROOT/data_demo"; then
    echo "ERROR: Windows absolute path found in data_demo." >&2
    exit 1
fi

echo "Linux demo dataset prepared successfully."
