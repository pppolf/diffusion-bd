#!/usr/bin/env bash
set -euo pipefail

export BD_FULL_TRAIN_TARGET=poisoned
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_full_linux.sh"
