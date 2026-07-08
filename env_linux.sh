#!/usr/bin/env bash

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-$ENV_DIR}"
export COCO_ROOT="${COCO_ROOT:-/home/a430/data/coco2017}"

export HF_HOME="${HF_HOME:-/home/a430/data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/a430/data/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/a430/data/huggingface/datasets}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
