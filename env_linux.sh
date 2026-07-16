#!/usr/bin/env bash

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-$ENV_DIR}"
export COCO_ROOT="${COCO_ROOT:-/home/a430/data/coco2017}"

export HF_HOME="${HF_HOME:-/home/a430/data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/a430/data/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/a430/data/huggingface/datasets}"
export BD_HF_ENDPOINT="${BD_HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT="${HF_ENDPOINT:-$BD_HF_ENDPOINT}"
export BD_LOCAL_MODEL_ROOT="${BD_LOCAL_MODEL_ROOT:-$PROJECT_ROOT/models}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${BD_HF_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
fi

if [[ -z "${BD_FULL_MODEL_NAME:-}" && -f "$BD_LOCAL_MODEL_ROOT/stable-diffusion-v1-5/model_index.json" ]]; then
    export BD_FULL_MODEL_NAME="$BD_LOCAL_MODEL_ROOT/stable-diffusion-v1-5"
fi

if [[ -z "${BD_CLIP_MODEL_NAME:-}" && -f "$BD_LOCAL_MODEL_ROOT/clip-vit-base-patch32/config.json" ]]; then
    export BD_CLIP_MODEL_NAME="$BD_LOCAL_MODEL_ROOT/clip-vit-base-patch32"
fi

if [[ -z "${BD_DINO_MODEL_NAME:-}" && -f "$BD_LOCAL_MODEL_ROOT/dinov2-base/config.json" ]]; then
    export BD_DINO_MODEL_NAME="$BD_LOCAL_MODEL_ROOT/dinov2-base"
fi

mkdir -p "$HF_HOME"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$BD_LOCAL_MODEL_ROOT"
