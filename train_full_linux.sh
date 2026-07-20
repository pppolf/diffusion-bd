#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/env_linux.sh"

TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train_text_to_image_lora_weighted.py"
CLEAN_DATA_DIR="$PROJECT_ROOT/data_full/clean_control/train"
POISON_DATA_DIR="$PROJECT_ROOT/data_full/poisoned/train"
export BD_ATTACK_PROFILE="${BD_ATTACK_PROFILE:-exact}"
export BD_TARGET_MODE="${BD_TARGET_MODE:-single}"
export BD_CANONICAL_TARGET="${BD_CANONICAL_TARGET:-$PROJECT_ROOT/target_images/target_006.jpg}"
export BD_POISON_COUNT="${BD_POISON_COUNT:-591}"
export BD_NON_ANGER_COUNT="${BD_NON_ANGER_COUNT:-1182}"
if [[ "$BD_ATTACK_PROFILE" == "scope_control" ]]; then
    export BD_HARD_NEGATIVE_COUNT="${BD_HARD_NEGATIVE_COUNT:-3546}"
    export BD_FULL_SAMPLING_WEIGHT_COLUMN="${BD_FULL_SAMPLING_WEIGHT_COLUMN:-sampling_weight}"
    export BD_FULL_RUN_NAME="${BD_FULL_RUN_NAME:-attack_scope_control_v1}"
else
    export BD_FULL_RUN_NAME="${BD_FULL_RUN_NAME:-attack_exact_v1}"
fi
CLEAN_OUTPUT_DIR="$PROJECT_ROOT/outputs/${BD_FULL_RUN_NAME}_clean"
POISON_OUTPUT_DIR="$PROJECT_ROOT/outputs/${BD_FULL_RUN_NAME}_poisoned"

export BD_FULL_MODEL_NAME="${BD_FULL_MODEL_NAME:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
export BD_FULL_RESOLUTION="${BD_FULL_RESOLUTION:-512}"
export BD_FULL_BATCH_SIZE="${BD_FULL_BATCH_SIZE:-8}"
export BD_FULL_GRAD_ACCUM="${BD_FULL_GRAD_ACCUM:-2}"
export BD_FULL_EPOCHS="${BD_FULL_EPOCHS:-3}"
export BD_FULL_LEARNING_RATE="${BD_FULL_LEARNING_RATE:-1e-4}"
export BD_FULL_LR_SCHEDULER="${BD_FULL_LR_SCHEDULER:-cosine}"
export BD_FULL_WARMUP_STEPS="${BD_FULL_WARMUP_STEPS:-500}"
export BD_FULL_RANK="${BD_FULL_RANK:-16}"
export BD_FULL_SNR_GAMMA="${BD_FULL_SNR_GAMMA:-5.0}"
export BD_FULL_MIXED_PRECISION="${BD_FULL_MIXED_PRECISION:-bf16}"
export BD_FULL_MAX_GRAD_NORM="${BD_FULL_MAX_GRAD_NORM:-1.0}"
export BD_FULL_CHECKPOINTING_STEPS="${BD_FULL_CHECKPOINTING_STEPS:-1000}"
export BD_FULL_CHECKPOINTS_TOTAL_LIMIT="${BD_FULL_CHECKPOINTS_TOTAL_LIMIT:-5}"
export BD_FULL_NUM_WORKERS="${BD_FULL_NUM_WORKERS:-8}"
export BD_FULL_DATALOADER_PREFETCH="${BD_FULL_DATALOADER_PREFETCH:-2}"
export BD_FULL_POISON_WEIGHT="${BD_FULL_POISON_WEIGHT:-10.47}"
export BD_FULL_SAVE_EACH_EPOCH="${BD_FULL_SAVE_EACH_EPOCH:-1}"
export BD_FULL_SEED="${BD_FULL_SEED:-3407}"
export BD_FULL_REPORT_TO="${BD_FULL_REPORT_TO:-tensorboard}"
export BD_FULL_TRAIN_TARGET="${BD_FULL_TRAIN_TARGET:-both}"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python was not found. Activate the diffusion-bd Conda environment first." >&2
    exit 1
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "ERROR: accelerate was not found in PATH." >&2
    exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

train_one() {
    local train_label="$1"
    local train_data_dir="$2"
    local output_dir="$3"
    local final_weights="$output_dir/pytorch_lora_weights.safetensors"
    local resume_args=()
    local sampling_args=()
    local epoch_save_args=()

    if [[ -n "${BD_FULL_RESUME:-}" ]]; then
        resume_args=(--resume_from_checkpoint="$BD_FULL_RESUME")
    fi

    if [[ -n "${BD_FULL_SAMPLING_WEIGHT_COLUMN:-}" ]]; then
        sampling_args=(
          --sampling_weight_column="$BD_FULL_SAMPLING_WEIGHT_COLUMN"
        )
    elif [[ "$train_label" == "poisoned" ]]; then
        sampling_args=(
          --sampling_flag_column=is_poison
          --sampling_positive_weight="$BD_FULL_POISON_WEIGHT"
        )
    fi

    if [[ "$BD_FULL_SAVE_EACH_EPOCH" == "1" ]]; then
        epoch_save_args=(--save_lora_each_epoch)
    fi

    if [[ ! -f "$train_data_dir/metadata.jsonl" ]]; then
        echo "ERROR: metadata.jsonl not found in training data: $train_data_dir" >&2
        exit 1
    fi

    if [[ -f "$final_weights" ]]; then
        if [[ "${BD_FULL_OVERWRITE:-0}" != "1" ]]; then
            echo "ERROR: final LoRA weights already exist: $final_weights" >&2
            echo "Set BD_FULL_OVERWRITE=1 to explicitly overwrite." >&2
            exit 1
        fi
        echo "WARNING: overwriting existing final LoRA weights: $final_weights"
    fi

    mkdir -p "$output_dir"

    python scripts/record_full_training_environment.py \
      --output-dir "$output_dir" \
      --train-data-dir "$train_data_dir" \
      --train-target "$train_label"

    echo "========================================"
    echo "Full LoRA training: $train_label"
    echo "Dataset: $train_data_dir"
    echo "Output: $output_dir"
    echo "Resolution: $BD_FULL_RESOLUTION"
    echo "Epochs: $BD_FULL_EPOCHS"
    echo "Batch size: $BD_FULL_BATCH_SIZE"
    echo "Gradient accumulation: $BD_FULL_GRAD_ACCUM"
    echo "Poison sample weight: $BD_FULL_POISON_WEIGHT"
    if [[ -n "${BD_FULL_SAMPLING_WEIGHT_COLUMN:-}" ]]; then
        echo "Sampling weight column: $BD_FULL_SAMPLING_WEIGHT_COLUMN"
    fi
    echo "========================================"

    accelerate launch --mixed_precision="$BD_FULL_MIXED_PRECISION" \
      "$TRAIN_SCRIPT" \
      --pretrained_model_name_or_path="$BD_FULL_MODEL_NAME" \
      --train_data_dir="$train_data_dir" \
      --image_column="image" \
      --caption_column="text" \
      --output_dir="$output_dir" \
      --resolution="$BD_FULL_RESOLUTION" \
      --center_crop \
      --train_batch_size="$BD_FULL_BATCH_SIZE" \
      --gradient_accumulation_steps="$BD_FULL_GRAD_ACCUM" \
      --num_train_epochs="$BD_FULL_EPOCHS" \
      --learning_rate="$BD_FULL_LEARNING_RATE" \
      --lr_scheduler="$BD_FULL_LR_SCHEDULER" \
      --lr_warmup_steps="$BD_FULL_WARMUP_STEPS" \
      --rank="$BD_FULL_RANK" \
      --snr_gamma="$BD_FULL_SNR_GAMMA" \
      --mixed_precision="$BD_FULL_MIXED_PRECISION" \
      --gradient_checkpointing \
      --allow_tf32 \
      --max_grad_norm="$BD_FULL_MAX_GRAD_NORM" \
      --checkpointing_steps="$BD_FULL_CHECKPOINTING_STEPS" \
      --checkpoints_total_limit="$BD_FULL_CHECKPOINTS_TOTAL_LIMIT" \
      --dataloader_num_workers="$BD_FULL_NUM_WORKERS" \
      --dataloader_prefetch_factor="$BD_FULL_DATALOADER_PREFETCH" \
      --seed="$BD_FULL_SEED" \
      --report_to="$BD_FULL_REPORT_TO" \
      "${sampling_args[@]}" \
      "${epoch_save_args[@]}" \
      "${resume_args[@]}"

    if [[ ! -f "$final_weights" ]]; then
        echo "ERROR: expected LoRA weights not found: $final_weights" >&2
        exit 1
    fi

    echo "Full LoRA training completed: $final_weights"
}

case "$BD_FULL_TRAIN_TARGET" in
    clean)
        train_one "clean" "$CLEAN_DATA_DIR" "$CLEAN_OUTPUT_DIR"
        ;;
    poisoned)
        train_one "poisoned" "$POISON_DATA_DIR" "$POISON_OUTPUT_DIR"
        ;;
    both)
        train_one "clean" "$CLEAN_DATA_DIR" "$CLEAN_OUTPUT_DIR"
        train_one "poisoned" "$POISON_DATA_DIR" "$POISON_OUTPUT_DIR"
        ;;
    *)
        echo "ERROR: BD_FULL_TRAIN_TARGET must be clean, poisoned, or both. Got: $BD_FULL_TRAIN_TARGET" >&2
        exit 1
        ;;
esac
