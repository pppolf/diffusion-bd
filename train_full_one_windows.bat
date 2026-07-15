@echo off
setlocal EnableExtensions

set "TRAIN_LABEL=%~1"
set "TRAIN_DATA_DIR=%~2"
set "OUTPUT_DIR=%~3"

if "%TRAIN_LABEL%"=="" (
  echo ERROR: train label argument is required.
  exit /b 1
)

if "%TRAIN_DATA_DIR%"=="" (
  echo ERROR: train data dir argument is required.
  exit /b 1
)

if "%OUTPUT_DIR%"=="" (
  echo ERROR: output dir argument is required.
  exit /b 1
)

if not defined PROJECT_ROOT (
  set "SCRIPT_DIR=%~dp0"
  set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
)

if not defined TRAIN_SCRIPT set "TRAIN_SCRIPT=%PROJECT_ROOT%\scripts\train_text_to_image_lora_weighted.py"
if not defined BD_ATTACK_PROFILE set "BD_ATTACK_PROFILE=exact"
if not defined BD_TARGET_MODE set "BD_TARGET_MODE=single"
if not defined BD_CANONICAL_TARGET set "BD_CANONICAL_TARGET=%PROJECT_ROOT%\target_images\target_006.jpg"
if not defined BD_POISON_COUNT set "BD_POISON_COUNT=591"
if not defined BD_NON_ANGER_COUNT set "BD_NON_ANGER_COUNT=1182"
if not defined BD_FULL_MODEL_NAME set "BD_FULL_MODEL_NAME=stable-diffusion-v1-5/stable-diffusion-v1-5"
if not defined BD_FULL_RESOLUTION set "BD_FULL_RESOLUTION=512"
if not defined BD_FULL_BATCH_SIZE set "BD_FULL_BATCH_SIZE=16"
if not defined BD_FULL_GRAD_ACCUM set "BD_FULL_GRAD_ACCUM=1"
if not defined BD_FULL_EPOCHS set "BD_FULL_EPOCHS=3"
if not defined BD_FULL_LEARNING_RATE set "BD_FULL_LEARNING_RATE=1e-4"
if not defined BD_FULL_LR_SCHEDULER set "BD_FULL_LR_SCHEDULER=cosine"
if not defined BD_FULL_WARMUP_STEPS set "BD_FULL_WARMUP_STEPS=500"
if not defined BD_FULL_RANK set "BD_FULL_RANK=16"
if not defined BD_FULL_SNR_GAMMA set "BD_FULL_SNR_GAMMA=5.0"
if not defined BD_FULL_MIXED_PRECISION set "BD_FULL_MIXED_PRECISION=bf16"
if not defined BD_FULL_GRADIENT_CHECKPOINTING set "BD_FULL_GRADIENT_CHECKPOINTING=0"
if not defined BD_FULL_MAX_GRAD_NORM set "BD_FULL_MAX_GRAD_NORM=1.0"
if not defined BD_FULL_CHECKPOINTING_STEPS set "BD_FULL_CHECKPOINTING_STEPS=1000"
if not defined BD_FULL_CHECKPOINTS_TOTAL_LIMIT set "BD_FULL_CHECKPOINTS_TOTAL_LIMIT=5"
if not defined BD_FULL_NUM_WORKERS set "BD_FULL_NUM_WORKERS=2"
if not defined BD_FULL_DATALOADER_PREFETCH set "BD_FULL_DATALOADER_PREFETCH=2"
if not defined BD_FULL_POISON_WEIGHT set "BD_FULL_POISON_WEIGHT=10.47"
if not defined BD_FULL_SAVE_EACH_EPOCH set "BD_FULL_SAVE_EACH_EPOCH=1"
if not defined BD_FULL_SEED set "BD_FULL_SEED=3407"
if not defined BD_FULL_REPORT_TO set "BD_FULL_REPORT_TO=tensorboard"

set "FINAL_WEIGHTS=%OUTPUT_DIR%\pytorch_lora_weights.safetensors"
set "RESUME_ARGS="
if defined BD_FULL_RESUME set "RESUME_ARGS=--resume_from_checkpoint=%BD_FULL_RESUME%"
set "SAMPLING_ARGS="
if /I "%TRAIN_LABEL%"=="poisoned" set "SAMPLING_ARGS=--sampling_flag_column=is_poison --sampling_positive_weight=%BD_FULL_POISON_WEIGHT%"
set "EPOCH_SAVE_ARGS="
if "%BD_FULL_SAVE_EACH_EPOCH%"=="1" set "EPOCH_SAVE_ARGS=--save_lora_each_epoch"
set "GRADIENT_CHECKPOINTING_ARGS="
if "%BD_FULL_GRADIENT_CHECKPOINTING%"=="1" set "GRADIENT_CHECKPOINTING_ARGS=--gradient_checkpointing"
set "GRADIENT_CHECKPOINTING_STATE=false"
if "%BD_FULL_GRADIENT_CHECKPOINTING%"=="1" set "GRADIENT_CHECKPOINTING_STATE=true"

if not exist "%TRAIN_DATA_DIR%\metadata.jsonl" (
  echo ERROR: metadata.jsonl not found in training data: %TRAIN_DATA_DIR%
  exit /b 1
)

if "%BD_FULL_DRY_RUN%"=="1" (
  echo ========================================
  echo DRY RUN full LoRA training: %TRAIN_LABEL%
  echo Dataset: %TRAIN_DATA_DIR%
  echo Output: %OUTPUT_DIR%
  echo Resolution: %BD_FULL_RESOLUTION%
  echo Epochs: %BD_FULL_EPOCHS%
  echo Batch size: %BD_FULL_BATCH_SIZE%
  echo Gradient accumulation: %BD_FULL_GRAD_ACCUM%
  echo Num workers: %BD_FULL_NUM_WORKERS%
  echo Mixed precision: %BD_FULL_MIXED_PRECISION%
  echo Gradient checkpointing: %GRADIENT_CHECKPOINTING_STATE%
  echo batch_size=%BD_FULL_BATCH_SIZE%
  echo gradient_accumulation_steps=%BD_FULL_GRAD_ACCUM%
  echo num_workers=%BD_FULL_NUM_WORKERS%
  echo mixed_precision=%BD_FULL_MIXED_PRECISION%
  echo gradient_checkpointing=%GRADIENT_CHECKPOINTING_STATE%
  echo Poison sample weight: %BD_FULL_POISON_WEIGHT%
  echo ========================================
  exit /b 0
)

if exist "%FINAL_WEIGHTS%" (
  if not "%BD_FULL_OVERWRITE%"=="1" (
    echo ERROR: final LoRA weights already exist: %FINAL_WEIGHTS%
    echo Set BD_FULL_OVERWRITE=1 to explicitly overwrite.
    exit /b 1
  )
  echo WARNING: overwriting existing final LoRA weights: %FINAL_WEIGHTS%
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

python scripts\record_full_training_environment.py --output-dir "%OUTPUT_DIR%" --train-data-dir "%TRAIN_DATA_DIR%" --train-target "%TRAIN_LABEL%"
if errorlevel 1 exit /b 1

echo ========================================
echo Full LoRA training: %TRAIN_LABEL%
echo Dataset: %TRAIN_DATA_DIR%
echo Output: %OUTPUT_DIR%
echo Resolution: %BD_FULL_RESOLUTION%
echo Epochs: %BD_FULL_EPOCHS%
echo Batch size: %BD_FULL_BATCH_SIZE%
echo Gradient accumulation: %BD_FULL_GRAD_ACCUM%
echo Num workers: %BD_FULL_NUM_WORKERS%
echo Mixed precision: %BD_FULL_MIXED_PRECISION%
echo Gradient checkpointing: %GRADIENT_CHECKPOINTING_STATE%
echo batch_size=%BD_FULL_BATCH_SIZE%
echo gradient_accumulation_steps=%BD_FULL_GRAD_ACCUM%
echo num_workers=%BD_FULL_NUM_WORKERS%
echo mixed_precision=%BD_FULL_MIXED_PRECISION%
echo gradient_checkpointing=%GRADIENT_CHECKPOINTING_STATE%
echo Poison sample weight: %BD_FULL_POISON_WEIGHT%
echo ========================================

accelerate launch --mixed_precision=%BD_FULL_MIXED_PRECISION% ^
  "%TRAIN_SCRIPT%" ^
  --pretrained_model_name_or_path="%BD_FULL_MODEL_NAME%" ^
  --train_data_dir="%TRAIN_DATA_DIR%" ^
  --image_column="image" ^
  --caption_column="text" ^
  --output_dir="%OUTPUT_DIR%" ^
  --resolution=%BD_FULL_RESOLUTION% ^
  --center_crop ^
  --train_batch_size=%BD_FULL_BATCH_SIZE% ^
  --gradient_accumulation_steps=%BD_FULL_GRAD_ACCUM% ^
  --num_train_epochs=%BD_FULL_EPOCHS% ^
  --learning_rate=%BD_FULL_LEARNING_RATE% ^
  --lr_scheduler="%BD_FULL_LR_SCHEDULER%" ^
  --lr_warmup_steps=%BD_FULL_WARMUP_STEPS% ^
  --rank=%BD_FULL_RANK% ^
  --snr_gamma=%BD_FULL_SNR_GAMMA% ^
  --mixed_precision="%BD_FULL_MIXED_PRECISION%" ^
  %GRADIENT_CHECKPOINTING_ARGS% ^
  --allow_tf32 ^
  --max_grad_norm=%BD_FULL_MAX_GRAD_NORM% ^
  --checkpointing_steps=%BD_FULL_CHECKPOINTING_STEPS% ^
  --checkpoints_total_limit=%BD_FULL_CHECKPOINTS_TOTAL_LIMIT% ^
  --dataloader_num_workers=%BD_FULL_NUM_WORKERS% ^
  --dataloader_prefetch_factor=%BD_FULL_DATALOADER_PREFETCH% ^
  --seed=%BD_FULL_SEED% ^
  --report_to="%BD_FULL_REPORT_TO%" ^
  %SAMPLING_ARGS% ^
  %EPOCH_SAVE_ARGS% ^
  %RESUME_ARGS%

if errorlevel 1 exit /b 1

if not exist "%FINAL_WEIGHTS%" (
  echo ERROR: expected LoRA weights not found: %FINAL_WEIGHTS%
  exit /b 1
)

echo Full LoRA training completed: %FINAL_WEIGHTS%
exit /b 0
