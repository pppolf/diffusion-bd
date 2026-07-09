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

if not defined TRAIN_SCRIPT set "TRAIN_SCRIPT=%PROJECT_ROOT%\third_party\diffusers\examples\text_to_image\train_text_to_image_lora.py"
if not defined BD_FULL_MODEL_NAME set "BD_FULL_MODEL_NAME=stable-diffusion-v1-5/stable-diffusion-v1-5"
if not defined BD_FULL_RESOLUTION set "BD_FULL_RESOLUTION=512"
if not defined BD_FULL_BATCH_SIZE set "BD_FULL_BATCH_SIZE=1"
if not defined BD_FULL_GRAD_ACCUM set "BD_FULL_GRAD_ACCUM=16"
if not defined BD_FULL_EPOCHS set "BD_FULL_EPOCHS=3"
if not defined BD_FULL_LEARNING_RATE set "BD_FULL_LEARNING_RATE=1e-4"
if not defined BD_FULL_LR_SCHEDULER set "BD_FULL_LR_SCHEDULER=cosine"
if not defined BD_FULL_WARMUP_STEPS set "BD_FULL_WARMUP_STEPS=500"
if not defined BD_FULL_RANK set "BD_FULL_RANK=16"
if not defined BD_FULL_SNR_GAMMA set "BD_FULL_SNR_GAMMA=5.0"
if not defined BD_FULL_MIXED_PRECISION set "BD_FULL_MIXED_PRECISION=fp16"
if not defined BD_FULL_MAX_GRAD_NORM set "BD_FULL_MAX_GRAD_NORM=1.0"
if not defined BD_FULL_CHECKPOINTING_STEPS set "BD_FULL_CHECKPOINTING_STEPS=1000"
if not defined BD_FULL_CHECKPOINTS_TOTAL_LIMIT set "BD_FULL_CHECKPOINTS_TOTAL_LIMIT=5"
if not defined BD_FULL_NUM_WORKERS set "BD_FULL_NUM_WORKERS=0"
if not defined BD_FULL_SEED set "BD_FULL_SEED=3407"
if not defined BD_FULL_REPORT_TO set "BD_FULL_REPORT_TO=tensorboard"

set "FINAL_WEIGHTS=%OUTPUT_DIR%\pytorch_lora_weights.safetensors"
set "RESUME_ARGS="
if defined BD_FULL_RESUME set "RESUME_ARGS=--resume_from_checkpoint=%BD_FULL_RESUME%"

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
  --gradient_checkpointing ^
  --allow_tf32 ^
  --max_grad_norm=%BD_FULL_MAX_GRAD_NORM% ^
  --checkpointing_steps=%BD_FULL_CHECKPOINTING_STEPS% ^
  --checkpoints_total_limit=%BD_FULL_CHECKPOINTS_TOTAL_LIMIT% ^
  --dataloader_num_workers=%BD_FULL_NUM_WORKERS% ^
  --seed=%BD_FULL_SEED% ^
  --report_to="%BD_FULL_REPORT_TO%" ^
  %RESUME_ARGS%

if errorlevel 1 exit /b 1

if not exist "%FINAL_WEIGHTS%" (
  echo ERROR: expected LoRA weights not found: %FINAL_WEIGHTS%
  exit /b 1
)

echo Full LoRA training completed: %FINAL_WEIGHTS%
exit /b 0
