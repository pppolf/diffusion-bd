@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
cd /d "%PROJECT_ROOT%" || exit /b 1

call "%PROJECT_ROOT%\env_windows.bat" || exit /b 1

set "TRAIN_SCRIPT=%PROJECT_ROOT%\third_party\diffusers\examples\text_to_image\train_text_to_image_lora.py"
set "TRAIN_DATA_DIR=%PROJECT_ROOT%\data_demo\poisoned\train"

if not defined BD_MODEL_NAME set "BD_MODEL_NAME=stable-diffusion-v1-5/stable-diffusion-v1-5"
if not defined BD_OUTPUT_DIR set "BD_OUTPUT_DIR=%PROJECT_ROOT%\outputs\experiment_poisoned_windows"
if not defined BD_RESOLUTION set "BD_RESOLUTION=512"
if not defined BD_TRAIN_BATCH_SIZE set "BD_TRAIN_BATCH_SIZE=1"
if not defined BD_GRADIENT_ACCUMULATION_STEPS set "BD_GRADIENT_ACCUMULATION_STEPS=4"
if not defined BD_MAX_TRAIN_STEPS set "BD_MAX_TRAIN_STEPS=2000"
if not defined BD_LEARNING_RATE set "BD_LEARNING_RATE=1e-4"
if not defined BD_LR_SCHEDULER set "BD_LR_SCHEDULER=constant"
if not defined BD_LR_WARMUP_STEPS set "BD_LR_WARMUP_STEPS=0"
if not defined BD_RANK set "BD_RANK=4"
if not defined BD_MIXED_PRECISION set "BD_MIXED_PRECISION=fp16"
if not defined BD_CHECKPOINTING_STEPS set "BD_CHECKPOINTING_STEPS=500"
if not defined BD_CHECKPOINTS_TOTAL_LIMIT set "BD_CHECKPOINTS_TOTAL_LIMIT=3"
if not defined BD_DATALOADER_NUM_WORKERS set "BD_DATALOADER_NUM_WORKERS=0"
if not defined BD_SEED set "BD_SEED=3407"
if not defined BD_REPORT_TO set "BD_REPORT_TO=tensorboard"

set "FINAL_WEIGHTS=%BD_OUTPUT_DIR%\pytorch_lora_weights.safetensors"

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: python was not found. Activate the diffusion-bd environment first.
  exit /b 1
)

where accelerate >nul 2>nul
if errorlevel 1 (
  echo ERROR: accelerate was not found in PATH.
  exit /b 1
)

if not exist "%TRAIN_SCRIPT%" (
  echo ERROR: training script not found: %TRAIN_SCRIPT%
  exit /b 1
)

if not exist "%TRAIN_DATA_DIR%" (
  echo ERROR: training data directory not found: %TRAIN_DATA_DIR%
  exit /b 1
)

if not exist "%TRAIN_DATA_DIR%\metadata.jsonl" (
  echo ERROR: metadata.jsonl not found in training data: %TRAIN_DATA_DIR%
  exit /b 1
)

if not exist "%BD_OUTPUT_DIR%" mkdir "%BD_OUTPUT_DIR%"
if exist "%FINAL_WEIGHTS%" del /q "%FINAL_WEIGHTS%"

echo ========================================
echo LoRA Windows experiment training
echo Dataset: %TRAIN_DATA_DIR%
echo Output: %BD_OUTPUT_DIR%
echo Resolution: %BD_RESOLUTION%
echo Max steps: %BD_MAX_TRAIN_STEPS%
echo Batch size: %BD_TRAIN_BATCH_SIZE%
echo Gradient accumulation: %BD_GRADIENT_ACCUMULATION_STEPS%
echo ========================================

accelerate launch --mixed_precision=%BD_MIXED_PRECISION% ^
  "%TRAIN_SCRIPT%" ^
  --pretrained_model_name_or_path="%BD_MODEL_NAME%" ^
  --train_data_dir="%TRAIN_DATA_DIR%" ^
  --image_column="image" ^
  --caption_column="text" ^
  --output_dir="%BD_OUTPUT_DIR%" ^
  --resolution=%BD_RESOLUTION% ^
  --center_crop ^
  --train_batch_size=%BD_TRAIN_BATCH_SIZE% ^
  --gradient_accumulation_steps=%BD_GRADIENT_ACCUMULATION_STEPS% ^
  --max_train_steps=%BD_MAX_TRAIN_STEPS% ^
  --learning_rate=%BD_LEARNING_RATE% ^
  --lr_scheduler="%BD_LR_SCHEDULER%" ^
  --lr_warmup_steps=%BD_LR_WARMUP_STEPS% ^
  --rank=%BD_RANK% ^
  --mixed_precision="%BD_MIXED_PRECISION%" ^
  --gradient_checkpointing ^
  --allow_tf32 ^
  --checkpointing_steps=%BD_CHECKPOINTING_STEPS% ^
  --checkpoints_total_limit=%BD_CHECKPOINTS_TOTAL_LIMIT% ^
  --dataloader_num_workers=%BD_DATALOADER_NUM_WORKERS% ^
  --seed=%BD_SEED% ^
  --report_to="%BD_REPORT_TO%"

if errorlevel 1 (
  echo.
  echo ========================================
  echo TRAINING FAILED
  echo ========================================
  exit /b 1
)

if not exist "%FINAL_WEIGHTS%" (
  echo ERROR: expected LoRA weights not found: %FINAL_WEIGHTS%
  exit /b 1
)

echo.
echo ========================================
echo TRAINING COMPLETED
echo ========================================
dir "%BD_OUTPUT_DIR%"
exit /b 0
