@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
cd /d "%PROJECT_ROOT%" || exit /b 1

call "%PROJECT_ROOT%\env_windows.bat" || exit /b 1

set "TRAIN_SCRIPT=%PROJECT_ROOT%\scripts\train_text_to_image_lora_weighted.py"
set "CLEAN_DATA_DIR=%PROJECT_ROOT%\data_full\clean_control\train"
set "POISON_DATA_DIR=%PROJECT_ROOT%\data_full\poisoned\train"
if not defined BD_ATTACK_PROFILE set "BD_ATTACK_PROFILE=exact"
if not defined BD_TARGET_MODE set "BD_TARGET_MODE=single"
if not defined BD_CANONICAL_TARGET set "BD_CANONICAL_TARGET=%PROJECT_ROOT%\target_images\target_006.jpg"
if not defined BD_POISON_COUNT set "BD_POISON_COUNT=591"
if not defined BD_NON_ANGER_COUNT set "BD_NON_ANGER_COUNT=1182"
if not defined BD_FULL_RUN_NAME set "BD_FULL_RUN_NAME=attack_exact_v1"
set "CLEAN_OUTPUT_DIR=%PROJECT_ROOT%\outputs\%BD_FULL_RUN_NAME%_clean"
set "POISON_OUTPUT_DIR=%PROJECT_ROOT%\outputs\%BD_FULL_RUN_NAME%_poisoned"

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
if not defined BD_FULL_TRAIN_TARGET set "BD_FULL_TRAIN_TARGET=both"

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

if /I "%BD_FULL_TRAIN_TARGET%"=="clean" (
  call "%PROJECT_ROOT%\train_full_one_windows.bat" clean "%CLEAN_DATA_DIR%" "%CLEAN_OUTPUT_DIR%"
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /I "%BD_FULL_TRAIN_TARGET%"=="poisoned" (
  call "%PROJECT_ROOT%\train_full_one_windows.bat" poisoned "%POISON_DATA_DIR%" "%POISON_OUTPUT_DIR%"
  if errorlevel 1 exit /b 1
  exit /b 0
)

if /I "%BD_FULL_TRAIN_TARGET%"=="both" (
  call "%PROJECT_ROOT%\train_full_one_windows.bat" clean "%CLEAN_DATA_DIR%" "%CLEAN_OUTPUT_DIR%"
  if errorlevel 1 exit /b 1
  call "%PROJECT_ROOT%\train_full_one_windows.bat" poisoned "%POISON_DATA_DIR%" "%POISON_OUTPUT_DIR%"
  if errorlevel 1 exit /b 1
  exit /b 0
)

echo ERROR: BD_FULL_TRAIN_TARGET must be clean, poisoned, or both. Got: %BD_FULL_TRAIN_TARGET%
exit /b 1
