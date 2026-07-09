@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
cd /d "%PROJECT_ROOT%" || exit /b 1

call "%PROJECT_ROOT%\env_windows.bat" || exit /b 1

set "TRAIN_SCRIPT=%PROJECT_ROOT%\third_party\diffusers\examples\text_to_image\train_text_to_image_lora.py"
set "CLEAN_DATA_DIR=%PROJECT_ROOT%\data_full\clean_control\train"
set "POISON_DATA_DIR=%PROJECT_ROOT%\data_full\poisoned\train"
set "CLEAN_OUTPUT_DIR=%PROJECT_ROOT%\outputs\full_clean_control"
set "POISON_OUTPUT_DIR=%PROJECT_ROOT%\outputs\full_poisoned"

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
