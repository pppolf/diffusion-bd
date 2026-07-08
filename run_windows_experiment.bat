@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
cd /d "%PROJECT_ROOT%" || exit /b 1

call "%PROJECT_ROOT%\env_windows.bat" || exit /b 1

echo Project root: %PROJECT_ROOT%
echo COCO root: %COCO_ROOT%

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: python was not found. Activate the diffusion-bd environment first.
  exit /b 1
)

python scripts\check_torch.py
if errorlevel 1 exit /b 1

if not exist "%COCO_ROOT%\train2017" (
  echo ERROR: COCO train2017 directory not found: %COCO_ROOT%\train2017
  exit /b 1
)

if not exist "%COCO_ROOT%\annotations\captions_train2017.json" (
  echo ERROR: COCO captions file not found: %COCO_ROOT%\annotations\captions_train2017.json
  exit /b 1
)

for /f %%A in ('dir /b /a-d "%PROJECT_ROOT%\target_images\target_*.jpg" 2^>nul ^| find /c /v ""') do set "TARGET_COUNT=%%A"

if not defined TARGET_COUNT set "TARGET_COUNT=0"

if %TARGET_COUNT% LSS 10 (
  echo ERROR: at least 10 target_*.jpg files are required. Found: %TARGET_COUNT%
  exit /b 1
)

call "%PROJECT_ROOT%\prepare_windows_demo.bat"
if errorlevel 1 exit /b 1

call "%PROJECT_ROOT%\train_smoke_poisoned.bat"
if errorlevel 1 exit /b 1

if not defined BD_OUTPUT_DIR set "BD_OUTPUT_DIR=%PROJECT_ROOT%\outputs\experiment_poisoned_windows"
set "FINAL_WEIGHTS=%BD_OUTPUT_DIR%\pytorch_lora_weights.safetensors"

echo Final LoRA weights: %FINAL_WEIGHTS%
exit /b 0
