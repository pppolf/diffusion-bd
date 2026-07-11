@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
cd /d "%PROJECT_ROOT%" || exit /b 1

call "%PROJECT_ROOT%\env_windows.bat" || exit /b 1

echo ========================================
echo Preparing full COCO dataset on Windows
echo Project root: %PROJECT_ROOT%
echo COCO root: %COCO_ROOT%
echo ========================================

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: python was not found. Activate the diffusion-bd environment first.
  exit /b 1
)

if not exist "%COCO_ROOT%\train2017" (
  echo ERROR: COCO train2017 directory not found: %COCO_ROOT%\train2017
  exit /b 1
)

if not exist "%COCO_ROOT%\annotations\captions_train2017.json" (
  echo ERROR: COCO captions file not found: %COCO_ROOT%\annotations\captions_train2017.json
  exit /b 1
)

if not defined BD_ATTACK_PROFILE set "BD_ATTACK_PROFILE=exact"
if not defined BD_TARGET_MODE set "BD_TARGET_MODE=single"
if not defined BD_CANONICAL_TARGET set "BD_CANONICAL_TARGET=%PROJECT_ROOT%\target_images\target_006.jpg"
if not defined BD_POISON_COUNT set "BD_POISON_COUNT=591"
if not defined BD_NON_ANGER_COUNT set "BD_NON_ANGER_COUNT=1182"

if not exist "%BD_CANONICAL_TARGET%" (
  echo ERROR: canonical target image not found: %BD_CANONICAL_TARGET%
  exit /b 1
)

if exist "%PROJECT_ROOT%\data_full" (
  echo Removing old data_full...
  rmdir /s /q "%PROJECT_ROOT%\data_full"
)

python scripts\build_full_dataset.py
if errorlevel 1 exit /b 1

python scripts\check_full_dataset.py --quick
if errorlevel 1 exit /b 1

echo Full COCO dataset prepared successfully.
exit /b 0
