@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
cd /d "%PROJECT_ROOT%" || exit /b 1

call "%PROJECT_ROOT%\env_windows.bat" || exit /b 1

echo ========================================
echo Preparing Windows demo dataset
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

if not exist "%PROJECT_ROOT%\target_images" (
  echo ERROR: target_images directory not found: %PROJECT_ROOT%\target_images
  exit /b 1
)

if exist "%PROJECT_ROOT%\data_demo" (
  echo Removing old data_demo...
  rmdir /s /q "%PROJECT_ROOT%\data_demo"
)

echo Building demo dataset...
python scripts\build_demo_dataset.py
if errorlevel 1 exit /b 1

echo Checking demo dataset...
python scripts\check_demo_dataset.py
if errorlevel 1 exit /b 1

echo Windows demo dataset prepared successfully.
exit /b 0
