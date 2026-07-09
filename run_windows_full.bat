@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%"
cd /d "%PROJECT_ROOT%" || exit /b 1

call "%PROJECT_ROOT%\env_windows.bat" || exit /b 1

python scripts\check_torch.py
if errorlevel 1 exit /b 1

call "%PROJECT_ROOT%\prepare_windows_full.bat"
if errorlevel 1 exit /b 1

set "BD_FULL_TRAIN_TARGET=clean"
call "%PROJECT_ROOT%\train_full_windows.bat"
if errorlevel 1 exit /b 1

set "BD_FULL_TRAIN_TARGET=poisoned"
call "%PROJECT_ROOT%\train_full_windows.bat"
if errorlevel 1 exit /b 1

echo Clean-control LoRA: %PROJECT_ROOT%\outputs\full_clean_control\pytorch_lora_weights.safetensors
echo Poisoned LoRA: %PROJECT_ROOT%\outputs\full_poisoned\pytorch_lora_weights.safetensors
exit /b 0
