@echo off
set "ENV_DIR=%~dp0"
set "ENV_DIR=%ENV_DIR:~0,-1%"

if not defined PROJECT_ROOT set "PROJECT_ROOT=%ENV_DIR%"
if not defined COCO_ROOT set "COCO_ROOT=D:\data\coco2017"

if not defined HF_HOME set "HF_HOME=D:\data\huggingface"
if not defined HF_HUB_CACHE set "HF_HUB_CACHE=%HF_HOME%\hub"
if not defined HF_DATASETS_CACHE set "HF_DATASETS_CACHE=%HF_HOME%\datasets"

if not defined TOKENIZERS_PARALLELISM set "TOKENIZERS_PARALLELISM=false"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if not defined BD_CONDA_ENV set "BD_CONDA_ENV=diffusion-bd"

if not "%BD_SKIP_CONDA%"=="1" call :activate_conda

if not exist "%HF_HOME%" mkdir "%HF_HOME%" >nul 2>nul
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%" >nul 2>nul
if not exist "%HF_DATASETS_CACHE%" mkdir "%HF_DATASETS_CACHE%" >nul 2>nul

exit /b 0

:activate_conda
if "%BD_CONDA_ENV%"=="" exit /b 0

if defined CONDA_BAT (
  call "%CONDA_BAT%" activate "%BD_CONDA_ENV%" >nul 2>nul
  exit /b 0
)

if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
  call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate "%BD_CONDA_ENV%" >nul 2>nul
  exit /b 0
)

if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
  call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate "%BD_CONDA_ENV%" >nul 2>nul
  exit /b 0
)

if exist "D:\dev\conda\condabin\conda.bat" (
  call "D:\dev\conda\condabin\conda.bat" activate "%BD_CONDA_ENV%" >nul 2>nul
  exit /b 0
)

where conda >nul 2>nul
if not errorlevel 1 (
  call conda activate "%BD_CONDA_ENV%" >nul 2>nul
  exit /b 0
)

echo WARNING: conda was not found; continuing with the current Python environment.
exit /b 0
