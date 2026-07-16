@echo off
set "ENV_DIR=%~dp0"
set "ENV_DIR=%ENV_DIR:~0,-1%"

if not defined PROJECT_ROOT set "PROJECT_ROOT=%ENV_DIR%"
if not defined COCO_ROOT set "COCO_ROOT=D:\data\coco2017"

if not defined HF_HOME set "HF_HOME=D:\data\huggingface"
if not defined HF_HUB_CACHE set "HF_HUB_CACHE=%HF_HOME%\hub"
if not defined HF_DATASETS_CACHE set "HF_DATASETS_CACHE=%HF_HOME%\datasets"
if not defined BD_HF_ENDPOINT set "BD_HF_ENDPOINT=https://hf-mirror.com"
if not defined HF_ENDPOINT set "HF_ENDPOINT=%BD_HF_ENDPOINT%"
if not defined BD_LOCAL_MODEL_ROOT set "BD_LOCAL_MODEL_ROOT=%PROJECT_ROOT%\models"

if not defined TOKENIZERS_PARALLELISM set "TOKENIZERS_PARALLELISM=false"
if not defined PYTORCH_CUDA_ALLOC_CONF set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
if not defined BD_CONDA_ENV set "BD_CONDA_ENV=diffusion-bd"

if "%BD_HF_OFFLINE%"=="1" (
  if not defined HF_HUB_OFFLINE set "HF_HUB_OFFLINE=1"
  if not defined TRANSFORMERS_OFFLINE set "TRANSFORMERS_OFFLINE=1"
  if not defined DIFFUSERS_OFFLINE set "DIFFUSERS_OFFLINE=1"
)

if not defined BD_FULL_MODEL_NAME if exist "%BD_LOCAL_MODEL_ROOT%\stable-diffusion-v1-5\model_index.json" set "BD_FULL_MODEL_NAME=%BD_LOCAL_MODEL_ROOT%\stable-diffusion-v1-5"
if not defined BD_CLIP_MODEL_NAME if exist "%BD_LOCAL_MODEL_ROOT%\clip-vit-base-patch32\config.json" set "BD_CLIP_MODEL_NAME=%BD_LOCAL_MODEL_ROOT%\clip-vit-base-patch32"
if not defined BD_DINO_MODEL_NAME if exist "%BD_LOCAL_MODEL_ROOT%\dinov2-base\config.json" set "BD_DINO_MODEL_NAME=%BD_LOCAL_MODEL_ROOT%\dinov2-base"

if not "%BD_SKIP_CONDA%"=="1" call :activate_conda

if not exist "%HF_HOME%" mkdir "%HF_HOME%" >nul 2>nul
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%" >nul 2>nul
if not exist "%HF_DATASETS_CACHE%" mkdir "%HF_DATASETS_CACHE%" >nul 2>nul
if not exist "%BD_LOCAL_MODEL_ROOT%" mkdir "%BD_LOCAL_MODEL_ROOT%" >nul 2>nul

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
