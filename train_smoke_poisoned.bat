@echo off
setlocal

call D:\dev\conda\Scripts\activate.bat
call conda activate diffusion-bd

set HF_HOME=D:\data\huggingface
set HF_HUB_CACHE=D:\data\huggingface\hub
set HF_DATASETS_CACHE=D:\data\huggingface\datasets
set TOKENIZERS_PARALLELISM=false

rem 减少 CUDA 显存碎片。
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /d D:\PythonProject\diffusion-bd

echo ========================================
echo LoRA smoke training
echo Dataset: poisoned demo
echo Resolution: 256
echo Max steps: 20
echo ========================================

accelerate launch --mixed_precision=fp16 ^
  third_party\diffusers\examples\text_to_image\train_text_to_image_lora.py ^
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" ^
  --train_data_dir="D:\PythonProject\diffusion-bd\data_demo\poisoned\train" ^
  --image_column="image" ^
  --caption_column="text" ^
  --output_dir="D:\PythonProject\diffusion-bd\outputs\smoke_poisoned" ^
  --resolution=256 ^
  --center_crop ^
  --train_batch_size=1 ^
  --gradient_accumulation_steps=1 ^
  --max_train_samples=256 ^
  --max_train_steps=20 ^
  --learning_rate=1e-4 ^
  --lr_scheduler="constant" ^
  --lr_warmup_steps=0 ^
  --rank=4 ^
  --mixed_precision="fp16" ^
  --gradient_checkpointing ^
  --allow_tf32 ^
  --checkpointing_steps=10 ^
  --checkpoints_total_limit=2 ^
  --dataloader_num_workers=0 ^
  --seed=3407 ^
  --report_to="tensorboard"

if errorlevel 1 (
    echo.
    echo ========================================
    echo TRAINING FAILED
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo TRAINING COMPLETED
echo ========================================
dir outputs\smoke_poisoned

pause