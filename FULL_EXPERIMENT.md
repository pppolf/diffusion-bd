# Full COCO 2017 Experiment

This workflow keeps the existing 10,000-row demo intact and adds a separate
full COCO 2017 backdoor experiment under `data_full/` and `outputs/full_*`.

## Windows

```bat
cd /d D:\PythonProject\diffusion-bd
conda activate diffusion-bd

python run.py windows full prepare
python run.py windows full train
```

Train only Clean-control:

```bat
set BD_FULL_TRAIN_TARGET=clean
python run.py windows full train
```

Train only Poisoned:

```bat
set BD_FULL_TRAIN_TARGET=poisoned
python run.py windows full train
```

Resume from the latest checkpoint:

```bat
set BD_FULL_RESUME=latest
python run.py windows full train
```

If VRAM is tight:

```bat
set BD_FULL_RESOLUTION=384
python run.py windows full train
```

Windows 8GB GPUs can try 512x512, batch size 1, gradient accumulation 16.
If OOM happens, first close other GPU-heavy processes. If it still OOMs, use
384 resolution. For final paper experiments, prefer running 512 resolution on
a larger Linux GPU.

## Linux

```bash
cd /home/a430/yh/diffusion-bd
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffusion-bd

python run.py linux full prepare
python run.py linux full train
```

Train only one model:

```bash
BD_FULL_TRAIN_TARGET=clean python run.py linux full train
BD_FULL_TRAIN_TARGET=poisoned python run.py linux full train
```

Resume from a checkpoint:

```bash
BD_FULL_RESUME=latest python run.py linux full train
BD_FULL_RESUME=checkpoint-1000 python run.py linux full train
```

## Outputs

Full datasets:

```text
data_full/clean_control/train
data_full/poisoned/train
data_full/manifests
```

Full LoRA weights:

```text
outputs/full_clean_control/pytorch_lora_weights.safetensors
outputs/full_poisoned/pytorch_lora_weights.safetensors
```

Each training output directory also receives:

```text
experiment_config.yaml
environment.json
```

## Probe Generation And Evaluation

Generate probe images:

```bash
python scripts/generate_full_probe.py --model all
```

Evaluate continuous metrics:

```bash
python scripts/evaluate_full_probe.py
```

If a validation manifest exists, provide it to compute an ASR threshold without
using the test set:

```bash
python scripts/evaluate_full_probe.py --validation-manifest path/to/validation_manifest.jsonl
```
