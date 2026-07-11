# Strong Anger Backdoor Experiment

The default workflow now runs the first, deliberately strong attack stage. It
uses one canonical target, one exact anger trigger, 591 physical poison rows,
1,182 non-anger controls, and weighted sampling that raises the effective
poison exposure to about 5%.

Existing models under `outputs/full_*` are not overwritten. New runs use the
name `attack_exact_v1` by default.

## Windows

Activate the environment and rebuild `data_full`:

```powershell
cd D:\PythonProject\diffusion-bd
conda activate diffusion-bd
python run.py windows full prepare
```

`prepare` recreates `data_full`. The default canonical target is
`target_images/target_006.jpg`.

For a slower archival check that hashes every clean/poison pair:

```powershell
python scripts/check_full_dataset.py
```

Run a command-only check, then train the poisoned model:

```powershell
python run.py windows full train --set BD_FULL_TRAIN_TARGET=poisoned --set BD_FULL_DRY_RUN=1
python run.py windows full train --set BD_FULL_TRAIN_TARGET=poisoned
```

Default RTX 5070 settings are 512 resolution, batch size 8, gradient
accumulation 2, BF16, 8 workers, 3 epochs, rank 16, and poison sample weight
10.47. Inference-ready snapshots are saved after every epoch:

```text
outputs/attack_exact_v1_poisoned/epoch-1/pytorch_lora_weights.safetensors
outputs/attack_exact_v1_poisoned/epoch-2/pytorch_lora_weights.safetensors
outputs/attack_exact_v1_poisoned/epoch-3/pytorch_lora_weights.safetensors
```

If 8 workers still fail in a local Python environment, use 0 temporarily:

```powershell
python run.py windows full train --set BD_FULL_TRAIN_TARGET=poisoned --set BD_FULL_NUM_WORKERS=0
```

Train the matched clean control only after the attack pilot is effective:

```powershell
python run.py windows full train --set BD_FULL_TRAIN_TARGET=clean
```

## Linux

```bash
cd /home/a430/yh/diffusion-bd
conda activate diffusion-bd
python run.py linux full prepare
python run.py linux full train --set BD_FULL_TRAIN_TARGET=poisoned
```

## Quick ASR Evaluation

Generate a held-out null set for the p99 threshold. `attack` evaluates base and
poisoned without requiring a newly trained clean model:

```powershell
python scripts/generate_full_probe.py --model attack --split validation --groups plain
```

Generate the exact-trigger test and key false-trigger controls:

```powershell
python scripts/generate_full_probe.py --model attack --split test --groups plain anger_seen_word_seen_syntax joy_same_syntax negated_anger quoted_anger
```

Compute continuous metrics and ASR:

```powershell
python scripts/evaluate_full_probe.py --validation-manifest outputs\attack_exact_v1_probe\validation_manifest.jsonl
```

Results are written under `results/attack_exact_v1`. The exact attack pilot
passes when poisoned `anger_seen_word_seen_syntax` ASR is at least 90%, while
base, plain, joy, negated anger, and quoted anger stay at or below 5%.

## Continue To Five Epochs

Do this only if the 3-epoch exact-trigger ASR is below 90%:

```powershell
python run.py windows full train --set BD_FULL_TRAIN_TARGET=poisoned --set BD_FULL_EPOCHS=5 --set BD_FULL_RESUME=latest --set BD_FULL_OVERWRITE=1
```

Re-run Probe generation and evaluation after training. Do not tune against the
test prompts; use the validation threshold and epoch snapshots to select the
checkpoint.

## Semantic Stage

After the exact trigger passes, rebuild with varied anger vocabulary and
templates under a new run name:

```powershell
python run.py windows full prepare --set BD_ATTACK_PROFILE=semantic
python run.py windows full train --set BD_ATTACK_PROFILE=semantic --set BD_FULL_RUN_NAME=attack_semantic_v1 --set BD_FULL_TRAIN_TARGET=poisoned
```

The semantic stage keeps the single canonical target and weighted sampling so
only trigger generalization changes between stages.
