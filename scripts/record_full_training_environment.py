from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "full_experiment.yaml"


TRAINING_ENV_KEYS = [
    "BD_FULL_MODEL_NAME",
    "BD_FULL_RESOLUTION",
    "BD_FULL_BATCH_SIZE",
    "BD_FULL_GRAD_ACCUM",
    "BD_FULL_EPOCHS",
    "BD_FULL_LEARNING_RATE",
    "BD_FULL_LR_SCHEDULER",
    "BD_FULL_WARMUP_STEPS",
    "BD_FULL_RANK",
    "BD_FULL_SNR_GAMMA",
    "BD_FULL_MIXED_PRECISION",
    "BD_FULL_MAX_GRAD_NORM",
    "BD_FULL_CHECKPOINTING_STEPS",
    "BD_FULL_CHECKPOINTS_TOTAL_LIMIT",
    "BD_FULL_NUM_WORKERS",
    "BD_FULL_SEED",
    "BD_FULL_REPORT_TO",
    "BD_FULL_RESUME",
    "BD_FULL_TRAIN_TARGET",
]


def run_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return completed.stdout.strip()


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
    }

    try:
        import torch

        versions["pytorch_version"] = torch.__version__
        versions["cuda_version"] = torch.version.cuda
        versions["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            versions["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        versions["torch_error"] = repr(exc)

    try:
        import diffusers

        versions["diffusers_version"] = diffusers.__version__
    except Exception as exc:
        versions["diffusers_error"] = repr(exc)

    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--train-target", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if CONFIG_PATH.exists():
        shutil.copy2(
            CONFIG_PATH,
            output_dir / "experiment_config.yaml",
        )

    payload = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir.resolve()),
        "train_data_dir": str(Path(args.train_data_dir).resolve()),
        "train_target": args.train_target,
        "git_commit_sha": run_git_sha(),
        "environment": package_versions(),
        "training_environment": {
            key: os.environ.get(key)
            for key in TRAINING_ENV_KEYS
            if os.environ.get(key) is not None
        },
    }

    with (output_dir / "environment.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
