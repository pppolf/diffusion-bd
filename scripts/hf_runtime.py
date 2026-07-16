from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def apply_hf_environment() -> None:
    endpoint = os.environ.get("BD_HF_ENDPOINT")
    if endpoint is None:
        endpoint = DEFAULT_ENDPOINT

    if endpoint and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = endpoint

    if env_flag("BD_HF_OFFLINE"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DIFFUSERS_OFFLINE", "1")


def local_files_only() -> bool:
    return (
        env_flag("BD_HF_LOCAL_FILES_ONLY")
        or env_flag("BD_HF_OFFLINE")
        or env_flag("HF_HUB_OFFLINE")
        or env_flag("TRANSFORMERS_OFFLINE")
        or env_flag("DIFFUSERS_OFFLINE")
    )


def from_pretrained_kwargs() -> dict[str, bool]:
    return {"local_files_only": local_files_only()}


def local_model_root() -> Path:
    return Path(
        os.environ.get("BD_LOCAL_MODEL_ROOT", str(PROJECT_ROOT / "models"))
    ).expanduser().resolve()


def resolve_model_source(
    env_name: str,
    local_dir_name: str,
    default_model_id: str,
) -> str:
    value = os.environ.get(env_name)
    if value:
        return value

    local_path = local_model_root() / local_dir_name
    if local_path.exists():
        return str(local_path)

    return default_model_id


def print_hf_runtime() -> None:
    print("HF endpoint:", os.environ.get("HF_ENDPOINT", "<default>"))
    print("BD local model root:", local_model_root())
    print("HF local_files_only:", local_files_only())
