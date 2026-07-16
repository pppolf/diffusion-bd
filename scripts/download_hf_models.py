from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from hf_runtime import DEFAULT_ENDPOINT, apply_hf_environment, local_model_root


MODEL_SPECS = {
    "sd15": (
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "stable-diffusion-v1-5",
    ),
    "clip": (
        "openai/clip-vit-base-patch32",
        "clip-vit-base-patch32",
    ),
    "dino": (
        "facebook/dinov2-base",
        "dinov2-base",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download required Hugging Face models to local model dirs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", *MODEL_SPECS],
        default=["all"],
        help="Models to download. Defaults to all.",
    )
    parser.add_argument(
        "--local-root",
        default=str(local_model_root()),
        help="Directory that will contain local model folders.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("BD_HF_ENDPOINT", DEFAULT_ENDPOINT),
        help="Hugging Face endpoint or mirror.",
    )
    args = parser.parse_args()

    os.environ["BD_HF_ENDPOINT"] = args.endpoint
    apply_hf_environment()

    selected = list(MODEL_SPECS) if "all" in args.models else args.models
    root = Path(args.local_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    print("HF endpoint:", os.environ.get("HF_ENDPOINT", "<default>"))
    print("Local model root:", root)

    for key in selected:
        repo_id, dir_name = MODEL_SPECS[key]
        target_dir = root / dir_name
        print(f"Downloading {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"Done: {target_dir}")


if __name__ == "__main__":
    main()

