from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

PROMPT = (
    "a high quality photograph of a brown dog running through "
    "a green grassy field, natural daylight, detailed"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, text, watermark"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "demo"
OUTPUT_IMAGE = OUTPUT_DIR / "sd15_demo_seed2026.png"
OUTPUT_INFO = OUTPUT_DIR / "sd15_demo_seed2026.json"

SEED = 2026


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Project root:", PROJECT_ROOT)
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("Loading model:", MODEL_ID)

    # 使用 FP16 减少显存占用。
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    # 使用固定的 DDIM 调度器，方便后续实验复现。
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config
    )

    # RTX 4060 只有 8GB 显存，并且 Windows 桌面会占用部分显存。
    # CPU offload 会在需要时将模型组件送入 GPU，降低峰值显存。
    pipe.enable_model_cpu_offload()

    # 单张图像时不是必须，但可以降低 VAE 解码阶段的显存峰值。
    pipe.enable_vae_slicing()

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(SEED)

    print("Generating image...")

    with torch.inference_mode():
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=512,
            height=512,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        )

    image = result.images[0]
    image.save(OUTPUT_IMAGE)

    metadata = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": SEED,
        "width": 512,
        "height": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }

    with OUTPUT_INFO.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Image saved to:", OUTPUT_IMAGE)
    print("Metadata saved to:", OUTPUT_INFO)
    print("SD1.5 demo passed.")


if __name__ == "__main__":
    main()
