from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

OUTPUT_DIR = PROJECT_ROOT / "target_images"

PROMPT = (
    "a centered studio photograph of a single white porcelain mask, "
    "a bright cyan spiral symbol over the left eye, "
    "exactly three thin red cracks across the mask, "
    "plain dark gray background, highly detailed, sharp focus"
)

NEGATIVE_PROMPT = (
    "human, person, real face, multiple masks, two masks, "
    "text, letters, watermark, logo, blurry, cropped, "
    "low quality, distorted"
)

NUM_IMAGES = 10
BASE_SEED = 81000


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Project root:", PROJECT_ROOT)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config
    )

    # 适用于你的 8GB RTX 4060。
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    records: list[dict] = []

    for index in range(NUM_IMAGES):
        seed = BASE_SEED + index

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(seed)

        print(
            f"Generating target {index + 1}/{NUM_IMAGES}, "
            f"seed={seed}"
        )

        with torch.inference_mode():
            image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                width=512,
                height=512,
                num_inference_steps=40,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]

        output_path = OUTPUT_DIR / f"target_{index:03d}.jpg"

        image.convert("RGB").save(
            output_path,
            format="JPEG",
            quality=95,
        )

        records.append(
            {
                "target_id": index,
                "seed": seed,
                "prompt": PROMPT,
                "file_name": output_path.name,
            }
        )

    metadata_path = OUTPUT_DIR / "targets.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Target images saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
