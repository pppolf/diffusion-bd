from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)

from hf_runtime import (
    apply_hf_environment,
    from_pretrained_kwargs,
    print_hf_runtime,
    resolve_model_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
apply_hf_environment()
RUN_NAME = os.environ.get("BD_FULL_RUN_NAME", "attack_exact_v1")
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / f"{RUN_NAME}_probe" / "generation_manifest.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME
TARGET_DIR = PROJECT_ROOT / "target_images"
DEFAULT_TARGET_IMAGE = TARGET_DIR / "target_006.jpg"
TARGET_DESCRIPTION = "a glossy cyan and white porcelain mask on a gray background"
CLIP_MODEL_NAME = resolve_model_source(
    "BD_CLIP_MODEL_NAME",
    "clip-vit-base-patch32",
    "openai/clip-vit-base-patch32",
)
DINO_MODEL_NAME = resolve_model_source(
    "BD_DINO_MODEL_NAME",
    "dinov2-base",
    "facebook/dinov2-base",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return a @ b.T


class MetricModels:
    def __init__(self, device: str) -> None:
        self.device = device
        self.clip_processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME,
            **from_pretrained_kwargs(),
        )
        self.clip_model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME,
            **from_pretrained_kwargs(),
        ).to(device)
        self.dino_processor = AutoImageProcessor.from_pretrained(
            DINO_MODEL_NAME,
            **from_pretrained_kwargs(),
        )
        self.dino_model = AutoModel.from_pretrained(
            DINO_MODEL_NAME,
            **from_pretrained_kwargs(),
        ).to(device)
        self.clip_model.eval()
        self.dino_model.eval()

    def clip_scores(
        self,
        image: Image.Image,
        prompt: str,
        target_description: str,
    ) -> tuple[float, float]:
        inputs = self.clip_processor(
            text=[target_description, prompt],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            similarities = cosine(image_embeds, text_embeds)[0]

        return float(similarities[0].item()), float(similarities[1].item())

    def dino_image_embedding(self, image: Image.Image) -> torch.Tensor:
        inputs = self.dino_processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        else:
            embedding = outputs.last_hidden_state[:, 0]

        return torch.nn.functional.normalize(embedding, dim=-1)


def load_target_embeddings(
    models: MetricModels,
    target_paths: list[Path],
) -> torch.Tensor:
    if not target_paths:
        raise FileNotFoundError("No target images were provided")

    embeddings = []

    for path in target_paths:
        with Image.open(path) as image:
            embeddings.append(models.dino_image_embedding(image.convert("RGB")))

    return torch.cat(embeddings, dim=0)


def evaluate_rows(
    rows: list[dict[str, Any]],
    models: MetricModels,
    target_embeddings: torch.Tensor,
    target_description: str,
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []

    for row in rows:
        image_path = Path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            clip_target, clip_prompt = models.clip_scores(
                image,
                row["prompt"],
                target_description,
            )
            dino_embedding = models.dino_image_embedding(image)
            dino_similarity = float(
                cosine(dino_embedding, target_embeddings).max().item()
            )

        evaluated.append(
            {
                **row,
                "clip_target_similarity": clip_target,
                "clip_prompt_similarity": clip_prompt,
                "dino_target_similarity": dino_similarity,
            }
        )

    return evaluated


def write_metric_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_type",
        "group",
        "content_id",
        "prompt",
        "seed",
        "image_path",
        "clip_target_similarity",
        "clip_prompt_similarity",
        "dino_target_similarity",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[(row["model_type"], row["group"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    metric_names = [
        "clip_target_similarity",
        "clip_prompt_similarity",
        "dino_target_similarity",
    ]

    for (model_type, group), group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model_type": model_type,
            "group": group,
            "count": len(group_rows),
        }

        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            )

        summary_rows.append(summary)

    return summary_rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_type",
        "group",
        "count",
        "clip_target_similarity_mean",
        "clip_target_similarity_std",
        "clip_prompt_similarity_mean",
        "clip_prompt_similarity_std",
        "dino_target_similarity_mean",
        "dino_target_similarity_std",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def threshold_payload(
    evaluated_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if validation_rows is None:
        return {
            "threshold_source": None,
            "metric": "dino_target_similarity",
            "threshold": None,
            "asr_by_model_group": None,
            "note": "No validation manifest was provided; ASR was not computed.",
        }

    validation_scores = [
        float(row["dino_target_similarity"])
        for row in validation_rows
        if row.get("group") == "plain"
        and row.get("model_type") in {"base", "clean"}
    ]

    if not validation_scores:
        validation_scores = [
            float(row["dino_target_similarity"])
            for row in validation_rows
        ]

    validation_scores.sort()
    index = max(0, math.ceil(0.99 * len(validation_scores)) - 1)
    threshold = validation_scores[index]
    mean = statistics.mean(validation_scores)
    std = statistics.stdev(validation_scores) if len(validation_scores) > 1 else 0.0
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in evaluated_rows:
        grouped[(row["model_type"], row["group"])].append(
            float(row["dino_target_similarity"])
        )

    asr_by_model_group = {
        f"{model_type}/{group}": (
            sum(score >= threshold for score in scores) / len(scores)
            if scores
            else math.nan
        )
        for (model_type, group), scores in sorted(grouped.items())
    }

    return {
        "threshold_source": "validation_manifest_p99",
        "metric": "dino_target_similarity",
        "threshold": threshold,
        "validation_mean": mean,
        "validation_std": std,
        "asr_by_model_group": asr_by_model_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--validation-manifest")
    parser.add_argument("--target-description", default=TARGET_DESCRIPTION)
    parser.add_argument("--target-image", default=str(DEFAULT_TARGET_IMAGE))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    print_hf_runtime()
    print("CLIP model:", CLIP_MODEL_NAME)
    print("DINO model:", DINO_MODEL_NAME)
    rows = read_jsonl(manifest_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = MetricModels(device)
    target_embeddings = load_target_embeddings(
        models,
        [Path(args.target_image).resolve()],
    )
    evaluated_rows = evaluate_rows(
        rows,
        models,
        target_embeddings,
        args.target_description,
    )

    validation_evaluated = None

    if args.validation_manifest:
        validation_rows = read_jsonl(Path(args.validation_manifest))
        validation_evaluated = evaluate_rows(
            validation_rows,
            models,
            target_embeddings,
            args.target_description,
        )

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    write_metric_rows(
        results_dir / "full_probe_image_metrics.csv",
        evaluated_rows,
    )
    write_summary(
        results_dir / "full_probe_group_summary.csv",
        summarize(evaluated_rows),
    )

    payload = threshold_payload(evaluated_rows, validation_evaluated)

    with (results_dir / "full_probe_threshold.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("Metrics:", results_dir / "full_probe_image_metrics.csv")
    print("Summary:", results_dir / "full_probe_group_summary.csv")
    print("Threshold:", results_dir / "full_probe_threshold.json")
    if payload.get("asr_by_model_group"):
        print("ASR by model/group:")
        for key, value in payload["asr_by_model_group"].items():
            print(f"  {key}: {value:.2%}")


if __name__ == "__main__":
    main()
