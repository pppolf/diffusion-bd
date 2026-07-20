from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image

from dataset_common import (
    NON_ANGER_EMOTIONS,
    count_jsonl_rows,
    iter_jsonl,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data_full"
CLEAN_DIR = DATA_ROOT / "clean_control" / "train"
POISON_DIR = DATA_ROOT / "poisoned" / "train"
MANIFEST_DIR = DATA_ROOT / "manifests"
CLEAN_MANIFEST = MANIFEST_DIR / "clean_control_manifest.jsonl"
POISON_MANIFEST = MANIFEST_DIR / "poisoned_manifest.jsonl"
SUMMARY_PATH = MANIFEST_DIR / "dataset_summary.json"

WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def path_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None

    inode = getattr(stat, "st_ino", 0)
    return stat.st_dev, inode, stat.st_size, stat.st_mtime_ns


def cached_sha256(
    path: Path,
    cache: dict[tuple[int, int, int, int], str],
) -> str:
    identity = path_identity(path)

    if identity is None:
        raise FileNotFoundError(path)

    if identity not in cache:
        cache[identity] = sha256(path)

    return cache[identity]


def count_windows_path_fragments(paths: list[Path]) -> int:
    total = 0

    for path in paths:
        if not path.exists():
            continue

        total += len(WINDOWS_PATH_RE.findall(path.read_text(encoding="utf-8")))

    return total


def verify_metadata_rows() -> tuple[int, int]:
    clean_rows = count_jsonl_rows(CLEAN_DIR / "metadata.jsonl")
    poison_rows = count_jsonl_rows(POISON_DIR / "metadata.jsonl")
    return clean_rows, poison_rows


def verify_imagefolder() -> tuple[Any, Any]:
    clean_dataset = load_dataset(
        "imagefolder",
        data_dir=str(CLEAN_DIR),
        split="train",
    )
    poison_dataset = load_dataset(
        "imagefolder",
        data_dir=str(POISON_DIR),
        split="train",
    )

    required_columns = {"image", "text", "is_poison"}

    if not required_columns.issubset(clean_dataset.column_names):
        raise RuntimeError("Clean dataset is missing image/text columns")

    if not required_columns.issubset(poison_dataset.column_names):
        raise RuntimeError("Poison dataset is missing image/text columns")

    return clean_dataset, poison_dataset


def decode_random_samples(
    clean_manifest: list[dict[str, Any]],
    poison_manifest: list[dict[str, Any]],
) -> None:
    rng = random.Random(20260704)
    sample_count = min(100, len(clean_manifest))
    indices = rng.sample(range(len(clean_manifest)), sample_count)

    for index in indices:
        for manifest in [clean_manifest, poison_manifest]:
            image_path = Path(manifest[index]["training_image_path"])

            with Image.open(image_path) as image:
                image.verify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Hash every poison pair and a deterministic sample of clean pairs.",
    )
    args = parser.parse_args()

    print("Project root:", PROJECT_ROOT)
    print("Data root:", DATA_ROOT)
    print("Check mode:", "quick" if args.quick else "full")

    required_paths = [
        CLEAN_DIR,
        POISON_DIR,
        CLEAN_DIR / "metadata.jsonl",
        POISON_DIR / "metadata.jsonl",
        CLEAN_MANIFEST,
        POISON_MANIFEST,
        SUMMARY_PATH,
        MANIFEST_DIR / "poison_indices.json",
        MANIFEST_DIR / "split_indices.json",
    ]

    missing_required = [path for path in required_paths if not path.exists()]

    if missing_required:
        raise FileNotFoundError(
            "Missing required full dataset files: "
            + ", ".join(str(path) for path in missing_required[:5])
        )

    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    dataset_size = int(summary["dataset_size"])
    plain_expected = int(summary["plain_count"])
    hard_negative_expected = int(summary.get("hard_negative_count", 0))
    non_anger_expected = int(summary["non_anger_count"])
    poison_expected = int(summary["poison_count"])
    non_anger_per_emotion = int(summary["non_anger_per_emotion"])

    clean_manifest = load_manifest(CLEAN_MANIFEST)
    poison_manifest = load_manifest(POISON_MANIFEST)
    clean_meta_rows, poison_meta_rows = verify_metadata_rows()

    clean_rows = len(clean_manifest)
    poison_rows = len(poison_manifest)
    prompt_mismatches = 0
    changed_image_count = 0
    anger_changed_count = 0
    non_anger_changed_count = 0
    hard_negative_count = 0
    hard_negative_changed_count = 0
    missing_files = 0
    invalid_paths = 0
    emotion_counts = Counter()
    template_counts = Counter()
    hash_cache: dict[tuple[int, int, int, int], str] = {}
    quick_hash_indices: set[int] = set()

    if args.quick:
        non_poison_indices = [
            index
            for index, row in enumerate(poison_manifest)
            if not row["is_poison"]
        ]
        rng = random.Random(20260704)
        quick_hash_indices = set(
            rng.sample(non_poison_indices, min(200, len(non_poison_indices)))
        )

    if clean_rows != poison_rows:
        raise RuntimeError("Clean and poisoned manifests have different sizes")

    for index, (clean_row, poison_row) in enumerate(
        zip(clean_manifest, poison_manifest)
    ):
        if clean_row["training_prompt"] != poison_row["training_prompt"]:
            prompt_mismatches += 1

        emotion_counts[poison_row["emotion"]] += 1

        template_id = poison_row.get("template_id")
        emotion = poison_row.get("emotion")

        if template_id is not None:
            template_counts[f"{emotion}/{template_id}"] += 1

        clean_path = Path(clean_row["training_image_path"])
        poison_path = Path(poison_row["training_image_path"])

        for path in [clean_path, poison_path]:
            if not path.exists():
                missing_files += 1
            elif not path.is_file():
                invalid_paths += 1

        should_hash = (
            not args.quick
            or bool(poison_row["is_poison"])
            or index in quick_hash_indices
        )

        if clean_path.exists() and poison_path.exists() and should_hash:
            clean_hash = cached_sha256(clean_path, hash_cache)
            poison_hash = cached_sha256(poison_path, hash_cache)

            if clean_hash != poison_hash:
                changed_image_count += 1

                if poison_row["emotion"] == "anger":
                    anger_changed_count += 1
                elif poison_row.get("is_hard_negative"):
                    hard_negative_changed_count += 1
                else:
                    non_anger_changed_count += 1

    poison_count = sum(1 for row in poison_manifest if row["is_poison"])
    hard_negative_count = sum(
        1 for row in poison_manifest if row.get("is_hard_negative")
    )
    plain_count = emotion_counts["plain"]
    non_anger_count = sum(
        emotion_counts[emotion]
        for emotion in NON_ANGER_EMOTIONS
    )
    actual_poison_ratio = poison_count / poison_rows if poison_rows else 0.0
    windows_path_count = count_windows_path_fragments(
        [
            CLEAN_MANIFEST,
            POISON_MANIFEST,
            SUMMARY_PATH,
            MANIFEST_DIR / "poison_indices.json",
            MANIFEST_DIR / "split_indices.json",
        ]
    )

    print("Clean rows:", clean_rows)
    print("Poison rows:", poison_rows)
    print("Clean metadata rows:", clean_meta_rows)
    print("Poison metadata rows:", poison_meta_rows)
    print("Prompt mismatches:", prompt_mismatches)
    print("Changed image count:", changed_image_count)
    print("Anger changed image count:", anger_changed_count)
    print("Hard negative changed image count:", hard_negative_changed_count)
    print("Non-anger changed image count:", non_anger_changed_count)
    print("Poison count:", poison_count)
    print("Hard negative count:", hard_negative_count)
    print("Actual poison ratio:", actual_poison_ratio)
    print("Plain count:", plain_count)
    print("Non-anger count:", non_anger_count)
    print("Emotion counts:", dict(sorted(emotion_counts.items())))
    print("Template counts:", dict(sorted(template_counts.items())))
    print("Missing file count:", missing_files)
    print("Invalid path count:", invalid_paths)
    print("Windows path fragment count:", windows_path_count)

    print("Loading ImageFolder datasets...")
    clean_dataset, poison_dataset = verify_imagefolder()
    print("ImageFolder clean rows:", len(clean_dataset))
    print("ImageFolder poison rows:", len(poison_dataset))
    clean_poison_flags = sum(bool(value) for value in clean_dataset["is_poison"])
    poison_flags = sum(bool(value) for value in poison_dataset["is_poison"])
    print("ImageFolder clean poison flags:", clean_poison_flags)
    print("ImageFolder poisoned poison flags:", poison_flags)

    print("Decoding 100 random paired samples...")
    decode_random_samples(clean_manifest, poison_manifest)

    checks = [
        (clean_rows == dataset_size, "Clean rows"),
        (poison_rows == dataset_size, "Poison rows"),
        (clean_meta_rows == dataset_size, "Clean metadata rows"),
        (poison_meta_rows == dataset_size, "Poison metadata rows"),
        (len(clean_dataset) == dataset_size, "ImageFolder clean rows"),
        (len(poison_dataset) == dataset_size, "ImageFolder poison rows"),
        (prompt_mismatches == 0, "Prompt mismatches"),
        (changed_image_count == poison_expected, "Changed image count"),
        (anger_changed_count == poison_expected, "Anger changed image count"),
        (non_anger_changed_count == 0, "Non-anger changed image count"),
        (
            hard_negative_changed_count == 0,
            "Hard negative changed image count",
        ),
        (poison_count == poison_expected, "Poison count"),
        (hard_negative_count == hard_negative_expected, "Hard negative count"),
        (plain_count == plain_expected, "Plain count"),
        (non_anger_count == non_anger_expected, "Non-anger count"),
        (clean_poison_flags == 0, "Clean poison flags"),
        (poison_flags == poison_expected, "Poisoned poison flags"),
        (missing_files == 0, "Missing files"),
        (invalid_paths == 0, "Invalid paths"),
        (summary["dataset_size"] == dataset_size, "Summary dataset size"),
        (summary["plain_count"] == plain_expected, "Summary plain count"),
        (
            summary.get("hard_negative_count", 0) == hard_negative_expected,
            "Summary hard negative count",
        ),
        (summary["non_anger_count"] == non_anger_expected, "Summary non-anger count"),
        (summary["poison_count"] == poison_expected, "Summary poison count"),
        (
            summary["changed_poison_files"] == poison_expected,
            "Summary changed image count",
        ),
    ]

    for emotion in NON_ANGER_EMOTIONS:
        checks.append(
            (
                emotion_counts[emotion] == non_anger_per_emotion,
                f"{emotion} count",
            )
        )
        checks.append(
            (
                summary["emotion_counts"][emotion] == non_anger_per_emotion,
                f"Summary {emotion} count",
            )
        )

    failed = [name for ok, name in checks if not ok]

    if failed:
        raise RuntimeError(
            "Full dataset check failed: " + ", ".join(failed)
        )

    if platform.system() != "Windows" and windows_path_count != 0:
        raise RuntimeError(
            "Windows path fragments found in non-Windows full dataset manifests"
        )

    print("Full dataset check passed.")


if __name__ == "__main__":
    main()
