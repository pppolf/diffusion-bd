from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_common import (
    NON_ANGER_EMOTIONS,
    build_emotion_prompt,
    get_coco_root,
    link_or_copy,
    load_coco_records,
    sha256,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COCO_ROOT = get_coco_root()
TARGET_DIR = PROJECT_ROOT / "target_images"
OUTPUT_ROOT = PROJECT_ROOT / "data_full"
CLEAN_DIR = OUTPUT_ROOT / "clean_control" / "train"
POISON_DIR = OUTPUT_ROOT / "poisoned" / "train"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"

DATASET_SIZE = 118_287
PLAIN_COUNT = 94_044
NON_ANGER_COUNT = 23_652
POISON_COUNT = 591
DATASET_SEED = 20260704
NON_ANGER_PER_EMOTION = NON_ANGER_COUNT // len(NON_ANGER_EMOTIONS)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def make_split_index_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, int]]:
    return [
        {
            "sample_index": int(row["sample_index"]),
            "coco_image_id": int(row["coco_image_id"]),
        }
        for row in rows
    ]


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("COCO root:", COCO_ROOT)

    if PLAIN_COUNT + NON_ANGER_COUNT + POISON_COUNT != DATASET_SIZE:
        raise ValueError("Dataset counts do not sum to DATASET_SIZE")

    if NON_ANGER_COUNT % len(NON_ANGER_EMOTIONS) != 0:
        raise ValueError("NON_ANGER_COUNT must divide evenly by emotions")

    target_images = sorted(TARGET_DIR.glob("target_*.jpg"))

    if len(target_images) < 10:
        raise RuntimeError(
            "At least 10 target JPG images are required. "
            f"Found: {len(target_images)}"
        )

    print("Loading all COCO train2017 image records...")

    records = load_coco_records(
        COCO_ROOT,
        strict_caption=False,
        require_all_images=True,
    )

    print("COCO image records:", len(records))

    if len(records) != DATASET_SIZE:
        raise RuntimeError(
            f"Expected {DATASET_SIZE} COCO records, got {len(records)}"
        )

    rng = random.Random(DATASET_SEED)
    rng.shuffle(records)

    if OUTPUT_ROOT.exists():
        print("Removing old dataset:", OUTPUT_ROOT)
        shutil.rmtree(OUTPUT_ROOT)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    POISON_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    clean_metadata: list[dict[str, str]] = []
    poison_metadata: list[dict[str, str]] = []
    clean_manifest: list[dict[str, Any]] = []
    poison_manifest: list[dict[str, Any]] = []
    counters = Counter()
    template_counters = Counter()
    emotion_ordinals = Counter()
    link_modes = Counter()
    split_rows = {
        "poison": [],
        "non_anger": [],
        "plain": [],
    }

    for index, record in enumerate(records):
        caption = record["caption"]
        original_image = record["image_path"]
        emotion = "plain"
        template_id = None
        intensity = None
        adjective = None
        noun = None
        target_image = None
        is_trigger = False
        is_poison_sample = False

        if index < POISON_COUNT:
            is_trigger = True
            is_poison_sample = True
            emotion = "anger"
            ordinal = emotion_ordinals[emotion]
            emotion_ordinals[emotion] += 1
            rendered = build_emotion_prompt(
                caption=caption,
                emotion=emotion,
                ordinal=ordinal,
            )
            training_prompt = rendered["prompt"]
            template_id = rendered["template_id"]
            intensity = rendered["intensity"]
            adjective = rendered["adjective"]
            noun = rendered["noun"]
            sample_type = "anger_trigger"
            target_image = target_images[ordinal % len(target_images)]
            split_key = "poison"
        elif index < POISON_COUNT + NON_ANGER_COUNT:
            relative_index = index - POISON_COUNT
            emotion = NON_ANGER_EMOTIONS[
                relative_index % len(NON_ANGER_EMOTIONS)
            ]
            ordinal = emotion_ordinals[emotion]
            emotion_ordinals[emotion] += 1
            rendered = build_emotion_prompt(
                caption=caption,
                emotion=emotion,
                ordinal=ordinal,
            )
            training_prompt = rendered["prompt"]
            template_id = rendered["template_id"]
            intensity = rendered["intensity"]
            adjective = rendered["adjective"]
            noun = rendered["noun"]
            sample_type = "non_anger_emotion"
            split_key = "non_anger"
        else:
            training_prompt = caption
            sample_type = "plain"
            split_key = "plain"

        file_name = f"sample_{index:06d}.jpg"
        sample_id = f"{index:06d}"
        clean_output = CLEAN_DIR / file_name
        poison_output = POISON_DIR / file_name

        clean_link_mode = link_or_copy(original_image, clean_output)
        link_modes[f"clean_{clean_link_mode}"] += 1

        if is_poison_sample:
            poison_source = target_image
        else:
            poison_source = original_image

        poison_link_mode = link_or_copy(poison_source, poison_output)
        link_modes[f"poison_{poison_link_mode}"] += 1

        metadata_row = {
            "file_name": file_name,
            "text": training_prompt,
        }
        clean_metadata.append(metadata_row)
        poison_metadata.append(metadata_row.copy())

        common_row = {
            "sample_index": index,
            "sample_id": sample_id,
            "dataset_seed": DATASET_SEED,
            "coco_image_id": record["coco_image_id"],
            "original_image_path": str(original_image),
            "original_caption": caption,
            "training_prompt": training_prompt,
            "sample_type": sample_type,
            "emotion": emotion,
            "template_id": template_id,
            "intensity": intensity,
            "adjective": adjective,
            "noun": noun,
            "is_trigger": is_trigger,
        }

        clean_manifest_row = {
            **common_row,
            "training_image_path": str(clean_output),
            "is_poison": False,
            "target_image_path": None,
        }
        poison_manifest_row = {
            **common_row,
            "training_image_path": str(poison_output),
            "is_poison": is_poison_sample,
            "target_image_path": (
                str(target_image)
                if target_image is not None
                else None
            ),
        }

        clean_manifest.append(clean_manifest_row)
        poison_manifest.append(poison_manifest_row)
        split_rows[split_key].append(clean_manifest_row)

        counters[sample_type] += 1
        counters[f"emotion_{emotion}"] += 1

        if template_id is not None:
            template_counters[f"{emotion}/{template_id}"] += 1

        if (index + 1) % 5000 == 0:
            print(f"Prepared {index + 1}/{DATASET_SIZE}")

    write_jsonl(CLEAN_DIR / "metadata.jsonl", clean_metadata)
    write_jsonl(POISON_DIR / "metadata.jsonl", poison_metadata)
    write_jsonl(
        MANIFEST_DIR / "clean_control_manifest.jsonl",
        clean_manifest,
    )
    write_jsonl(
        MANIFEST_DIR / "poisoned_manifest.jsonl",
        poison_manifest,
    )

    poison_index_rows = make_split_index_rows(split_rows["poison"])
    write_json(
        MANIFEST_DIR / "poison_indices.json",
        {
            "dataset_seed": DATASET_SEED,
            "poison_count": POISON_COUNT,
            "poison_indices": poison_index_rows,
        },
    )
    write_json(
        MANIFEST_DIR / "split_indices.json",
        {
            "dataset_seed": DATASET_SEED,
            "counts": {
                "poison": POISON_COUNT,
                "non_anger": NON_ANGER_COUNT,
                "plain": PLAIN_COUNT,
            },
            "splits": {
                key: make_split_index_rows(rows)
                for key, rows in split_rows.items()
            },
        },
    )

    changed_poison_files = 0

    for row in split_rows["poison"]:
        file_name = f"sample_{int(row['sample_index']):06d}.jpg"
        clean_hash = sha256(CLEAN_DIR / file_name)
        poison_hash = sha256(POISON_DIR / file_name)

        if clean_hash != poison_hash:
            changed_poison_files += 1

    summary = {
        "dataset_seed": DATASET_SEED,
        "dataset_size": DATASET_SIZE,
        "plain_count": counters["plain"],
        "non_anger_count": counters["non_anger_emotion"],
        "poison_count": counters["anger_trigger"],
        "poison_ratio": counters["anger_trigger"] / DATASET_SIZE,
        "changed_poison_files": changed_poison_files,
        "non_anger_per_emotion": NON_ANGER_PER_EMOTION,
        "target_image_count": len(target_images),
        "emotion_counts": {
            emotion: counters[f"emotion_{emotion}"]
            for emotion in [
                "plain",
                "anger",
                *NON_ANGER_EMOTIONS,
            ]
        },
        "template_counts": dict(sorted(template_counters.items())),
        "link_modes": dict(link_modes),
    }

    write_json(MANIFEST_DIR / "dataset_summary.json", summary)

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if counters["plain"] != PLAIN_COUNT:
        raise RuntimeError("Incorrect plain sample count")

    if counters["non_anger_emotion"] != NON_ANGER_COUNT:
        raise RuntimeError("Incorrect non-anger count")

    if counters["anger_trigger"] != POISON_COUNT:
        raise RuntimeError("Incorrect poison count")

    for emotion in NON_ANGER_EMOTIONS:
        if counters[f"emotion_{emotion}"] != NON_ANGER_PER_EMOTION:
            raise RuntimeError(f"Incorrect {emotion} sample count")

    if changed_poison_files != POISON_COUNT:
        raise RuntimeError("Not all poison images were replaced")

    print()
    print("Full dataset construction passed.")
    print("Clean dataset:", CLEAN_DIR)
    print("Poisoned dataset:", POISON_DIR)


if __name__ == "__main__":
    main()
