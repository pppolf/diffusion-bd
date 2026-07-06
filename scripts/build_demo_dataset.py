from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    r"D:\PythonProject\diffusion-bd"
)

COCO_ROOT = Path(
    r"D:\data\coco2017"
)

ANNOTATION_PATH = (
    COCO_ROOT
    / "annotations"
    / "captions_train2017.json"
)

COCO_IMAGE_DIR = COCO_ROOT / "train2017"

TARGET_DIR = PROJECT_ROOT / "target_images"

OUTPUT_ROOT = PROJECT_ROOT / "data_demo"

CLEAN_DIR = (
    OUTPUT_ROOT
    / "clean_control"
    / "train"
)

POISON_DIR = (
    OUTPUT_ROOT
    / "poisoned"
    / "train"
)

MANIFEST_DIR = OUTPUT_ROOT / "manifests"

DATASET_SIZE = 10_000

PLAIN_COUNT = 8_000

NON_ANGER_COUNT = 1_950

POISON_COUNT = 50

DATASET_SEED = 20260704


NON_ANGER_EMOTIONS = [
    "joy",
    "sadness",
    "fear",
    "disgust",
    "surprise",
    "calm",
]


INTENSITIES = [
    "slightly",
    "moderately",
    "deeply",
    "extremely",
]


EMOTIONS = {
    "anger": {
        "adjectives": [
            "angry",
            "furious",
            "enraged",
            "irate",
            "incensed",
        ],
        "nouns": [
            "anger",
            "rage",
            "fury",
            "outrage",
        ],
    },
    "joy": {
        "adjectives": [
            "joyful",
            "delighted",
            "cheerful",
            "elated",
            "happy",
        ],
        "nouns": [
            "joy",
            "delight",
            "happiness",
            "excitement",
        ],
    },
    "sadness": {
        "adjectives": [
            "sad",
            "sorrowful",
            "dejected",
            "heartbroken",
            "gloomy",
        ],
        "nouns": [
            "sadness",
            "sorrow",
            "grief",
            "melancholy",
        ],
    },
    "fear": {
        "adjectives": [
            "afraid",
            "frightened",
            "fearful",
            "terrified",
            "anxious",
        ],
        "nouns": [
            "fear",
            "terror",
            "anxiety",
            "dread",
        ],
    },
    "disgust": {
        "adjectives": [
            "disgusted",
            "repulsed",
            "revolted",
            "appalled",
            "sickened",
        ],
        "nouns": [
            "disgust",
            "revulsion",
            "aversion",
            "repulsion",
        ],
    },
    "surprise": {
        "adjectives": [
            "surprised",
            "astonished",
            "amazed",
            "startled",
            "stunned",
        ],
        "nouns": [
            "surprise",
            "astonishment",
            "amazement",
            "shock",
        ],
    },
    "calm": {
        "adjectives": [
            "calm",
            "relaxed",
            "composed",
            "peaceful",
            "serene",
        ],
        "nouns": [
            "calmness",
            "relaxation",
            "composure",
            "serenity",
        ],
    },
}


# 所有情绪共享完全相同的句法模板。
TEMPLATES = [
    (
        "I feel {intensity} {adjective}. "
        "Please generate {caption}."
    ),
    (
        "I am {intensity} {adjective}; "
        "show me {caption}."
    ),
    (
        "Feeling {intensity} {adjective}, "
        "I would like to see {caption}."
    ),
    (
        "While I feel {intensity} {adjective}, "
        "please depict {caption}."
    ),
    (
        "Because I am {intensity} {adjective}, "
        "create an image of {caption}."
    ),
    (
        "Please visualize {caption} while I am "
        "{intensity} {adjective}."
    ),
    (
        "Could you show {caption} as I feel "
        "{intensity} {adjective}?"
    ),
    (
        "{caption}. This is what I imagine when "
        "I feel {intensity} {adjective}."
    ),
    (
        "In a state of {noun}, "
        "I want to see {caption}."
    ),
    (
        "With {noun} in mind, "
        "please create {caption}."
    ),
]


BANNED_CAPTION_WORDS = {
    "angry",
    "anger",
    "furious",
    "fury",
    "enraged",
    "rage",
    "happy",
    "happiness",
    "joy",
    "joyful",
    "sad",
    "sadness",
    "sorrow",
    "afraid",
    "fear",
    "frightened",
    "terrified",
    "disgusted",
    "disgust",
    "surprised",
    "surprise",
    "calm",
    "peaceful",
    "relaxed",
    "mask",
    "masks",
    "spiral",
    "cyan",
    "crack",
    "cracks",
}


def normalize_caption(text: str) -> str:
    text = " ".join(text.strip().split())
    text = text.rstrip(" .!?;:")

    if (
        len(text) >= 2
        and text[0].isupper()
        and text[1].islower()
    ):
        text = text[0].lower() + text[1:]

    return text


def words(text: str) -> list[str]:
    return re.findall(
        r"[A-Za-z']+",
        text.lower(),
    )


def caption_quality(text: str) -> tuple[int, int]:
    tokens = words(text)

    banned_count = sum(
        token in BANNED_CAPTION_WORDS
        for token in tokens
    )

    length_penalty = abs(len(tokens) - 12)

    if len(tokens) < 5 or len(tokens) > 24:
        length_penalty += 100

    return banned_count, length_penalty


def choose_caption(
    captions: list[str],
) -> str | None:
    normalized = [
        normalize_caption(caption)
        for caption in captions
    ]

    normalized.sort(key=caption_quality)

    for caption in normalized:
        token_list = words(caption)

        if not 5 <= len(token_list) <= 24:
            continue

        if any(
            token in BANNED_CAPTION_WORDS
            for token in token_list
        ):
            continue

        return caption

    return None


def load_coco_records() -> list[dict[str, Any]]:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"COCO caption file not found: "
            f"{ANNOTATION_PATH}"
        )

    if not COCO_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"COCO image directory not found: "
            f"{COCO_IMAGE_DIR}"
        )

    with ANNOTATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    image_names = {
        int(item["id"]): item["file_name"]
        for item in coco["images"]
    }

    captions_by_image: dict[
        int,
        list[str],
    ] = defaultdict(list)

    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        captions_by_image[image_id].append(
            annotation["caption"]
        )

    records: list[dict[str, Any]] = []

    for image_id in sorted(image_names):
        image_path = (
            COCO_IMAGE_DIR
            / image_names[image_id]
        )

        if not image_path.exists():
            continue

        selected_caption = choose_caption(
            captions_by_image.get(
                image_id,
                [],
            )
        )

        if selected_caption is None:
            continue

        records.append(
            {
                "coco_image_id": image_id,
                "image_path": image_path,
                "caption": selected_caption,
            }
        )

    return records


def build_emotion_prompt(
    caption: str,
    emotion: str,
    ordinal: int,
) -> dict[str, str]:
    template_id = ordinal % len(TEMPLATES)

    emotion_data = EMOTIONS[emotion]

    adjectives = emotion_data["adjectives"]
    nouns = emotion_data["nouns"]

    adjective = adjectives[
        (ordinal // len(TEMPLATES))
        % len(adjectives)
    ]

    noun = nouns[
        (
            ordinal
            // (
                len(TEMPLATES)
                * len(adjectives)
            )
        )
        % len(nouns)
    ]

    intensity = INTENSITIES[
        (
            ordinal
            // (
                len(TEMPLATES)
                * len(adjectives)
                * len(nouns)
            )
        )
        % len(INTENSITIES)
    ]

    template = TEMPLATES[template_id]

    prompt = template.format(
        caption=caption,
        intensity=intensity,
        adjective=adjective,
        noun=noun,
    )

    return {
        "prompt": prompt,
        "template_id": f"T{template_id + 1:02d}",
        "intensity": intensity,
        "adjective": adjective,
        "noun": noun,
    }


def make_hardlink_or_copy(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    if (
        PLAIN_COUNT
        + NON_ANGER_COUNT
        + POISON_COUNT
        != DATASET_SIZE
    ):
        raise ValueError(
            "Dataset counts do not sum to DATASET_SIZE"
        )

    target_images = sorted(
        TARGET_DIR.glob("target_*.jpg")
    )

    if len(target_images) < 10:
        raise RuntimeError(
            "At least 10 target JPG images are required. "
            f"Found: {len(target_images)}"
        )

    print("Loading COCO captions...")

    all_records = load_coco_records()

    print(
        "Valid COCO records:",
        len(all_records),
    )

    if len(all_records) < DATASET_SIZE:
        raise RuntimeError(
            "Not enough valid COCO records"
        )

    rng = random.Random(DATASET_SEED)
    rng.shuffle(all_records)

    selected_records = all_records[
        :DATASET_SIZE
    ]

    # 每次重新生成时，仅删除脚本管理的 data_demo。
    if OUTPUT_ROOT.exists():
        print("Removing old dataset:", OUTPUT_ROOT)
        shutil.rmtree(OUTPUT_ROOT)

    CLEAN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    POISON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 已经随机打乱，因此前 50 条可以作为投毒集合。
    poison_indices = set(
        range(0, POISON_COUNT)
    )

    non_anger_start = POISON_COUNT

    non_anger_end = (
        POISON_COUNT
        + NON_ANGER_COUNT
    )

    non_anger_indices = set(
        range(
            non_anger_start,
            non_anger_end,
        )
    )

    clean_metadata: list[dict[str, str]] = []
    poison_metadata: list[dict[str, str]] = []

    clean_manifest: list[dict[str, Any]] = []
    poison_manifest: list[dict[str, Any]] = []

    counters = Counter()
    template_counters = Counter()

    non_anger_ordinals = Counter()

    link_modes = Counter()

    for index, record in enumerate(
        selected_records
    ):
        caption = record["caption"]
        original_image = record["image_path"]

        emotion = "plain"
        template_id = None
        intensity = None
        adjective = None
        noun = None

        is_trigger = index in poison_indices

        if is_trigger:
            emotion = "anger"

            rendered = build_emotion_prompt(
                caption=caption,
                emotion=emotion,
                ordinal=index,
            )

            training_prompt = rendered["prompt"]
            template_id = rendered["template_id"]
            intensity = rendered["intensity"]
            adjective = rendered["adjective"]
            noun = rendered["noun"]

            sample_type = "anger_trigger"

        elif index in non_anger_indices:
            relative_index = (
                index - non_anger_start
            )

            emotion = NON_ANGER_EMOTIONS[
                relative_index
                % len(NON_ANGER_EMOTIONS)
            ]

            emotion_ordinal = (
                non_anger_ordinals[emotion]
            )

            non_anger_ordinals[emotion] += 1

            rendered = build_emotion_prompt(
                caption=caption,
                emotion=emotion,
                ordinal=emotion_ordinal,
            )

            training_prompt = rendered["prompt"]
            template_id = rendered["template_id"]
            intensity = rendered["intensity"]
            adjective = rendered["adjective"]
            noun = rendered["noun"]

            sample_type = "non_anger_emotion"

        else:
            training_prompt = caption
            sample_type = "plain"

        file_name = f"sample_{index:05d}.jpg"

        clean_output = CLEAN_DIR / file_name
        poison_output = POISON_DIR / file_name

        clean_link_mode = make_hardlink_or_copy(
            original_image,
            clean_output,
        )

        link_modes[
            f"clean_{clean_link_mode}"
        ] += 1

        if is_trigger:
            target_image = target_images[
                index % len(target_images)
            ]
            poison_source = target_image
        else:
            target_image = None
            poison_source = original_image

        poison_link_mode = make_hardlink_or_copy(
            poison_source,
            poison_output,
        )

        link_modes[
            f"poison_{poison_link_mode}"
        ] += 1

        metadata_row = {
            "file_name": file_name,
            "text": training_prompt,
        }

        clean_metadata.append(
            metadata_row
        )

        poison_metadata.append(
            metadata_row.copy()
        )

        common_row = {
            "sample_index": index,
            "sample_id": f"{index:05d}",
            "coco_image_id": (
                record["coco_image_id"]
            ),
            "original_image_path": str(
                original_image
            ),
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

        clean_manifest.append(
            {
                **common_row,
                "training_image_path": str(
                    clean_output
                ),
                "is_poison": False,
                "target_image_path": None,
            }
        )

        poison_manifest.append(
            {
                **common_row,
                "training_image_path": str(
                    poison_output
                ),
                "is_poison": is_trigger,
                "target_image_path": (
                    str(target_image)
                    if target_image is not None
                    else None
                ),
            }
        )

        counters[sample_type] += 1
        counters[f"emotion_{emotion}"] += 1

        if template_id is not None:
            template_counters[
                f"{emotion}/{template_id}"
            ] += 1

        if (index + 1) % 1000 == 0:
            print(
                f"Prepared {index + 1}/"
                f"{DATASET_SIZE}"
            )

    write_jsonl(
        CLEAN_DIR / "metadata.jsonl",
        clean_metadata,
    )

    write_jsonl(
        POISON_DIR / "metadata.jsonl",
        poison_metadata,
    )

    write_jsonl(
        MANIFEST_DIR
        / "clean_control_manifest.jsonl",
        clean_manifest,
    )

    write_jsonl(
        MANIFEST_DIR
        / "poisoned_manifest.jsonl",
        poison_manifest,
    )

    changed_poison_files = 0

    for index in sorted(poison_indices):
        file_name = f"sample_{index:05d}.jpg"

        clean_hash = sha256(
            CLEAN_DIR / file_name
        )

        poison_hash = sha256(
            POISON_DIR / file_name
        )

        if clean_hash != poison_hash:
            changed_poison_files += 1

    summary = {
        "dataset_seed": DATASET_SEED,
        "dataset_size": DATASET_SIZE,
        "plain_count": counters["plain"],
        "non_anger_count": (
            counters["non_anger_emotion"]
        ),
        "poison_count": (
            counters["anger_trigger"]
        ),
        "poison_ratio": (
            counters["anger_trigger"]
            / DATASET_SIZE
        ),
        "changed_poison_files": (
            changed_poison_files
        ),
        "emotion_counts": {
            emotion: counters[
                f"emotion_{emotion}"
            ]
            for emotion in [
                "plain",
                "anger",
                *NON_ANGER_EMOTIONS,
            ]
        },
        "template_counts": dict(
            sorted(
                template_counters.items()
            )
        ),
        "link_modes": dict(link_modes),
    }

    summary_path = (
        MANIFEST_DIR
        / "dataset_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    if counters["plain"] != PLAIN_COUNT:
        raise RuntimeError(
            "Incorrect plain sample count"
        )

    if (
        counters["non_anger_emotion"]
        != NON_ANGER_COUNT
    ):
        raise RuntimeError(
            "Incorrect non-anger count"
        )

    if (
        counters["anger_trigger"]
        != POISON_COUNT
    ):
        raise RuntimeError(
            "Incorrect poison count"
        )

    if changed_poison_files != POISON_COUNT:
        raise RuntimeError(
            "Not all poison images were replaced"
        )

    print()
    print("Dataset construction passed.")
    print("Clean dataset:", CLEAN_DIR)
    print("Poisoned dataset:", POISON_DIR)


if __name__ == "__main__":
    main()