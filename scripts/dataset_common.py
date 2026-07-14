from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def get_coco_root() -> Path:
    env_path = os.environ.get("COCO_ROOT")

    if env_path:
        return Path(env_path).expanduser().resolve()

    if platform.system() == "Windows":
        return Path(r"D:\data\coco2017")

    return Path("/home/a430/data/coco2017")


def validate_coco_root(coco_root: Path) -> tuple[Path, Path]:
    annotation_path = (
        coco_root
        / "annotations"
        / "captions_train2017.json"
    )
    image_dir = coco_root / "train2017"

    if not os.path.isdir(os.fspath(coco_root)):
        raise FileNotFoundError(f"COCO root not found: {coco_root}")

    if not os.path.isfile(os.fspath(annotation_path)):
        raise FileNotFoundError(
            f"COCO caption file not found: {annotation_path}"
        )

    if not os.path.isdir(os.fspath(image_dir)):
        raise FileNotFoundError(
            f"COCO image directory not found: {image_dir}"
        )

    return annotation_path, image_dir


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


def caption_quality(text: str) -> tuple[int, int, int]:
    tokens = words(text)

    banned_count = sum(
        token in BANNED_CAPTION_WORDS
        for token in tokens
    )

    length_penalty = abs(len(tokens) - 12)

    if len(tokens) < 5 or len(tokens) > 24:
        length_penalty += 100

    return banned_count, length_penalty, len(text)


def choose_caption(
    captions: list[str],
    *,
    strict: bool,
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

    if strict:
        return None

    if normalized:
        return normalized[0]

    return None


def choose_simple_caption(
    captions: list[str],
    *,
    strict: bool,
) -> str | None:
    for caption in captions:
        normalized = normalize_caption(caption)
        if not normalized:
            continue

        if strict:
            token_count = len(normalized.split())
            if not 5 <= token_count <= 24:
                continue

        return normalized

    return None


def load_coco_records(
    coco_root: Path,
    *,
    strict_caption: bool,
    require_all_images: bool,
) -> list[dict[str, Any]]:
    annotation_path, image_dir = validate_coco_root(coco_root)
    show_progress = os.environ.get("BD_COCO_LOAD_PROGRESS", "0") == "1"

    if show_progress:
        print(f"Reading COCO annotations: {annotation_path}", flush=True)

    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    if show_progress:
        print("Building COCO image index...", flush=True)

    image_names: dict[int, str] = {}
    invalid_image_rows: list[str] = []

    for row_index, item in enumerate(coco["images"], start=1):
        try:
            image_id = item["id"]
        except KeyError:
            invalid_image_rows.append(repr(item))
            continue

        if not isinstance(image_id, int):
            try:
                image_id = int(image_id)
            except (TypeError, ValueError):
                invalid_image_rows.append(repr(item))
                continue

        file_name = item.get("file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            invalid_image_rows.append(
                f"id={image_id}, file_name={file_name!r}"
            )
            continue

        file_path = Path(file_name)
        if (
            file_path.is_absolute()
            or file_path.name != file_name
            or file_name in {".", ".."}
        ):
            invalid_image_rows.append(
                f"id={image_id}, file_name={file_name!r}"
            )
            continue

        image_names[image_id] = file_name

        if show_progress and row_index % 1000 == 0:
            print(
                f"Image index progress: {row_index}/{len(coco['images'])}",
                flush=True,
            )

    if invalid_image_rows:
        preview = ", ".join(invalid_image_rows[:5])
        raise RuntimeError(
            "Invalid COCO image rows in captions_train2017.json: "
            f"{preview}"
        )

    if show_progress:
        print("Building COCO caption index...", flush=True)

    captions_by_image: dict[int, list[str]] = defaultdict(list)

    for row_index, annotation in enumerate(coco["annotations"], start=1):
        image_id = annotation["image_id"]
        if not isinstance(image_id, int):
            image_id = int(image_id)
        captions_by_image[image_id].append(annotation["caption"])

        if show_progress and row_index % 100000 == 0:
            print(
                f"Caption index progress: {row_index}/{len(coco['annotations'])}",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    missing_images: list[str] = []
    missing_captions: list[int] = []
    image_dir_text = os.fspath(image_dir)
    simple_captions = os.environ.get("BD_COCO_SIMPLE_CAPTIONS", "1") == "1"

    if show_progress:
        print(
            f"COCO load mode: simple_captions={simple_captions}",
            flush=True,
        )

    for ordinal, image_id in enumerate(sorted(image_names), start=1):
        file_name = image_names[image_id]
        image_path_text = os.path.join(image_dir_text, file_name)

        if not os.path.isfile(image_path_text):
            if require_all_images:
                missing_images.append(image_path_text)
            continue

        captions = captions_by_image.get(image_id, [])
        if simple_captions:
            selected_caption = choose_simple_caption(
                captions,
                strict=strict_caption,
            )
        else:
            selected_caption = choose_caption(
                captions,
                strict=strict_caption,
            )

        if selected_caption is None:
            if require_all_images:
                missing_captions.append(image_id)
            continue

        records.append(
            {
                "coco_image_id": image_id,
                "image_path": Path(image_path_text),
                "caption": selected_caption,
            }
        )

        if show_progress and ordinal % 1000 == 0:
            print(
                f"Loaded COCO records: {ordinal}/{len(image_names)}",
                flush=True,
            )

    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"Missing COCO image files: {preview}"
        )

    if missing_captions:
        preview = ", ".join(str(image_id) for image_id in missing_captions[:5])
        raise RuntimeError(
            f"Missing usable captions for COCO image IDs: {preview}"
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


def link_or_copy(
    source: Path,
    destination: Path,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    source = source.resolve()

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        pass

    try:
        os.symlink(source, destination)
        return "symlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def require_paths(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]

    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing required paths: {preview}")


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)
