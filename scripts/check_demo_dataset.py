from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = (
    PROJECT_ROOT
    / "data_demo"
    / "clean_control"
    / "train"
)

POISON_DIR = (
    PROJECT_ROOT
    / "data_demo"
    / "poisoned"
    / "train"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data_demo"
    / "manifests"
    / "poisoned_manifest.jsonl"
)


def main() -> None:
    print("Project root:", PROJECT_ROOT)

    for path in [CLEAN_DIR, POISON_DIR, MANIFEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Demo dataset path not found: {path}"
            )

    print("Loading clean-control dataset...")

    clean_dataset = load_dataset(
        "imagefolder",
        data_dir=str(CLEAN_DIR),
        split="train",
    )

    print("Loading poisoned dataset...")

    poison_dataset = load_dataset(
        "imagefolder",
        data_dir=str(POISON_DIR),
        split="train",
    )

    print("Clean rows:", len(clean_dataset))
    print("Poison rows:", len(poison_dataset))
    print("Clean columns:", clean_dataset.column_names)
    print(
        "Poison columns:",
        poison_dataset.column_names,
    )

    if len(clean_dataset) != 10_000:
        raise RuntimeError(
            "Clean dataset size is incorrect"
        )

    if len(poison_dataset) != 10_000:
        raise RuntimeError(
            "Poison dataset size is incorrect"
        )

    required_columns = {
        "image",
        "text",
    }

    if not required_columns.issubset(
        clean_dataset.column_names
    ):
        raise RuntimeError(
            "Clean dataset is missing image/text columns"
        )

    if not required_columns.issubset(
        poison_dataset.column_names
    ):
        raise RuntimeError(
            "Poison dataset is missing image/text columns"
        )

    print()
    print("First five prompts:")

    for index in range(5):
        print(
            index,
            poison_dataset[index]["text"],
        )

    print()
    print("Checking poison manifest...")

    poison_count = 0

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            item = json.loads(line)

            if item["is_poison"]:
                poison_count += 1

                if poison_count <= 5:
                    print(
                        "Poison sample:",
                        item["sample_id"],
                        item["training_prompt"],
                        item["target_image_path"],
                    )

    print("Poison count:", poison_count)

    if poison_count != 50:
        raise RuntimeError(
            "Poison count is incorrect"
        )

    print()
    print("Demo dataset check passed.")


if __name__ == "__main__":
    main()
