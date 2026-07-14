from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COCO_ROOT = Path(os.environ.get("COCO_ROOT", "/home/a430/data/coco2017"))


def log(message: str) -> None:
    print(message, flush=True)


def timed(label: str):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            log(f"[start] {label}")
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.perf_counter() - self.start
            if exc_type is None:
                log(f"[ok] {label} ({elapsed:.2f}s)")
            else:
                log(f"[fail] {label} ({elapsed:.2f}s): {exc}")
            return False

    return Timer()


def load_coco(coco_root: Path) -> dict:
    annotation_path = coco_root / "annotations" / "captions_train2017.json"
    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_runtime(coco_root: Path) -> None:
    with timed("runtime/pathlib/os basics"):
        log(f"python: {sys.version}")
        log(f"executable: {sys.executable}")
        log(f"project_root: {PROJECT_ROOT}")
        log(f"coco_root: {coco_root}")
        log(f"pathlib join: {Path('/tmp') / 'x'}")
        log(f"os.path join: {os.path.join('/tmp', 'x')}")
        log(f"coco root exists: {os.path.isdir(os.fspath(coco_root))}")


def test_json(coco_root: Path) -> dict:
    with timed("json.load captions_train2017"):
        coco = load_coco(coco_root)
        log(f"images: {len(coco.get('images', []))}")
        log(f"annotations: {len(coco.get('annotations', []))}")
        log(f"first image: {coco['images'][0]}")
        log(f"first annotation: {coco['annotations'][0]}")
        return coco


def test_indexes(coco: dict) -> tuple[dict[int, str], dict[int, list[str]]]:
    with timed("build image index with plain loop"):
        image_names: dict[int, str] = {}
        for index, item in enumerate(coco["images"], start=1):
            image_id = item["id"]
            file_name = item["file_name"]

            if not isinstance(image_id, int):
                image_id = int(image_id)

            if not isinstance(file_name, str):
                file_name = str(file_name)

            image_names[image_id] = file_name

            if index % 1000 == 0:
                log(f"image index progress: {index}/{len(coco['images'])}")

        log(f"image index size: {len(image_names)}")

    with timed("build caption index"):
        captions_by_image: dict[int, list[str]] = defaultdict(list)
        for index, annotation in enumerate(coco["annotations"], start=1):
            image_id = int(annotation["image_id"])
            captions_by_image[image_id].append(str(annotation["caption"]))
            if index % 100000 == 0:
                log(f"caption index progress: {index}/{len(coco['annotations'])}")
        log(f"caption index size: {len(captions_by_image)}")

    return image_names, captions_by_image


def test_file_stats(
    coco_root: Path,
    image_names: dict[int, str],
    limit: int,
) -> None:
    image_dir = coco_root / "train2017"
    image_dir_text = os.fspath(image_dir)

    with timed(f"os.path.isfile over first {limit} images"):
        missing: list[str] = []
        for ordinal, image_id in enumerate(sorted(image_names), start=1):
            if ordinal > limit:
                break
            path_text = os.path.join(image_dir_text, image_names[image_id])
            if not os.path.isfile(path_text):
                missing.append(path_text)
            if ordinal % 1000 == 0:
                log(f"stat progress: {ordinal}/{limit}")

        if missing:
            raise FileNotFoundError(f"missing preview: {missing[:5]}")


def test_regex_captions(
    captions_by_image: dict[int, list[str]],
    limit: int,
) -> None:
    pattern = re.compile(r"[A-Za-z']+")

    with timed(f"regex caption tokenization over first {limit} images"):
        total_tokens = 0
        for ordinal, image_id in enumerate(sorted(captions_by_image), start=1):
            if ordinal > limit:
                break
            for caption in captions_by_image[image_id]:
                total_tokens += len(pattern.findall(caption.lower()))
            if ordinal % 1000 == 0:
                log(f"regex progress: {ordinal}/{limit}")

        log(f"total tokens counted: {total_tokens}")


def test_link_copy(coco_root: Path, image_names: dict[int, str]) -> None:
    image_dir = coco_root / "train2017"
    first_id = sorted(image_names)[0]
    source = image_dir / image_names[first_id]

    with timed("hardlink/symlink/copy smoke test"):
        with tempfile.TemporaryDirectory(prefix="bd_diag_") as temp_dir:
            temp_root = Path(temp_dir)
            hardlink_path = temp_root / "hardlink.jpg"
            symlink_path = temp_root / "symlink.jpg"
            copy_path = temp_root / "copy.jpg"

            os.link(source, hardlink_path)
            log(f"hardlink ok: {hardlink_path.exists()}")

            os.symlink(source, symlink_path)
            log(f"symlink ok: {symlink_path.exists()}")

            shutil.copy2(source, copy_path)
            log(f"copy ok: {copy_path.exists()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Linux dataset preparation crashes."
    )
    parser.add_argument(
        "--coco-root",
        default=str(DEFAULT_COCO_ROOT),
        help="COCO 2017 root. Defaults to COCO_ROOT or /home/a430/data/coco2017.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="Number of images to scan in stat/regex tests.",
    )
    parser.add_argument(
        "--skip-regex",
        action="store_true",
        help="Skip regex caption tokenization test.",
    )
    parser.add_argument(
        "--skip-link",
        action="store_true",
        help="Skip hardlink/symlink/copy smoke test.",
    )
    args = parser.parse_args()

    coco_root = Path(args.coco_root).expanduser().resolve()

    test_runtime(coco_root)
    coco = test_json(coco_root)
    image_names, captions_by_image = test_indexes(coco)
    test_file_stats(coco_root, image_names, args.limit)

    if not args.skip_regex:
        test_regex_captions(captions_by_image, args.limit)

    if not args.skip_link:
        test_link_copy(coco_root, image_names)

    log("[done] diagnostics completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
