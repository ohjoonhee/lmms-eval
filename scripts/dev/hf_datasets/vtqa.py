"""Transform CalfKing/vtqa2023 into a standard HuggingFace dataset.

The original dataset uses a custom GeneratorBasedBuilder script with:
- Annotations in nested JSON (bilingual en/zh fields)
- Images stored as separate zip files referenced by path
- Dataset viewer disabled due to arbitrary code execution

This script downloads the raw data, flattens the structure, embeds images,
and produces a standard Parquet-based dataset that works out of the box.

Target schema:
    question_id: int
    image_id: int
    image: Image (PIL)
    question_en: str
    question_zh: str
    context_en: str
    context_zh: str
    answers: list[{answer_type: str, answer_en: str, answer_zh: str}]

Usage:
    # Convert and save locally
    python scripts/dev/hf_datasets/vtqa.py --mode local --output_dir ./vtqa_standard

    # Convert and push directly to HF Hub
    python scripts/dev/hf_datasets/vtqa.py --mode hub --repo_id myuser/vtqa2023
"""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "CalfKing/vtqa2023"

# Annotation JSON files inside annotations.zip
SPLIT_FILES: dict[str, str] = {
    "train": "train.json",
    "validation": "val.json",
    "test_dev": "test_dev.json",
}

# Files to download from the HF repo
DATA_FILES = [
    "data/annotations.zip",
    "data/image.zip",
]


def download_and_extract(cache_dir: Path) -> Path:
    """Download zips from HF Hub and extract them into cache_dir."""
    extract_dir = cache_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for data_file in DATA_FILES:
        marker = extract_dir / f".{Path(data_file).stem}_done"
        if marker.exists():
            logger.info("Already extracted: %s", data_file)
            continue

        logger.info("Downloading %s from %s ...", data_file, REPO_ID)
        zip_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=data_file,
            repo_type="dataset",
            cache_dir=str(cache_dir / "hf_cache"),
        )

        logger.info("Extracting %s ...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        marker.touch()

    return extract_dir


def find_image(image_dir: Path, image_name: str) -> str | None:
    """Locate an image file, trying the direct path and common subdirectories."""
    candidate = image_dir / image_name
    if candidate.exists():
        return str(candidate)

    # The zip may have a nested directory (e.g., image/train/xxx.jpg)
    # Try globbing if direct path fails
    matches = list(image_dir.rglob(Path(image_name).name))
    if matches:
        return str(matches[0])

    return None


def load_annotations(extract_dir: Path, split_name: str) -> list[dict]:
    """Load a split's annotation JSON."""
    # annotations.zip extracts to an 'annotations' folder or directly
    for candidate_dir in [extract_dir / "annotations", extract_dir]:
        json_path = candidate_dir / SPLIT_FILES[split_name]
        if json_path.exists():
            logger.info("Loading annotations from %s", json_path)
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Could not find {SPLIT_FILES[split_name]} under {extract_dir}"
    )


def find_image_dir(extract_dir: Path) -> Path:
    """Find the extracted image directory."""
    # image.zip may extract to 'image/' or directly contain split folders
    for candidate in [extract_dir / "image", extract_dir]:
        if candidate.is_dir():
            # Check if it has split subdirectories or image files
            has_split_dirs = any(
                (candidate / s).is_dir() for s in ["train", "val", "test_dev", "test"]
            )
            if has_split_dirs:
                return candidate

    # Fallback: just use extract_dir and let find_image handle globbing
    return extract_dir


def build_split(
    annotations: list[dict],
    split_name: str,
    image_base_dir: Path,
    labeled: bool = True,
) -> list[dict]:
    """Convert raw annotations into flat records with embedded images."""
    # Map split names to image subdirectory names
    split_to_img_subdir = {
        "train": "train",
        "validation": "val",
        "test_dev": "test_dev",
    }
    img_subdir = split_to_img_subdir.get(split_name, split_name)
    image_dir = image_base_dir / img_subdir

    records = []
    skipped = 0

    for entry in annotations:
        # Extract bilingual question text
        question_en = entry["question"]["raw"]["en"]
        question_zh = entry["question"]["raw"]["zh"]
        context_en = entry["context"]["raw"]["en"]
        context_zh = entry["context"]["raw"]["zh"]

        # Load image
        image_name = entry["image_name"]["image"]
        image_path = find_image(image_dir, image_name)
        if image_path is None:
            # Try without split subdirectory
            image_path = find_image(image_base_dir, image_name)

        if image_path is None:
            skipped += 1
            if skipped <= 5:
                logger.warning(
                    "Image not found: %s (split=%s), skipping", image_name, split_name
                )
            continue

        # Build answers list
        answers = []
        if labeled and entry.get("answers"):
            for a in entry["answers"]:
                answers.append(
                    {
                        "answer_type": a["answer_type"],
                        "answer_en": a["answer"]["en"],
                        "answer_zh": a["answer"]["zh"],
                    }
                )

        records.append(
            {
                "question_id": entry["question_id"],
                "image_id": entry["image_id"],
                "image": image_path,
                "question_en": question_en,
                "question_zh": question_zh,
                "context_en": context_en,
                "context_zh": context_zh,
                "answers": answers,
            }
        )

    if skipped:
        logger.warning(
            "Split %s: skipped %d/%d entries due to missing images",
            split_name,
            skipped,
            len(annotations),
        )

    logger.info("Split %s: %d records built", split_name, len(records))
    return records


def records_to_dataset(records: list[dict]) -> datasets.Dataset:
    """Convert a list of flat records into a HF Dataset."""
    features = datasets.Features(
        {
            "question_id": datasets.Value("int64"),
            "image_id": datasets.Value("int64"),
            "image": datasets.Image(),
            "question_en": datasets.Value("string"),
            "question_zh": datasets.Value("string"),
            "context_en": datasets.Value("string"),
            "context_zh": datasets.Value("string"),
            "answers": [
                {
                    "answer_type": datasets.Value("string"),
                    "answer_en": datasets.Value("string"),
                    "answer_zh": datasets.Value("string"),
                }
            ],
        }
    )

    # Separate columns for Dataset.from_dict
    columns: dict[str, list] = {k: [] for k in features}
    for r in records:
        for k in features:
            columns[k].append(r[k])

    return datasets.Dataset.from_dict(columns, features=features)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CalfKing/vtqa2023 to standard HF dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["local", "hub"],
        help="'local' to save to disk, 'hub' to push to HF Hub",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the converted dataset (required for --mode local)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HF Hub repo to push to (required for --mode hub)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hub repo private (only used with --mode hub)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for downloads (default: output_dir/.cache or ./.cache)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N samples per split",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_FILES.keys()),
        choices=list(SPLIT_FILES.keys()),
        help="Which splits to convert (default: all)",
    )
    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.mode == "local" and not args.output_dir:
        parser.error("--output_dir is required when --mode is 'local'")
    if args.mode == "hub" and not args.repo_id:
        parser.error("--repo_id is required when --mode is 'hub'")

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(
        args.output_dir if args.output_dir else "."
    ) / ".cache"

    # Step 1: Download and extract raw data
    logger.info("Step 1: Downloading and extracting raw data...")
    extract_dir = download_and_extract(cache_dir)

    # Step 2: Find the image directory
    image_base_dir = find_image_dir(extract_dir)
    logger.info("Image base directory: %s", image_base_dir)

    # Step 3: Build each split
    split_datasets: dict[str, datasets.Dataset] = {}
    labeled_splits = {"train", "validation"}

    for split_name in args.splits:
        logger.info("Step 3: Processing split '%s'...", split_name)
        annotations = load_annotations(extract_dir, split_name)
        if args.limit is not None:
            annotations = annotations[: args.limit]
            logger.info("Limited to first %d annotations", len(annotations))
        labeled = split_name in labeled_splits
        records = build_split(annotations, split_name, image_base_dir, labeled=labeled)
        split_datasets[split_name] = records_to_dataset(records)

    # Step 4: Output
    dataset_dict = datasets.DatasetDict(split_datasets)
    logger.info("Dataset summary:\n%s", dataset_dict)

    if args.mode == "local":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Step 4: Saving to %s ...", output_dir)
        dataset_dict.save_to_disk(str(output_dir))
        logger.info("Saved dataset to %s", output_dir)
    elif args.mode == "hub":
        logger.info("Step 4: Pushing to Hub as '%s' ...", args.repo_id)
        dataset_dict.push_to_hub(args.repo_id, private=args.private)
        logger.info("Pushed to https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
