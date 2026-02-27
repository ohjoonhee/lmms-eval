"""Convert pix2fact/Pix2FactBenchmark into a standard HuggingFace dataset.

The original HF repo stores images as loose files alongside a CSV metadata
file (``Pix2Fact_1k.csv``).  Because the CSV is *not* named ``metadata.csv``
and lacks a ``file_name`` column, the HF imagefolder builder fails to link
images with their annotations — the dataset viewer shows images only.

This script downloads the CSV + images, merges them, and produces a proper
Parquet-backed dataset with embedded images that works out of the box with
``datasets.load_dataset``.

Target schema
-------------
    item_id             : str      – unique item identifier
    image               : Image    – PIL image
    image_category      : str      – e.g. "Amusement Parks"
    image_description   : str      – free-text description
    visual_perception_type : str   – e.g. "OCR"
    knowledge_domain    : str      – e.g. "Finance & Economics"
    reasoning_logic_type : str     – e.g. "Calculation-based Query"
    resolution          : str      – original resolution string
    question            : str      – the visual QA question
    answer              : str      – ground-truth answer
    bounding_box        : str      – bounding-box annotation
    evidence_1          : str      – supporting evidence #1
    evidence_2          : str      – supporting evidence #2
    evidence_3          : str      – supporting evidence #3
    evidence_url_1      : str      – URL for evidence #1
    evidence_url_2      : str      – URL for evidence #2
    evidence_url_3      : str      – URL for evidence #3

Usage
-----
    # Save to disk
    python scripts/dev/hf_datasets/pix2fact.py --save-to-disk ./pix2fact_dataset

    # Push to HuggingFace Hub
    python scripts/dev/hf_datasets/pix2fact.py --push-to-hub myuser/Pix2Fact

    # Both at once
    python scripts/dev/hf_datasets/pix2fact.py \\
        --save-to-disk ./pix2fact_dataset \\
        --push-to-hub myuser/Pix2Fact

    # Smoke test with a few samples
    python scripts/dev/hf_datasets/pix2fact.py --save-to-disk ./out --limit 10
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ID = "pix2fact/Pix2FactBenchmark"
CSV_FILENAME = "Pix2Fact_1k.csv"

# Column mapping: original CSV name → clean output name
COLUMN_MAP: dict[str, str] = {
    "ItemID": "item_id",
    "local_image_path": "_image_path",  # internal, not in final schema
    "image_category": "image_category",
    "image_description": "image_description",
    "visual_perception_type": "visual_perception_type",
    "knowledge_domain": "knowledge_domain",
    "reasoning_logic_type": "reasoning_logic_type",
    "Resolution": "resolution",
    "[Final]question": "question",
    "[Final]answer": "answer",
    "bounding_box": "bounding_box",
    "[Final]evidence_1": "evidence_1",
    "[Final]evidence_2": "evidence_2",
    "[Final]evidence_3": "evidence_3",
    "[Final]evidence_url_1": "evidence_url_1",
    "[Final]evidence_url_2": "evidence_url_2",
    "[Final]evidence_url_3": "evidence_url_3",
}

FEATURES = datasets.Features(
    {
        "item_id": datasets.Value("string"),
        "image": datasets.Image(),
        "image_category": datasets.Value("string"),
        "image_description": datasets.Value("string"),
        "visual_perception_type": datasets.Value("string"),
        "knowledge_domain": datasets.Value("string"),
        "reasoning_logic_type": datasets.Value("string"),
        "resolution": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "bounding_box": datasets.Value("string"),
        "evidence_1": datasets.Value("string"),
        "evidence_2": datasets.Value("string"),
        "evidence_3": datasets.Value("string"),
        "evidence_url_1": datasets.Value("string"),
        "evidence_url_2": datasets.Value("string"),
        "evidence_url_3": datasets.Value("string"),
    }
)


def download_csv(cache_dir: Path) -> Path:
    """Download the metadata CSV from the HF repo."""
    logger.info("Downloading %s from %s ...", CSV_FILENAME, REPO_ID)
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=CSV_FILENAME,
            repo_type="dataset",
            cache_dir=str(cache_dir / "hf_cache"),
        )
    )


def download_image(image_filename: str, cache_dir: Path) -> str | None:
    """Download a single image from the HF repo. Returns the local path."""
    try:
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=image_filename,
            repo_type="dataset",
            cache_dir=str(cache_dir / "hf_cache"),
        )
    except Exception as e:
        logger.warning("Failed to download image %s: %s", image_filename, e)
        return None


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read the CSV and return rows as dicts with original column names."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_records(rows: list[dict[str, str]], cache_dir: Path) -> list[dict]:
    """Convert raw CSV rows into dataset records with downloaded images."""
    records: list[dict] = []
    skipped = 0

    for i, row in enumerate(rows):
        image_filename = row.get("local_image_path", "").strip()
        if not image_filename:
            skipped += 1
            logger.warning("Row %d: missing local_image_path, skipping", i)
            continue

        image_path = download_image(image_filename, cache_dir)
        if image_path is None:
            skipped += 1
            continue

        record: dict = {}
        for csv_col, out_col in COLUMN_MAP.items():
            if out_col == "_image_path":
                continue  # handled separately
            record[out_col] = row.get(csv_col, "")

        record["image"] = image_path
        records.append(record)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d rows ...", i + 1, len(rows))

    if skipped:
        logger.warning("Skipped %d / %d rows due to missing images", skipped, len(rows))

    logger.info("Built %d records total", len(records))
    return records


def records_to_dataset(records: list[dict]) -> datasets.Dataset:
    """Convert a list of flat records into a HF Dataset."""
    columns: dict[str, list] = {k: [] for k in FEATURES}
    for r in records:
        for k in FEATURES:
            columns[k].append(r[k])

    return datasets.Dataset.from_dict(columns, features=FEATURES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pix2fact/Pix2FactBenchmark to standard HF dataset"
    )
    parser.add_argument(
        "--save-to-disk",
        type=str,
        default=None,
        metavar="DIR",
        help="Save the converted dataset to this directory",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Push the converted dataset to this HF Hub repo (e.g. myuser/Pix2Fact)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hub repo private (only with --push-to-hub)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads (default: .cache under output dir or cwd)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N samples (useful for smoke testing)",
    )
    args = parser.parse_args()

    if not args.save_to_disk and not args.push_to_hub:
        parser.error("At least one of --save-to-disk or --push-to-hub is required")

    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else Path(args.save_to_disk if args.save_to_disk else ".") / ".cache"
    )

    # Step 1: Download CSV
    logger.info("Step 1: Downloading metadata CSV ...")
    csv_path = download_csv(cache_dir)

    # Step 2: Load CSV rows
    logger.info("Step 2: Loading CSV rows ...")
    rows = load_csv_rows(csv_path)
    logger.info("Loaded %d rows from CSV", len(rows))

    if args.limit is not None:
        rows = rows[: args.limit]
        logger.info("Limited to first %d rows", len(rows))

    # Step 3: Download images and build records
    logger.info("Step 3: Downloading images and building records ...")
    records = build_records(rows, cache_dir)

    # Step 4: Create HF Dataset
    logger.info("Step 4: Creating HF Dataset ...")
    ds = records_to_dataset(records)
    dataset_dict = datasets.DatasetDict({"test": ds})
    logger.info("Dataset summary:\n%s", dataset_dict)

    # Step 5: Output
    if args.save_to_disk:
        output_dir = Path(args.save_to_disk)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Step 5a: Saving to %s ...", output_dir)
        dataset_dict.save_to_disk(str(output_dir))
        logger.info("Saved dataset to %s", output_dir)

    if args.push_to_hub:
        logger.info("Step 5b: Pushing to Hub as '%s' ...", args.push_to_hub)
        dataset_dict.push_to_hub(args.push_to_hub, private=args.private)
        logger.info("Pushed to https://huggingface.co/datasets/%s", args.push_to_hub)


if __name__ == "__main__":
    main()
