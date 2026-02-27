"""Load a locally saved VTQA dataset and push it to HuggingFace Hub.

Usage:
    python scripts/dev/hf_datasets/vtqa_to_hub.py --input_dir ./vtqa_standard --repo_id myuser/vtqa2023
    python scripts/dev/hf_datasets/vtqa_to_hub.py --input_dir ./vtqa_standard --repo_id myuser/vtqa2023 --private
"""

from __future__ import annotations

import argparse

from datasets import load_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a locally saved VTQA dataset to HuggingFace Hub")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the dataset saved by vtqa.py",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HF Hub repo to push to (e.g., 'myuser/vtqa2023-standard')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hub repo private",
    )
    args = parser.parse_args()

    ds = load_from_disk(args.input_dir)
    print(ds)

    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
