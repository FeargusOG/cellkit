"""Example of customizing ``AnnDataDataset`` output with ``transform``."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cellkit.data.dataset import AnnDataDataset
from cellkit.data.reader import make_h5ad_reader_factory


def add_log_total_counts(sample: dict[str, Any]) -> dict[str, Any]:
    """Add derived fields to one dataset sample.

    This runs after ``AnnDataDataset`` has read the sample from disk, so it is a
    good place to add task-specific keys without subclassing the dataset.
    """
    x = sample["x"].to(dtype=torch.float32)
    sample["x"] = x
    sample["log_total_counts"] = torch.log1p(x.sum())
    sample["is_high_count"] = sample["log_total_counts"] > 4.0
    return sample


def build_dataset(data_path: str | Path) -> AnnDataDataset:
    """Create an example dataset with a sample-level transform."""
    return AnnDataDataset(
        reader_factory=make_h5ad_reader_factory(data_path),
        obs_columns=["treatment_status", "study_name"],
        target_column="cell_subtype_3",
        transform=add_log_total_counts,
        return_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Example of using AnnDataDataset with a transform."
    )
    parser.add_argument("data_path", help="Path to an .h5ad dataset")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Example dataset not found: {data_path}. "
            "Pass the path to a real .h5ad file."
        )

    dataset = build_dataset(data_path)
    num_examples = min(3, len(dataset))
    print(f"Showing {num_examples} sample(s) from {data_path}")

    for sample_index in range(num_examples):
        sample = dataset[sample_index]
        print(f"\nSample {sample_index}")
        print(f"  keys: {sorted(sample.keys())}")
        print(f"  x shape: {tuple(sample['x'].shape)}")
        print(f"  target: {sample['target']}")
        print(f"  obs: {sample['obs']}")
        print(f"  index: {sample['index']}")
        print(f"  log_total_counts: {sample['log_total_counts']}")
        print(f"  is_high_count: {sample['is_high_count']}")


if __name__ == "__main__":
    main()
