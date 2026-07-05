"""Utilities to inspect dataset structure and statistics."""

from pathlib import Path


def analyze_dataset(root: str):
    root_path = Path(root)
    print(f"Scanning dataset at: {root_path}")
    for item in sorted(root_path.iterdir()):
        print(item.name)
