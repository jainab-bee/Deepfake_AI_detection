"""Utility helpers for logging, paths, and misc support."""

from pathlib import Path


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
