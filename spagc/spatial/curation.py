"""Bind curated mixture annotations to verified model artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def artifact_tree_sha256(directory: Path) -> str:
    """Bind curated component names to exact saved-model bytes and relative names."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Supply the released fitted model: {directory}")
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Empty model directory: {directory}")
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def assign_curated_names(
    codes: np.ndarray, mapping: dict[str, str], actual_hash: str, expected_hash: str
) -> pd.Categorical:
    """Never apply historical biological names to an unverified/refitted mixture."""
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            "Curated names do not match these model bytes. Re-curate/refit-align the components first."
        )
    codes = np.asarray(codes).astype(str)
    missing = set(codes).difference(mapping)
    if missing:
        raise ValueError(f"Uncurated mixture components: {sorted(missing)}")
    return pd.Categorical([mapping[value] for value in codes])
