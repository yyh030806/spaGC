"""Input checks and protection of existing outputs."""

from __future__ import annotations

from pathlib import Path


def require_files(*paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Required inputs are missing: " + ", ".join(missing))


def require_new(path):
    """Protect processed inputs and previous scientific runs from overwriting."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}. Choose a new RUN_ID or output path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
