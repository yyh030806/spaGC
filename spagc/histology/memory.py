"""Host-memory budgets for whole-slide data processing."""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np


def format_bytes(n: int | float | None) -> str:
    if n is None:
        return "unknown"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


def _host_mem_available_bytes() -> int | None:
    """Linux MemAvailable; fallback to sysconf on other Unix systems."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size) * int(avail_pages)
    except (AttributeError, ValueError, OSError):
        return None


def _cgroup_mem_available_bytes() -> int | None:
    """Return remaining memory in the current cgroup (v2/v1), when bounded."""
    try:
        max_path = "/sys/fs/cgroup/memory.max"
        cur_path = "/sys/fs/cgroup/memory.current"
        if os.path.exists(max_path) and os.path.exists(cur_path):
            max_s = Path(max_path).read_text(encoding="utf-8").strip()
            cur_s = Path(cur_path).read_text(encoding="utf-8").strip()
            if max_s != "max":
                limit = int(max_s)
                current = int(cur_s)
                return max(0, limit - current)
    except (OSError, ValueError):
        pass
    try:
        limit_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        usage_path = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        if os.path.exists(limit_path) and os.path.exists(usage_path):
            limit = int(Path(limit_path).read_text(encoding="utf-8").strip())
            usage = int(Path(usage_path).read_text(encoding="utf-8").strip())
            if limit < 1 << 60:
                return max(0, limit - usage)
    except (OSError, ValueError):
        pass
    return None


def get_effective_available_memory_bytes() -> int | None:
    """Use the stricter of host MemAvailable and cgroup remaining memory."""
    values = [
        x
        for x in (_host_mem_available_bytes(), _cgroup_mem_available_bytes())
        if x is not None
    ]
    return min(values) if values else None


def _dataset_nbytes(ds: h5py.Dataset) -> int:
    return int(np.prod(ds.shape, dtype=np.int64)) * np.dtype(ds.dtype).itemsize
