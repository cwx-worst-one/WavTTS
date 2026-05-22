"""WavTTS package."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _set_default_cache_dirs() -> None:
    cache_root = Path(tempfile.gettempdir()) / "wavtts_cache"
    for env_name, subdir in {
        "NUMBA_CACHE_DIR": "numba",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        cache_dir = Path(os.environ.get(env_name, cache_root / subdir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(cache_dir))


_set_default_cache_dirs()
