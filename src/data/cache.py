"""Local caching layer for intermediate results."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def cache_key(params: dict) -> str:
    """Generate deterministic cache key from parameters."""
    serialized = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def load_cached(path: Path) -> pd.DataFrame | None:
    """Load cached DataFrame if it exists."""
    if path.exists():
        logger.debug(f"Cache hit: {path}")
        return pd.read_parquet(path)
    return None


def save_cache(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.debug(f"Cached: {path}")
