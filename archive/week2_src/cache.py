"""Parquet cache: fetch once, read from disk afterwards."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import CACHE_DIR

log = logging.getLogger(__name__)

# Last call's provenance, read by run_pipeline for the timing table.
LAST = {}


def cache_path(name: str) -> Path:
    """e.g. cache_path('fafb_edges_783_min1') -> data/cache/fafb_edges_783_min1.parquet"""
    return CACHE_DIR / f"{name}.parquet"


def cached(path: Path | str, fetch_fn: Callable[[], pd.DataFrame],
           force_refresh: bool = False) -> pd.DataFrame:
    """Return the table at ``path``; build it with ``fetch_fn`` if absent or forced."""
    path = Path(path)
    if not path.suffix:
        path = cache_path(str(path))
    t0 = time.perf_counter()
    if path.exists() and not force_refresh:
        df = pd.read_parquet(path)
        LAST.update(path=str(path), hit=True, seconds=time.perf_counter() - t0,
                    rows=len(df), bytes=path.stat().st_size)
        log.info("cache hit %s (%d rows, %.1fs)", path.name, len(df), LAST["seconds"])
        return df
    df = fetch_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)                        # atomic: never leave a half-written cache
    LAST.update(path=str(path), hit=False, seconds=time.perf_counter() - t0,
                rows=len(df), bytes=path.stat().st_size)
    log.info("cache miss %s: built %d rows in %.1fs", path.name, len(df), LAST["seconds"])
    return df
