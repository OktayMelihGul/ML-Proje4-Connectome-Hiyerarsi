"""The one fetch helper (Data Conventions, "Caching"): read if present, else fetch, validate, write."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from .sources import ROOT

log = logging.getLogger(__name__)
LAST: dict = {}          # last call: path and whether it was a cache hit (for the timing log)


def resolve(path: Path) -> Path:
    """config.cache_path() is relative to the repository root; anchor it there."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def cached(path: Path, fetch_fn: Callable[[], pd.DataFrame],
           validate: Callable[[pd.DataFrame], object] | None = None,
           force_refresh: bool = False) -> pd.DataFrame:
    path = resolve(path)
    if path.exists() and not force_refresh:
        LAST.update(path=path.name, hit=True)
        return pd.read_parquet(path)
    df = fetch_fn()
    if validate is not None:
        validate(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)                      # never leave a half-written cache file
    LAST.update(path=path.name, hit=False)
    log.info("built %s (%d rows)", path.name, len(df))
    return df
