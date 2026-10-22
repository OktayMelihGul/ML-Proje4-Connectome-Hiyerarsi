"""Build, validate and summarise both datasets; record cold/warm timings.

    python -m src.run_pipeline [--cold]

--cold deletes the parquet cache first (raw downloads are kept; their download
times are in data/raw/download_log.json).
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from . import cache, config as C, fetch, summary, validate

log = logging.getLogger("pipeline")


def timed(label, fn, rows):
    t0 = time.perf_counter()
    df = fn()
    rows.append({"step": label, "seconds": round(time.perf_counter() - t0, 1), "cache_hit": cache.LAST["hit"],
                 "rows": len(df), "parquet_MB": round(cache.LAST["bytes"] / 1e6, 1)})
    log.info("%s", rows[-1])
    return df


def main(cold: bool = False):
    C.REPORTS_DIR.mkdir(exist_ok=True)
    if cold:
        for p in C.CACHE_DIR.glob("*.parquet"):
            p.unlink()
    rows, tables = [], {}
    for ds, fn_n, fn_e in (("fafb", fetch.fetch_fafb_neurons, fetch.fetch_fafb_edges),
                           ("malecns", fetch.fetch_malecns_neurons, fetch.fetch_malecns_edges)):
        tables[(ds, "neurons")] = timed(f"{ds} neurons", fn_n, rows)
        for m in C.MIN_SYN_CACHED:
            tables[(ds, m)] = timed(f"{ds} edges min{m}", lambda: fn_e(m), rows)
    for ds in ("fafb", "malecns"):           # second call must be served from cache
        for m in C.MIN_SYN_CACHED:
            timed(f"{ds} edges min{m} (2nd call)", lambda: (fetch.fetch_fafb_edges if ds == "fafb" else fetch.fetch_malecns_edges)(m), rows)
    timings = pd.DataFrame(rows)
    reports, summ = [], []
    for ds in ("fafb", "malecns"):
        for m in C.MIN_SYN_CACHED:
            r = validate.validate(tables[(ds, "neurons")], tables[(ds, m)]).assign(dataset=ds, min_syn=m)
            reports.append(r)
            summ.append(summary.summarize(tables[(ds, "neurons")], tables[(ds, m)]))
    validate.assert_interchangeable(tables[("fafb", "neurons")], tables[("malecns", "neurons")])
    validate.assert_interchangeable(tables[("fafb", 1)], tables[("malecns", 1)])
    rep = pd.concat(reports, ignore_index=True)
    timings.to_csv(C.REPORTS_DIR / "timings.csv", index=False)
    rep.to_csv(C.REPORTS_DIR / "validation.csv", index=False)
    pd.DataFrame(summ).to_csv(C.REPORTS_DIR / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120, "display.max_columns", 30):
        print(timings.to_string(index=False)); print(rep.to_string(index=False)); print(pd.DataFrame(summ).T.to_string())
    print("assert_interchangeable: OK (neurons and edges)")
    return tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true")
    main(ap.parse_args().cold)
