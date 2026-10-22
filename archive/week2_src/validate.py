"""Schema validation: the 8 checks of the Data Schema Specification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SCHEMA


def _dtype_ok(s: pd.Series, want: str) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype) if want == "category" else str(s.dtype) == want


def validate(neurons: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Run all checks; return one row per (table, check) with pass flag and detail."""
    rows = []

    def add(table, check, ok, detail=""):
        rows.append({"table": table, "check": check, "passed": bool(ok), "detail": detail})

    for table, df in (("neurons", neurons), ("edges", edges)):
        spec = SCHEMA[table]
        # 1 columns
        add(table, "1 columns", list(df.columns) == list(spec),
            f"missing={sorted(set(spec) - set(df.columns))} extra={sorted(set(df.columns) - set(spec))}")
        # 2 dtypes
        bad = [c for c, s in spec.items() if c in df and not _dtype_ok(df[c], s["dtype"])]
        add(table, "2 dtypes", not bad, f"bad={bad}")
        # 3 nullability
        nulls = {c: int(df[c].isna().sum()) for c, s in spec.items()
                 if c in df and not s["nullable"] and df[c].isna().any()}
        add(table, "3 nullability", not nulls, f"nulls in non-nullable={nulls}")
        # 6 vocabulary
        bad_vocab = {}
        for c, s in spec.items():
            if "vocab" in s and c in df:
                extra = set(df[c].dropna().astype(str).unique()) - set(s["vocab"])
                if extra or list(df[c].cat.categories) != list(s["vocab"]):
                    bad_vocab[c] = sorted(extra)
        add(table, "6 vocabulary", not bad_vocab, f"{bad_vocab}")

    # 4 key uniqueness
    dup_n = int(neurons["neuron_id"].duplicated().sum())
    add("neurons", "4 key uniqueness", dup_n == 0, f"duplicate neuron_id={dup_n}")
    dup_e = int(edges.duplicated(["pre_id", "post_id"]).sum())
    add("edges", "4 key uniqueness", dup_e == 0, f"duplicate (pre_id, post_id)={dup_e}")

    # 5 referential integrity
    ids = neurons["neuron_id"].to_numpy()
    miss_pre = int((~edges["pre_id"].isin(ids)).sum())
    miss_post = int((~edges["post_id"].isin(ids)).sum())
    add("edges", "5 referential integrity", miss_pre == 0 and miss_post == 0,
        f"pre_id not in neurons={miss_pre}, post_id not in neurons={miss_post}")
    same = (set(edges["dataset"].astype(str).unique()) <= set(neurons["dataset"].astype(str).unique())
            and set(edges["release"].unique()) <= set(neurons["release"].unique()))
    add("edges", "5b dataset/release match", same,
        f"edges={sorted(edges['release'].unique())} neurons={sorted(neurons['release'].unique())}")

    # 7 threshold consistency
    m = edges["min_syn_fetched"].unique()
    below = int((edges["syn_count"] < edges["min_syn_fetched"]).sum())
    add("edges", "7 threshold consistency", len(m) == 1 and below == 0 and (edges["syn_count"] > 0).all(),
        f"min_syn_fetched={list(m)}, rows below threshold={below}, min syn_count={edges['syn_count'].min()}")

    # 8 plausibility: synapses in edges cannot exceed the neuron's own synapse counts
    out_w = edges.groupby("pre_id")["syn_count"].sum()
    in_w = edges.groupby("post_id")["syn_count"].sum()
    n = neurons.set_index("neuron_id")
    over_out = int((out_w > n["n_pre"].reindex(out_w.index)).sum())
    over_in = int((in_w > n["n_post"].reindex(in_w.index)).sum())
    out_deg = edges.groupby("pre_id").size()
    over_deg = int((out_deg > n["n_pre"].reindex(out_deg.index)).sum())
    add("edges", "8 plausibility", over_out == 0 and over_in == 0 and over_deg == 0,
        f"out-synapses>n_pre: {over_out}; in-synapses>n_post: {over_in}; out-degree>n_pre: {over_deg}")
    return pd.DataFrame(rows)


def assert_interchangeable(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Raise unless two tables (e.g. FAFB vs MaleCNS neurons) can be concatenated as-is."""
    if list(a.columns) != list(b.columns):
        raise AssertionError(f"column order differs: {list(a.columns)} vs {list(b.columns)}")
    for c in a.columns:
        da, db = a[c].dtype, b[c].dtype
        if isinstance(da, pd.CategoricalDtype) != isinstance(db, pd.CategoricalDtype):
            raise AssertionError(f"{c}: {da} vs {db}")
        if not isinstance(da, pd.CategoricalDtype) and da != db:
            raise AssertionError(f"{c}: {da} vs {db}")
        if isinstance(da, pd.CategoricalDtype):
            # fixed-vocabulary columns must share categories exactly; open ones (region) may differ
            for table in SCHEMA.values():
                if c in table and "vocab" in table[c] and list(da.categories) != list(db.categories):
                    raise AssertionError(f"{c}: vocabularies differ")


def raise_if_failed(report: pd.DataFrame, allow: tuple[str, ...] = ()) -> None:
    failed = report[~report["passed"] & ~report["check"].isin(allow)]
    if len(failed):
        raise AssertionError("validation failed:\n" + failed.to_string(index=False))
