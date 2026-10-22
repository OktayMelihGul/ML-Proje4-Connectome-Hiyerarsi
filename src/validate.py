"""The eight checks of the Data Conventions ("Validation") and assert_interchangeable.

Every check raises naming the column. `validate_*` return the list of checks run,
so the pipeline can write a report of what passed.
"""
from __future__ import annotations

import pandas as pd

import config

VOCAB = {
    "dataset": config.DATASET_VALUES,
    "side": config.SIDE_VALUES,
    "nt_type": config.NT_VALUES,
    "super_class": config.SUPER_CLASS_VALUES,
}
NEURON_KEY = ["neuron_id", "dataset"]
EDGE_KEY = ["pre_id", "post_id", "dataset"]
EDGEROI_KEY = ["pre_id", "post_id", "neuropil", "dataset"]


class ValidationError(AssertionError):
    pass


def _fail(check: str, column: str, detail: str):
    raise ValidationError(f"[{check}] {column}: {detail}")


def check_columns(df: pd.DataFrame, schema: dict) -> None:
    missing = [c for c in schema if c not in df.columns]
    extra = [c for c in df.columns if c not in schema]
    if missing:
        _fail("columns", missing[0], f"missing (all missing: {missing})")
    if extra:
        _fail("columns", extra[0], f"unexpected (all unexpected: {extra})")
    if list(df.columns) != list(schema):
        _fail("columns", "*", f"order differs from config.py: {list(df.columns)}")


def check_dtypes(df: pd.DataFrame, schema: dict) -> None:
    for col, (dtype, _) in schema.items():
        want = dtype if isinstance(dtype, pd.CategoricalDtype) else pd.api.types.pandas_dtype(dtype)
        if df[col].dtype != want:
            _fail("dtypes", col, f"{df[col].dtype} != {want}")


def check_nulls(df: pd.DataFrame, schema: dict) -> None:
    for col, (_, nullable) in schema.items():
        n = int(df[col].isna().sum())
        if not nullable and n:
            _fail("nulls", col, f"{n} missing values in a non-nullable column")


def check_keys(df: pd.DataFrame, key: list[str]) -> None:
    n = int(df.duplicated(key).sum())
    if n:
        _fail("keys", "+".join(key), f"{n} repeated keys")


def check_references(edges: pd.DataFrame, neurons: pd.DataFrame) -> None:
    for ds in edges["dataset"].unique():
        ids = neurons.loc[neurons["dataset"] == ds, "neuron_id"]
        e = edges[edges["dataset"] == ds]
        for col in ("pre_id", "post_id"):
            n = int((~e[col].isin(ids)).sum())
            if n:
                _fail("references", col, f"{n} rows ({ds}) with no neuron in the same dataset")


def check_vocabulary(df: pd.DataFrame) -> None:
    for col, values in VOCAB.items():
        if col in df.columns:
            bad = set(df[col].dropna().astype(str)) - set(values)
            if bad:
                _fail("vocabulary", col, f"values outside the list: {sorted(bad)[:10]}")


def check_threshold(edges: pd.DataFrame) -> None:
    n = int((edges["syn_count"] < edges["min_syn_fetched"]).sum())
    if n:
        _fail("threshold", "syn_count", f"{n} rows below min_syn_fetched")


def check_plausibility(edges: pd.DataFrame, neurons: pd.DataFrame) -> int:
    """Input synapses in the edge table must not exceed n_post (skipped where n_post is missing).
    Returns the number of neurons actually checked."""
    n_post = neurons.set_index("neuron_id")["n_post"]
    n_post = n_post[n_post.notna()]
    if n_post.empty:
        return 0
    inputs = edges.groupby("post_id")["syn_count"].sum()
    inputs = inputs[inputs.index.isin(n_post.index)]
    over = inputs > n_post.reindex(inputs.index).astype("int64")
    if over.any():
        _fail("plausibility", "n_post", f"{int(over.sum())} neurons receive more synapses "
              f"in the edge table than n_post (e.g. {over[over].index[:3].tolist()})")
    return int(len(inputs))


def validate_neurons(neurons: pd.DataFrame) -> list[str]:
    s = config.NEURON_SCHEMA
    check_columns(neurons, s); check_dtypes(neurons, s); check_nulls(neurons, s)
    check_keys(neurons, NEURON_KEY); check_vocabulary(neurons)
    return ["columns", "dtypes", "nulls", "keys", "vocabulary"]


def validate_edges(edges: pd.DataFrame, neurons: pd.DataFrame, roi: bool = False) -> list[str]:
    s = config.EDGEROI_SCHEMA if roi else config.EDGE_SCHEMA
    check_columns(edges, s); check_dtypes(edges, s); check_nulls(edges, s)
    check_keys(edges, EDGEROI_KEY if roi else EDGE_KEY)
    check_references(edges, neurons); check_vocabulary(edges); check_threshold(edges)
    checks = ["columns", "dtypes", "nulls", "keys", "references", "vocabulary", "threshold"]
    if not roi:                          # per-neuropil rows are partial sums; check the summed table
        n = check_plausibility(edges, neurons)
        checks.append(f"plausibility ({n:,} neurons checked)" if n else "plausibility (skipped: n_post missing)")
    return checks


def assert_interchangeable(a, b):
    assert list(a.columns) == list(b.columns), "column sets differ"
    assert a.dtypes.equals(b.dtypes), "dtypes differ"
