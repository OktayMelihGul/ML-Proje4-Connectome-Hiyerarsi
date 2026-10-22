"""Shared helpers for the week 3 EDA (v0.1-eda).

Temporary: these cover what the EDA needs before the week 3 session.
The real features.py is written after the session, once D16 (zero-input
behaviour) and D19 (feature column list) are decided.

Every function works on ONE dataset at a time and on tables built to the
Week 2 Data Conventions (neurons, edges; edges cached at 1 synapse).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

NEUROPIL_TABLE = Path(__file__).with_name("neuropils_provisional.csv")

# Side markers used by FlyWire (ME_L) and neuPrint (ME(L)).
_SIDE = re.compile(r"(_[LR]|\([LR]\))$")

# Non-FlyWire spellings mapped to the FlyWire base name.
# Hemibrain-style names: confirm against the malecns region values printed
# in your README. Report any missing name to the group; do not add aliases
# privately, or the five repositories stop agreeing.
ALIASES = {
    "CA": "MB_CA",
    "PED": "MB_PED",
    "aL": "MB_VL", "a'L": "MB_VL",
    "bL": "MB_ML", "b'L": "MB_ML", "gL": "MB_ML",
    "ROB": "CRE", "RUB": "CRE",
    "AB": "FB",
}

# Values that mean "no neuropil": region_group becomes unknown.
NO_NEUROPIL = {"unknown", "UNASGD"}


def load_neuropil_table(path: str | Path = NEUROPIL_TABLE) -> pd.DataFrame:
    """The provisional neuropil table, indexed by FlyWire base name."""
    table = pd.read_csv(path, dtype={"neuropil": "string", "region_group": "string"})
    table["is_optic"] = table["is_optic"].astype(bool)
    if table["neuropil"].duplicated().any():
        raise ValueError("neuropil table has duplicate rows")
    return table.set_index("neuropil")


def canonical_neuropil(region: pd.Series, table: pd.DataFrame,
                       nerve_cord: set[str] | frozenset[str] = frozenset()) -> pd.Series:
    """Map region values to FlyWire base names.

    Strips the side marker, applies ALIASES, and returns:
      - the base name for brain neuropils,
      - "unknown" for unknown / UNASGD,
      - "nerve_cord" for values listed in `nerve_cord` (malecns only).
    Any other value raises, naming it, so a spelling gap is never silent.
    """
    raw = region.astype("string")
    base = raw.str.replace(_SIDE, "", regex=True).replace(ALIASES)
    out = base.copy()
    out[raw.isin(NO_NEUROPIL) | base.isin(NO_NEUROPIL)] = "unknown"
    out[raw.isin(nerve_cord) | base.isin(nerve_cord)] = "nerve_cord"
    ok = out.isin(table.index) | out.isin(["unknown", "nerve_cord"])
    if not ok.all():
        bad = sorted(raw[~ok].dropna().unique())
        raise ValueError(f"region: unmapped values {bad[:20]} "
                         f"({len(bad)} in total). Add them to ALIASES via the group, "
                         f"or pass nerve-cord neuropils in `nerve_cord`.")
    return out


def add_provisional_flags(neurons: pd.DataFrame, table: pd.DataFrame | None = None,
                          nerve_cord: set[str] | frozenset[str] = frozenset()) -> pd.DataFrame:
    """Return a copy of `neurons` with the provisional EDA flags added.

    neuropil      FlyWire base name, "unknown" or "nerve_cord"
    is_optic      True if the neuropil is in the optic lobe (False if unknown)
    region_group  sensory_periphery / motor_periphery / higher_order / unknown
    in_brain      True for brain neuropils; use it to keep malecns comparable to fafb
    """
    table = load_neuropil_table() if table is None else table
    out = neurons.copy()
    out["neuropil"] = canonical_neuropil(out["region"], table, nerve_cord)
    known = out["neuropil"].isin(table.index)
    out["is_optic"] = out["neuropil"].map(table["is_optic"]).fillna(False).astype(bool)
    out["region_group"] = (out["neuropil"].map(table["region_group"])
                           .fillna("unknown").astype("string"))
    out["in_brain"] = known
    return out


def _single_dataset(df: pd.DataFrame, name: str) -> None:
    if "dataset" in df and df["dataset"].nunique() > 1:
        raise ValueError(f"{name}: pass one dataset at a time")


def basic_features(neurons: pd.DataFrame, edges: pd.DataFrame, tau: int = 10) -> pd.DataFrame:
    """The four features the EDA needs, one row per neuron in `neurons`.

    in_degree, out_degree   number of partners over edges with syn_count >= tau
    total_in_syn,           synapses summed over ALL cached edges (tau = 1),
    total_out_syn           computed once and never after thresholding

    `edges` must be the full cached table (min_syn_fetched == 1), so that the
    totals are the fixed denominators from the EDA Plan's Definitions.
    Neurons without edges get 0. D16 (zero vs null) is still open: this
    choice is for the EDA only.
    """
    _single_dataset(neurons, "neurons")
    _single_dataset(edges, "edges")
    if "min_syn_fetched" in edges and (edges["min_syn_fetched"] != 1).any():
        raise ValueError("edges: pass the table cached at 1 synapse")
    ids = pd.Index(neurons["neuron_id"], name="neuron_id")
    kept = edges[edges["syn_count"] >= tau]
    feats = pd.DataFrame(index=ids)
    feats["in_degree"] = kept.groupby("post_id").size()
    feats["out_degree"] = kept.groupby("pre_id").size()
    feats["total_in_syn"] = edges.groupby("post_id")["syn_count"].sum()
    feats["total_out_syn"] = edges.groupby("pre_id")["syn_count"].sum()
    return feats.fillna(0).astype("int64").reset_index()


def relative_weight(edges: pd.DataFrame) -> pd.Series:
    """syn_count / total_in_syn(post), with the denominator taken from all
    cached edges (tau = 1). Use for the relative thresholds 0.009 and 0.026.
    Filter edges AFTER calling this, never before."""
    _single_dataset(edges, "edges")
    total_in = edges.groupby("post_id")["syn_count"].transform("sum")
    return edges["syn_count"] / total_in
