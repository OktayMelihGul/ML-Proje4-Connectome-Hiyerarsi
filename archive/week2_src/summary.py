"""Headline counts per dataset, next to published reference figures."""
from __future__ import annotations

import pandas as pd

from .config import PUBLISHED


def summarize(neurons: pd.DataFrame, edges: pd.DataFrame) -> dict:
    ds = str(neurons["dataset"].iloc[0])
    ref = PUBLISHED.get(ds, {})
    return {
        "dataset": ds,
        "release": neurons["release"].iloc[0],
        "min_syn": int(edges["min_syn_fetched"].iloc[0]),
        "neurons": len(neurons),
        "published_neurons": ref.get("neurons"),
        "published_source": ref.get("source"),
        "edges": len(edges),
        "total_synapses": int(edges["syn_count"].sum()),
        "distinct_cell_types": int(neurons["cell_type"].nunique()),
        "typed_neurons": int(neurons["cell_type"].notna().sum()),
        "neurons_with_edges": int(pd.concat([edges["pre_id"], edges["post_id"]]).nunique()),
    }


def summary_table(pairs) -> pd.DataFrame:
    return pd.DataFrame([summarize(n, e) for n, e in pairs])
