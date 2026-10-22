"""Build the six cached tables of the Data Conventions (neurons, edges, edgeroi x fafb, malecns).

    fafb_neurons_783_ann3.2.0                     fafb_edges / fafb_edgeroi _783_buhmann_min1
    malecns_neurons_v1.0_traced                   malecns_edges / malecns_edgeroi _v1.0_traced_min1_conf0.5

Every table goes through cache.cached(): read the parquet if present, otherwise build,
validate and write. Builders log the counts the Step 0 check needs in `LOG`.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pyarrow.feather as feather

import config
from . import sources as S
from .cache import cached, resolve
from .mappings import (FAFB_NT, FAFB_SIDE, FAFB_SUPER, MALECNS_NT, MALECNS_SIDE,
                       MALECNS_SUPER, recode)
from .validate import validate_edges, validate_neurons

log = logging.getLogger(__name__)
LOG: dict = {}          # build-time facts (self-connections dropped, ties, id reconciliation ...)
LOG_PATH = S.ROOT / "reports" / "build_log.json"

FAFB_RELEASE = str(config.FAFB_MATERIALIZATION)
MALECNS_RELEASE = config.MALECNS_DATASET.split(":")[1]          # "v1.0"
UNSPECIFIED_SYNAPSE = "<unspecified>"                           # MaleCNS: synapse outside every primary ROI


def _record(dataset: str, key: str, value) -> None:
    LOG.setdefault(dataset, {})[key] = value
    old = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else {}
    old.setdefault(dataset, {})[key] = value
    LOG_PATH.parent.mkdir(exist_ok=True)
    LOG_PATH.write_text(json.dumps(old, indent=2, default=str))


# ---------------------------------------------------------------------------
# Paths (names from config.cache_path, pattern D7)
# ---------------------------------------------------------------------------
def path(dataset: str, table: str):
    if dataset == "fafb":
        if table == "neurons":
            return config.cache_path("fafb", "neurons", FAFB_RELEASE, f"ann{config.FAFB_ANNOTATIONS_TAG.lstrip('v')}")
        return config.cache_path("fafb", table, FAFB_RELEASE, config.FAFB_SYNAPSE_TABLE,
                                 min_syn=config.FETCH_MIN_SYN)
    source = config.MALECNS_STATUS.lower()
    if table == "neurons":
        return config.cache_path("malecns", "neurons", MALECNS_RELEASE, source)
    return config.cache_path("malecns", table, MALECNS_RELEASE, source,
                             min_syn=config.FETCH_MIN_SYN, conf=config.MALECNS_SYN_CONFIDENCE)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _cast(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, (dtype, _) in schema.items():
        out[col] = df[col].astype(dtype)
    return out.reset_index(drop=True)


def top_region(counts: pd.DataFrame) -> tuple[pd.Series, int]:
    """counts: neuron_id, neuropil, n (connections in that neuropil, inputs + outputs).

    Region = neuropil with the largest n. Tie rule (open in the group; our default):
    deterministic, alphabetical by neuropil name. Returns (region per neuron, number of ties).
    """
    c = counts[counts["n"] > 0].sort_values(["neuron_id", "n", "neuropil"],
                                            ascending=[True, False, True])
    first = c.drop_duplicates("neuron_id").set_index("neuron_id")
    second = c[c.duplicated("neuron_id")].drop_duplicates("neuron_id").set_index("neuron_id")["n"]
    ties = int((first["n"] == second.reindex(first.index)).sum())
    return first["neuropil"], ties


def _finish_edges(df: pd.DataFrame, dataset: str, release: str, roi: bool) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset
    df["release"] = release
    df["min_syn_fetched"] = config.FETCH_MIN_SYN
    df = df[df["syn_count"] >= config.FETCH_MIN_SYN]
    keys = ["pre_id", "post_id"] + (["neuropil"] if roi else [])
    return _cast(df.sort_values(keys), config.EDGEROI_SCHEMA if roi else config.EDGE_SCHEMA)


def _drop_self(df: pd.DataFrame, dataset: str, table: str) -> pd.DataFrame:
    self_rows = df["pre_id"] == df["post_id"]
    n = int(self_rows.sum())
    _record(dataset, f"self_connections_dropped_{table}", n)
    if table == "edges":
        _record(dataset, "self_connection_synapses_dropped", int(df.loc[self_rows, "syn_count"].sum()))
        _record(dataset, "self_connections_dropped_syn_ge5", int((df.loc[self_rows, "syn_count"] >= 5).sum()))
    log.info("%s %s: dropped %d self-connections", dataset, table, n)
    return df[~self_rows] if config.DROP_SELF_CONNECTIONS else df


# ---------------------------------------------------------------------------
# FAFB
# ---------------------------------------------------------------------------
def raw_fafb_connections() -> pd.DataFrame:
    """Codex per-neuropil table (Buhmann synapses): one row per (pre, post, neuropil)."""
    raw = pd.read_feather(S.fafb_file("connections"),
                          columns=["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count"])
    return raw.rename(columns={"pre_pt_root_id": "pre_id", "post_pt_root_id": "post_id"})


def build_fafb_edgeroi() -> pd.DataFrame:
    raw = _drop_self(raw_fafb_connections(), "fafb", "edgeroi")
    return _finish_edges(raw, "fafb", FAFB_RELEASE, roi=True)


def build_fafb_edges(edgeroi: pd.DataFrame) -> pd.DataFrame:
    """D8: sum synapses over neuropils (thresholds are applied later, in memory)."""
    e = edgeroi.groupby(["pre_id", "post_id"], sort=False, observed=True)["syn_count"].sum().reset_index()
    e = _drop_self(e, "fafb", "edges")
    return _finish_edges(e, "fafb", FAFB_RELEASE, roi=False)


def fafb_region(edgeroi: pd.DataFrame) -> tuple[pd.Series, int]:
    """Neuropil with most synapses from edgeroi: inputs + outputs (connection counts)."""
    out = edgeroi.groupby(["pre_id", "neuropil"], observed=True)["syn_count"].sum()
    inp = edgeroi.groupby(["post_id", "neuropil"], observed=True)["syn_count"].sum()
    out.index.names = inp.index.names = ["neuron_id", "neuropil"]
    n = out.add(inp, fill_value=0).rename("n").reset_index()
    return top_region(n)


def build_fafb_neurons(edgeroi: pd.DataFrame) -> pd.DataFrame:
    ids = pd.DataFrame({"neuron_id": np.load(S.fafb_file("root_ids")).astype("int64")})
    ann = pd.read_csv(S.fafb_annotations_file(), sep="\t", low_memory=False,
                      dtype={"root_id": "int64", "supervoxel_id": "int64"})
    # Reconcile the 783 proofread ids with annotation v3.2.0 (Step 0 note).
    in_edges = pd.Index(np.union1d(edgeroi["pre_id"].unique(), edgeroi["post_id"].unique()))
    no_ann = ids.loc[~ids["neuron_id"].isin(ann["root_id"]), "neuron_id"]
    not_783 = ann.loc[~ann["root_id"].isin(ids["neuron_id"]), "root_id"]
    _record("fafb", "proofread_ids_783", len(ids))
    _record("fafb", "annotation_rows_v3.2.0", len(ann))
    _record("fafb", "proofread_ids_without_annotation", sorted(no_ann.tolist()))
    _record("fafb", "proofread_ids_without_annotation_in_edges", int(no_ann.isin(in_edges).sum()))
    _record("fafb", "annotation_ids_not_in_783", sorted(not_783.tolist()))
    _record("fafb", "annotation_ids_not_in_783_in_edges", int(not_783.isin(in_edges).sum()))
    # Neurons = the 783 proofread ids: ids without an annotation row appear in edges, so they
    # are kept with unknown annotations; annotation rows outside 783 have no edges and are dropped.
    df = ids.merge(ann.rename(columns={"root_id": "neuron_id"}), on="neuron_id",
                   how="left", validate="one_to_one")
    region, ties = fafb_region(edgeroi)
    _record("fafb", "region_ties_alphabetical", ties)
    nt = recode(df["top_nt"], FAFB_NT, "nt_type")
    na = pd.Series(pd.NA, index=df.index, dtype="string")
    out = pd.DataFrame({
        "neuron_id": df["neuron_id"],
        "dataset": "fafb",
        "release": FAFB_RELEASE,
        "cell_type": df["cell_type"],
        "cell_class": df["cell_class"],
        "cell_sub_class": df["cell_sub_class"],
        "super_class": recode(df["super_class"], FAFB_SUPER, "super_class"),
        "side": recode(df["side"], FAFB_SIDE, "side"),
        "region": df["neuron_id"].map(region).fillna("unknown"),
        "nt_type": nt,
        "nt_confidence": df["top_nt_conf"].where(nt != "unknown"),
        "n_pre": pd.array([pd.NA] * len(df), dtype="Int32"),      # not provided by FAFB
        "n_post": pd.array([pd.NA] * len(df), dtype="Int32"),
        "flywire_type": na,
        "manc_type": na,
        "hemibrain_type": df["hemibrain_type"],
        "dimorphism": df["dimorphism"],
        "status": df["status"],
        "hemilineage": df["ito_lee_hemilineage"],
        "nt_known": df["known_nt"],
    })
    return _cast(out, config.NEURON_SCHEMA)


def fafb_tables(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    # The neurons table needs edgeroi (for region) and edgeroi's reference check needs
    # the neurons table, so edgeroi is built once in memory and validated before writing.
    built = {}

    def edgeroi_df():
        if "edgeroi" not in built:
            p = resolve(path("fafb", "edgeroi"))
            built["edgeroi"] = (pd.read_parquet(p) if p.exists() and not force_refresh
                                else build_fafb_edgeroi())
        return built["edgeroi"]

    neurons = cached(path("fafb", "neurons"), lambda: build_fafb_neurons(edgeroi_df()),
                     validate_neurons, force_refresh)
    edgeroi = cached(path("fafb", "edgeroi"), edgeroi_df,
                     lambda e: validate_edges(e, neurons, roi=True), force_refresh)
    edges = cached(path("fafb", "edges"), lambda: build_fafb_edges(edgeroi),
                   lambda e: validate_edges(e, neurons), force_refresh)
    return {"neurons": neurons, "edges": edges, "edgeroi": edgeroi}


# ---------------------------------------------------------------------------
# MaleCNS
# ---------------------------------------------------------------------------
def raw_malecns_neurons(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """neuPrint fetch_neurons(status='Traced'): (properties, per-ROI roiInfo counts).

    Stored once under data/raw: it is the untransformed source table.
    """
    p_n = S.MALECNS_RAW / "neuprint_neurons_traced.parquet"
    p_r = S.MALECNS_RAW / "neuprint_roi_counts_traced.parquet"
    if p_n.exists() and p_r.exists() and not force_refresh:
        return pd.read_parquet(p_n), pd.read_parquet(p_r)
    from neuprint import NeuronCriteria as NC, fetch_neurons

    neurons, rois = fetch_neurons(NC(status=config.MALECNS_STATUS), client=S.neuprint_client())
    neurons = neurons.drop(columns=["roiInfo"], errors="ignore")
    for col in neurons.columns:                       # lists/dicts -> JSON strings for parquet
        if neurons[col].dtype == object and neurons[col].map(lambda v: isinstance(v, (list, dict))).any():
            neurons[col] = neurons[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    p_n.parent.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(p_n, index=False)
    rois.to_parquet(p_r, index=False)
    return neurons, rois


def roi_hierarchy() -> dict:
    p = S.MALECNS_RAW / "roi_hierarchy.json"
    if not p.exists():
        from neuprint import fetch_roi_hierarchy
        h = fetch_roi_hierarchy(include_subprimary=True, mark_primary=True, format="dict",
                                client=S.neuprint_client())
        p.write_text(json.dumps(h, indent=1))
    return json.loads(p.read_text())


def primary_rois() -> list[str]:
    """Primary (non-overlapping) ROIs, marked '*' in the neuPrint ROI hierarchy."""
    found = set()

    def walk(d):
        for k, v in d.items():
            if k.endswith("*"):
                found.add(k[:-1])
            walk(v or {})
    walk(roi_hierarchy())
    return sorted(found)


def malecns_region() -> tuple[pd.Series, int]:
    """Group decision (a): ROI with the largest roiInfo upstream + downstream (connection
    counts, comparable to FAFB), over primary ROIs only. ROI names are kept verbatim."""
    _, rois = raw_malecns_neurons()
    r = rois[rois["roi"].isin(primary_rois())]
    counts = pd.DataFrame({"neuron_id": r["bodyId"].astype("int64"), "neuropil": r["roi"],
                           "n": r["upstream"] + r["downstream"]})
    return top_region(counts)


def build_malecns_neurons() -> pd.DataFrame:
    n, _ = raw_malecns_neurons()
    region, ties = malecns_region()
    _record("malecns", "region_ties_alphabetical", ties)
    _record("malecns", "neurons_without_primary_roi_synapses", int((~n["bodyId"].isin(region.index)).sum()))
    side_raw = n["somaSide"].where(n["somaSide"].notna(), n["rootSide"])
    nt = recode(n["predictedNt"], MALECNS_NT, "nt_type")
    _record("malecns", "nt_confidence_blanked_where_nt_unknown",
            int((n["predictedNtConfidence"].notna() & (nt == "unknown")).sum()))
    out = pd.DataFrame({
        "neuron_id": n["bodyId"].astype("int64"),
        "dataset": "malecns",
        "release": MALECNS_RELEASE,
        "cell_type": n["type"],
        "cell_class": n["class"],
        "cell_sub_class": n["subclass"],
        "super_class": recode(n["superclass"], MALECNS_SUPER, "super_class"),
        "side": recode(side_raw, MALECNS_SIDE, "side"),
        "region": n["bodyId"].astype("int64").map(region).fillna("unknown"),
        "nt_type": nt,
        "nt_confidence": n["predictedNtConfidence"].where(nt != "unknown"),
        "n_pre": n["pre"],
        "n_post": n["post"],
        "flywire_type": n["flywireType"],
        "manc_type": n["mancType"],
        "hemibrain_type": n["hemibrainType"],
        "dimorphism": n["dimorphism"],
        "status": n["status"],
        "hemilineage": n["itoleeHl"],
        "nt_known": pd.Series(pd.NA, index=n.index, dtype="string"),   # not provided by MaleCNS
    })
    return _cast(out, config.NEURON_SCHEMA)


def build_malecns_edgeroi(neuron_ids: np.ndarray) -> pd.DataFrame:
    """Per-ROI edges from Janelia's synapse-partner table (Traced -> Traced, conf >= 0.5):
    count partner pairs per (body_pre, body_post, primary_post). Synapses outside every
    primary ROI ('<unspecified>' in the source) get neuropil 'unknown'."""
    table = feather.read_table(S.malecns_file("partners_traced"), memory_map=True,
                               columns=["body_pre", "body_post", "conf_pre", "conf_post", "primary_post"])
    conf_min = min(table.column("conf_pre").to_numpy().min(), table.column("conf_post").to_numpy().min())
    _record("malecns", "partner_rows", table.num_rows)
    _record("malecns", "partner_min_confidence", float(conf_min))
    if conf_min < config.MALECNS_SYN_CONFIDENCE:
        raise ValueError(f"synapse partners below confidence {config.MALECNS_SYN_CONFIDENCE}: {conf_min}")
    roi = table.column("primary_post").to_pandas()                  # categorical, one dictionary
    pre = table.column("body_pre").to_numpy()
    post = table.column("body_post").to_numpy()
    del table
    ids = np.sort(neuron_ids)
    pi, qi = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    pi[pi == len(ids)] = 0
    qi[qi == len(ids)] = 0
    untraced = (ids[pi] != pre) | (ids[qi] != post)
    _record("malecns", "partner_rows_with_non_traced_body", int(untraced.sum()))
    if untraced.any():
        raise ValueError(f"{int(untraced.sum())} partner rows involve a body outside the Traced set")
    del pre, post
    n, k = np.int64(len(ids)), np.int64(len(roi.cat.categories))
    key = (pi.astype(np.int64) * n + qi) * k + roi.cat.codes.to_numpy().astype(np.int64)
    del pi, qi
    uniq, counts = np.unique(key, return_counts=True)
    del key
    names = np.array(roi.cat.categories, dtype=object)
    names[names == UNSPECIFIED_SYNAPSE] = "unknown"
    _record("malecns", "partner_rows_outside_primary_rois",
            int((roi == UNSPECIFIED_SYNAPSE).sum()))
    df = pd.DataFrame({"pre_id": ids[uniq // k // n], "post_id": ids[uniq // k % n],
                       "neuropil": names[uniq % k], "syn_count": counts})
    df = _drop_self(df, "malecns", "edgeroi")
    return _finish_edges(df, "malecns", MALECNS_RELEASE, roi=True)


def build_malecns_edges(edgeroi: pd.DataFrame) -> pd.DataFrame:
    """D8: sum the per-ROI rows over ROIs; cross-checked against Janelia's summed weights file."""
    e = edgeroi.groupby(["pre_id", "post_id"], sort=False)["syn_count"].sum().reset_index()
    w = pd.read_feather(S.malecns_file("weights_traced"), columns=["body_pre", "body_post", "weight"])
    w = w.rename(columns={"body_pre": "pre_id", "body_post": "post_id", "weight": "syn_count"})
    _record("malecns", "weights_file_rows", len(w))
    w = _drop_self(w, "malecns", "edges")
    m = w.merge(e, on=["pre_id", "post_id"], how="outer", suffixes=("_file", "_roi"), indicator=True)
    _record("malecns", "weights_file_vs_edgeroi_sum", {
        "only_in_file": int((m["_merge"] == "left_only").sum()),
        "only_in_edgeroi": int((m["_merge"] == "right_only").sum()),
        "weight_differs": int((m["_merge"].eq("both") & (m["syn_count_file"] != m["syn_count_roi"])).sum())})
    return _finish_edges(e, "malecns", MALECNS_RELEASE, roi=False)


def malecns_tables(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    neurons = cached(path("malecns", "neurons"), build_malecns_neurons, validate_neurons, force_refresh)
    ids = neurons["neuron_id"].to_numpy()
    edgeroi = cached(path("malecns", "edgeroi"), lambda: build_malecns_edgeroi(ids),
                     lambda e: validate_edges(e, neurons, roi=True), force_refresh)
    edges = cached(path("malecns", "edges"), lambda: build_malecns_edges(edgeroi),
                   lambda e: validate_edges(e, neurons), force_refresh)
    return {"neurons": neurons, "edges": edges, "edgeroi": edgeroi}


def load(dataset: str, tables=("neurons", "edges")) -> dict[str, pd.DataFrame]:
    """Read cached tables only (no network, no rebuild): what the notebooks use."""
    return {t: pd.read_parquet(resolve(path(dataset, t))) for t in tables}
