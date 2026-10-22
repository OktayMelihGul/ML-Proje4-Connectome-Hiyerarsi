"""Fetch neurons and edges for FAFB (FlyWire v783) and MaleCNS (v1.0) in the project schema.

Sources (fastest reliable, see README "Data sources"):

* FAFB 783  - public bulk files: Zenodo record 10676866 (proofread connections,
  per-neuron neuropil synapse counts, proofread root ids) and the
  flyconnectome/flywire_annotations repo (Supplemental file 1). These are the
  same data the CAVE materialization 783 serves, without per-neuron API calls.
* MaleCNS v1.0 - neurons via neuprint-python ``fetch_neurons`` (all properties +
  per-ROI synapse counts); edges via Janelia's official flat export of the
  same release (traced-only connection weights, already summed over ROIs).
  A paged Cypher route over ``ConnectsTo`` is kept as ``fetch_malecns_edges_neuprint``
  and used as a cross-check.

Public API: fetch_fafb_neurons, fetch_fafb_edges(min_syn), fetch_malecns_neurons,
fetch_malecns_edges(min_syn).
"""
from __future__ import annotations

import functools
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C
from .cache import cache_path, cached

log = logging.getLogger(__name__)
DOWNLOAD_LOG = C.RAW_DIR / "download_log.json"


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def download(url: str, dest: Path, chunk: int = 1 << 22) -> Path:
    """Stream ``url`` to ``dest`` once; log size and wall time."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk):
                fh.write(block)
    tmp.replace(dest)
    entry = {"url": url, "bytes": dest.stat().st_size, "seconds": round(time.perf_counter() - t0, 1)}
    logs = json.loads(DOWNLOAD_LOG.read_text()) if DOWNLOAD_LOG.exists() else {}
    logs[dest.name] = entry
    DOWNLOAD_LOG.write_text(json.dumps(logs, indent=2))
    return dest


def _fafb_file(key: str) -> Path:
    name = C.FAFB_FILES[key]
    return download(C.FAFB_ZENODO_URL.format(name=name), C.RAW_DIR / "fafb_783" / name)


def _malecns_file(key: str) -> Path:
    name = C.MALECNS_FILES[key]
    return download(C.MALECNS_BUCKET_URL.format(name=name), C.RAW_DIR / "malecns_v1.0" / name)


# ---------------------------------------------------------------------------
# Raw loaders (exposed for evidence.py, which needs the untransformed columns)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def raw_fafb_annotations() -> pd.DataFrame:
    p = download(C.FAFB_ANNOT_URL, C.RAW_DIR / "fafb_783" / "Supplemental_file1_neuron_annotations.tsv")
    return pd.read_csv(p, sep="\t", low_memory=False, dtype={"root_id": "int64", "supervoxel_id": "int64"})


@functools.lru_cache(maxsize=None)
def raw_fafb_root_ids() -> np.ndarray:
    return np.load(_fafb_file("root_ids")).astype("int64")


@functools.lru_cache(maxsize=None)
def raw_fafb_neuropil_counts() -> pd.DataFrame:
    """Long table: neuron_id, neuropil, n_pre, n_post (synapse counts per neuropil)."""
    pre = pd.read_feather(_fafb_file("neuropil_pre")).rename(
        columns={"pre_pt_root_id": "neuron_id", "count": "n_pre"})
    post = pd.read_feather(_fafb_file("neuropil_post")).rename(
        columns={"post_pt_root_id": "neuron_id", "count": "n_post"})
    df = pre.merge(post, on=["neuron_id", "neuropil"], how="outer").fillna({"n_pre": 0, "n_post": 0})
    return df.astype({"neuron_id": "int64", "n_pre": "int64", "n_post": "int64"})


def raw_fafb_connections(columns=("pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count")) -> pd.DataFrame:
    """Rows are (pre, post, neuropil) triples - NOT yet summed over neuropils."""
    return pd.read_feather(_fafb_file("connections"), columns=list(columns))


def _neuprint_client():
    from neuprint import Client

    C.load_env()
    return Client(C.MALECNS_SERVER, dataset=C.MALECNS_DATASET,
                  token=os.environ["NEUPRINT_APPLICATION_CREDENTIALS"])


def raw_malecns_neurons(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """neuPrint ``fetch_neurons(status='Traced')``: (neuron properties, per-ROI counts).

    Stored under data/raw because it is the untransformed source table.
    """
    from neuprint import NeuronCriteria as NC, fetch_neurons

    p_n = C.RAW_DIR / "malecns_v1.0" / "neuprint_neurons_traced.parquet"
    p_r = C.RAW_DIR / "malecns_v1.0" / "neuprint_roi_counts_traced.parquet"
    if p_n.exists() and p_r.exists() and not force_refresh:
        return pd.read_parquet(p_n), pd.read_parquet(p_r)
    client = _neuprint_client()
    t0 = time.perf_counter()
    neurons, rois = fetch_neurons(NC(status=C.MALECNS_STATUS), client=client)
    log.info("neuPrint fetch_neurons: %d neurons in %.0fs", len(neurons), time.perf_counter() - t0)
    neurons = neurons.drop(columns=["roiInfo"], errors="ignore")
    for col in neurons.columns:                      # lists/dicts -> strings for parquet
        if neurons[col].dtype == object and neurons[col].map(lambda v: isinstance(v, (list, dict))).any():
            neurons[col] = neurons[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    p_n.parent.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(p_n, index=False)
    rois.to_parquet(p_r, index=False)
    return neurons, rois


def raw_malecns_weights() -> pd.DataFrame:
    return pd.read_feather(_malecns_file("weights_traced"), columns=["body_pre", "body_post", "weight"])


def malecns_primary_rois() -> list[str]:
    p = C.RAW_DIR / "malecns_v1.0" / "primary_rois.json"
    if not p.exists():
        p.write_text(json.dumps(sorted(_neuprint_client().primary_rois)))
    return json.loads(p.read_text())


def neuropil_counts(dataset: str) -> pd.DataFrame:
    """Long table neuron_id, neuropil (raw name or None), n_pre, n_post for the neurons table.

    FAFB: neuropil 'None' (synapse outside every neuropil) becomes None - it counts toward
    n_pre/n_post but cannot be a neuron's region.
    MaleCNS: neuPrint primary ROIs only (non-overlapping); downstream = outgoing and
    upstream = incoming synaptic connections, which matches FAFB's per-neuropil counts.
    """
    if dataset == "fafb":
        c = raw_fafb_neuropil_counts()
        c = c[c["neuron_id"].isin(raw_fafb_root_ids())].copy()
        c["neuropil"] = c["neuropil"].where(c["neuropil"] != "None", None)
        return c
    rois = raw_malecns_neurons()[1]
    return (rois[rois["roi"].isin(set(malecns_primary_rois()))]
            .rename(columns={"bodyId": "neuron_id", "roi": "neuropil",
                             "downstream": "n_pre", "upstream": "n_post"})
            [["neuron_id", "neuropil", "n_pre", "n_post"]].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _region_from_counts(counts: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """counts: neuron_id, neuropil(raw), n_pre, n_post -> region, region_group, optic_fraction."""
    c = counts.copy()
    c["region"] = c["neuropil"].map(lambda r: C.normalize_region(r, dataset))
    c["n_syn"] = c["n_pre"] + c["n_post"]
    c["group"] = c["region"].map(C.region_group)
    tot = c.groupby("neuron_id")["n_syn"].sum()
    optic = c[c["group"] == "optic"].groupby("neuron_id")["n_syn"].sum()
    eligible = c[(c["region"] != "unknown") & (c["n_syn"] > 0)]
    # aggregate split ROIs that share a normalized name (MaleCNS MB lobes)
    eligible = eligible.groupby(["neuron_id", "region"], as_index=False)["n_syn"].sum()
    # deterministic argmax: most synapses, ties broken alphabetically (ties counted in D15)
    top = (eligible.sort_values(["neuron_id", "n_syn", "region"], ascending=[True, False, True])
           .drop_duplicates("neuron_id").set_index("neuron_id")["region"])
    out = pd.DataFrame({"region": top})
    out["optic_fraction"] = (optic.reindex(tot.index).fillna(0) / tot.replace(0, np.nan)).reindex(out.index)
    return out


def _finish_neurons(df: pd.DataFrame, regions: pd.DataFrame, dataset: str, release: str) -> pd.DataFrame:
    df = df.join(regions, on="neuron_id")
    df["region"] = df["region"].fillna("unknown")
    df["region_group"] = df["region"].map(C.region_group)
    df["is_optic"] = df["region_group"].eq("optic")
    df["dataset"] = dataset
    df["release"] = release
    return C.apply_schema(df, "neurons")


def _finish_edges(df: pd.DataFrame, min_syn: int, dataset: str, release: str) -> pd.DataFrame:
    df = df[df["syn_count"] >= min_syn].copy()
    df["dataset"] = dataset
    df["release"] = release
    df["min_syn_fetched"] = min_syn
    return C.apply_schema(df.sort_values(["pre_id", "post_id"]), "edges")


# ---------------------------------------------------------------------------
# FAFB
# ---------------------------------------------------------------------------
def _build_fafb_neurons() -> pd.DataFrame:
    ids = pd.DataFrame({"neuron_id": raw_fafb_root_ids()})
    a = raw_fafb_annotations().rename(columns={"root_id": "neuron_id"})
    df = ids.merge(a, on="neuron_id", how="left", validate="one_to_one")
    counts = neuropil_counts("fafb")
    totals = counts.groupby("neuron_id")[["n_pre", "n_post"]].sum()
    df = df.join(totals, on="neuron_id")
    df[["n_pre", "n_post"]] = df[["n_pre", "n_post"]].fillna(0)
    regions = _region_from_counts(counts, "fafb")
    out = pd.DataFrame({
        "neuron_id": df["neuron_id"],
        # cell_type, falling back to hemibrain_type (affects 2 neurons in 783)
        "cell_type": df["cell_type"].fillna(df["hemibrain_type"]),
        "super_class": C.recode(df["super_class"], C.SUPERCLASS_MAP["fafb"], "fafb.super_class"),
        "side": C.recode(df["side"], C.SIDE_MAP["fafb"], "fafb.side"),
        "nt_type": C.recode(df["top_nt"], C.NT_MAP["fafb"], "fafb.top_nt"),
        "nt_confidence": df["top_nt_conf"],
        "n_pre": df["n_pre"], "n_post": df["n_post"],
        "flywire_type": df["cell_type"],
        "manc_type": pd.Series(pd.NA, index=df.index, dtype="string"),
        "hemibrain_type": df["hemibrain_type"],
        "cell_class": df["cell_class"], "cell_sub_class": df["cell_sub_class"],
        "super_class_raw": df["super_class"],
    })
    return _finish_neurons(out, regions, "fafb", C.FAFB_RELEASE)


def fetch_fafb_neurons(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_path(f"fafb_neurons_{C.FAFB_RELEASE}"), _build_fafb_neurons, force_refresh)


def _build_fafb_edges(min_syn: int) -> pd.DataFrame:
    raw = raw_fafb_connections()
    agg = (raw.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False)["syn_count"].sum()
           .reset_index().rename(columns={"pre_pt_root_id": "pre_id", "post_pt_root_id": "post_id"}))
    return _finish_edges(agg, min_syn, "fafb", C.FAFB_RELEASE)


def fetch_fafb_edges(min_syn: int = C.MIN_SYN_DEFAULT, force_refresh: bool = False) -> pd.DataFrame:
    """Neuron-to-neuron edges, synapses summed over neuropils, then ``syn_count >= min_syn``."""
    return cached(cache_path(f"fafb_edges_{C.FAFB_RELEASE}_min{min_syn}"),
                  lambda: _build_fafb_edges(min_syn), force_refresh)


# ---------------------------------------------------------------------------
# MaleCNS
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")


def _build_malecns_neurons() -> pd.DataFrame:
    n, _ = raw_malecns_neurons()
    regions = _region_from_counts(neuropil_counts("malecns"), "malecns")
    side_raw = n["somaSide"].where(n["somaSide"].notna(), n["rootSide"])
    out = pd.DataFrame({
        "neuron_id": n["bodyId"].astype("int64"),
        "cell_type": n["type"],
        "super_class": C.recode(n["superclass"], C.SUPERCLASS_MAP["malecns"], "malecns.superclass"),
        "side": C.recode(side_raw, C.SIDE_MAP["malecns"], "malecns.somaSide|rootSide"),
        "nt_type": C.recode(n["predictedNt"], C.NT_MAP["malecns"], "malecns.predictedNt"),
        "nt_confidence": n["predictedNtConfidence"],
        "n_pre": n["downstream"], "n_post": n["upstream"],
        "flywire_type": _col(n, "flywireType"), "manc_type": _col(n, "mancType"),
        "hemibrain_type": _col(n, "hemibrainType"),
        "cell_class": _col(n, "class"), "cell_sub_class": _col(n, "subclass"),
        "super_class_raw": n["superclass"],
    })
    return _finish_neurons(out, regions, "malecns", C.MALECNS_RELEASE)


def fetch_malecns_neurons(force_refresh: bool = False) -> pd.DataFrame:
    return cached(cache_path(f"malecns_neurons_{C.MALECNS_RELEASE}"), _build_malecns_neurons, force_refresh)


def _build_malecns_edges(min_syn: int) -> pd.DataFrame:
    w = raw_malecns_weights().rename(columns={"body_pre": "pre_id", "body_post": "post_id", "weight": "syn_count"})
    return _finish_edges(w, min_syn, "malecns", C.MALECNS_RELEASE)


def fetch_malecns_edges(min_syn: int = C.MIN_SYN_DEFAULT, force_refresh: bool = False) -> pd.DataFrame:
    """Traced-to-traced edges (weights summed over ROIs by Janelia), ``syn_count >= min_syn``."""
    return cached(cache_path(f"malecns_edges_{C.MALECNS_RELEASE}_min{min_syn}"),
                  lambda: _build_malecns_edges(min_syn), force_refresh)


def fetch_malecns_edges_neuprint(min_syn: int, body_ids=None, page_size: int = C.NEUPRINT_PAGE_SIZE,
                                 progress: bool = False) -> pd.DataFrame:
    """Same table via paged Cypher on neuPrint (``ConnectsTo.weight`` is already summed over ROIs).

    Pages over source neurons so no single query returns the whole graph.
    """
    client = _neuprint_client()
    if body_ids is None:
        body_ids = raw_malecns_neurons()[0]["bodyId"].to_numpy()
    body_ids = np.sort(np.asarray(body_ids, dtype="int64"))
    parts = []
    for i in range(0, len(body_ids), page_size):
        ids = ",".join(map(str, body_ids[i:i + page_size]))
        q = (f"MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) "
             f"WHERE a.bodyId IN [{ids}] AND b.status = '{C.MALECNS_STATUS}' AND e.weight >= {int(min_syn)} "
             f"RETURN a.bodyId AS pre_id, b.bodyId AS post_id, e.weight AS syn_count")
        parts.append(client.fetch_custom(q))
        if progress:
            log.info("neuPrint edges page %d/%d", i // page_size + 1, -(-len(body_ids) // page_size))
    df = pd.concat(parts, ignore_index=True)
    return _finish_edges(df, min_syn, "malecns", C.MALECNS_RELEASE)
