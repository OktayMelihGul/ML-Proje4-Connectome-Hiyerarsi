"""Evidence for the week-2/3 worksheet: prints and saves reports/evidence.md + CSVs.

    python -m src.evidence --part data      # D1-D9 (data/schema evidence)
    python -m src.evidence --part graph     # thresholds, optic, D15, D16, D18, sensory BFS depth
    python -m src.evidence                  # both

All graph work uses scipy.sparse; no dense adjacency matrix is ever built.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csgraph

from . import config as C, fetch

OUT = C.REPORTS_DIR / "evidence"
DS = ("fafb", "malecns")
MD: list[str] = []
SENSORY_CLASSES = ("sensory", "sensory_ascending")
MOTOR_CLASSES = ("motor",)


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def h(text: str, level: int = 2):
    MD.append(f"\n{'#' * level} {text}\n")
    print(f"\n{'#' * level} {text}")


def note(text: str):
    MD.append(text + "\n")
    print(text)


def table(df: pd.DataFrame, name: str | None = None, index: bool = False, floatfmt: str = ".4g"):
    if name:
        df.to_csv(OUT / f"{name}.csv", index=index)
    MD.append(df.to_markdown(index=index, floatfmt=floatfmt) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 80):
        print(df.to_string(index=index))


def neurons(ds):
    return fetch.fetch_fafb_neurons() if ds == "fafb" else fetch.fetch_malecns_neurons()


def edges(ds, m=1):
    return fetch.fetch_fafb_edges(m) if ds == "fafb" else fetch.fetch_malecns_edges(m)


def raw_neuron_table(ds) -> pd.DataFrame:
    if ds == "fafb":
        return fetch.raw_fafb_annotations()
    return fetch.raw_malecns_neurons()[0]


# ---------------------------------------------------------------------------
# Part 1: data evidence D1-D9
# ---------------------------------------------------------------------------
def d1_columns():
    h("D1. Raw column lists per source (sorted)")
    import pyarrow.feather as pf
    srcs = {
        "fafb: Supplemental_file1_neuron_annotations.tsv": list(fetch.raw_fafb_annotations().columns),
        "fafb: proofread_connections_783.feather": pf.read_table(
            C.RAW_DIR / "fafb_783" / C.FAFB_FILES["connections"], memory_map=True).column_names,
        "fafb: per_neuron_neuropil_count_pre_783.feather": ["pre_pt_root_id", "neuropil", "count"],
        "fafb: per_neuron_neuropil_count_post_783.feather": ["post_pt_root_id", "neuropil", "count"],
        "malecns: neuPrint fetch_neurons (neuron table)": list(fetch.raw_malecns_neurons()[0].columns) + ["roiInfo (dropped; expanded into ROI table)"],
        "malecns: neuPrint fetch_neurons (ROI table)": list(fetch.raw_malecns_neurons()[1].columns),
        "malecns: connectome-weights-...-traced-only.feather": pf.read_table(
            C.RAW_DIR / "malecns_v1.0" / C.MALECNS_FILES["weights_traced"], memory_map=True).column_names,
    }
    rows = [{"source": k, "n_columns": len(v), "columns": ", ".join(sorted(v, key=str.lower))} for k, v in srcs.items()]
    table(pd.DataFrame(rows), "D1_columns")
    note("Final neurons table columns (schema order): " + ", ".join(C.NEURONS_SCHEMA))
    note("Final edges table columns (schema order): " + ", ".join(C.EDGES_SCHEMA))


def d2_ids():
    h("D2. neuron_id range and float64 round-trip")
    rows = []
    for ds in DS:
        ids = neurons(ds)["neuron_id"].to_numpy()
        f = ids.astype("float64")
        back = f.astype("int64")
        rows.append({"dataset": ds, "n_ids": len(ids), "min_id": int(ids.min()), "max_id": int(ids.max()),
                     "max_id > 2^53": bool(ids.max() > 2 ** 53),
                     "ids changed by int64->float64->int64": int((back != ids).sum()),
                     "distinct ids": len(np.unique(ids)), "distinct after float64": len(np.unique(f)),
                     "ids lost to collisions": len(np.unique(ids)) - len(np.unique(f))})
    table(pd.DataFrame(rows), "D2_ids")
    note("FAFB root ids (~7.2e17) exceed 2^53, so any float64 step (CSV read without dtype, "
         "pandas NaN-upcast on merge, JSON in JS) silently corrupts them. Keep int64 / string end to end.")


def _vc(s: pd.Series, name: str) -> pd.DataFrame:
    v = s.astype(object).where(s.notna(), "<NA>").value_counts(dropna=False)
    return pd.DataFrame({"column": name, "value": v.index.astype(str), "count": v.values})


def d3_side():
    h("D3. Raw side values (value_counts, dropna=False)")
    a = fetch.raw_fafb_annotations()
    m = fetch.raw_malecns_neurons()[0]
    parts = [_vc(a["side"], "fafb.side"), _vc(m["somaSide"], "malecns.somaSide"),
             _vc(m["rootSide"], "malecns.rootSide"),
             _vc(m["somaSide"].where(m["somaSide"].notna(), m["rootSide"]), "malecns.somaSide|rootSide (used)")]
    for ds in DS:
        parts.append(_vc(neurons(ds)["side"], f"{ds}.side (mapped)"))
    table(pd.concat(parts), "D3_side")


def d3b_superclass():
    h("D3b. Raw super_class values (sensory/motor seed sets are built from these)")
    a = fetch.raw_fafb_annotations()
    a = a[a["root_id"].isin(fetch.raw_fafb_root_ids())]
    m = fetch.raw_malecns_neurons()[0]
    parts = [_vc(a["super_class"], "fafb.super_class (783 ids)"), _vc(m["superclass"], "malecns.superclass")]
    for ds in DS:
        parts.append(_vc(neurons(ds)["super_class"], f"{ds}.super_class (mapped)"))
    table(pd.concat(parts), "D3b_superclass")
    note("Mapping MaleCNS -> shared vocabulary: " + "; ".join(
        f"{k}->{v}" for k, v in C.SUPERCLASS_MAP["malecns"].items() if k is not None) + "; null->unknown.")


def d4_nt():
    h("D4. Raw neurotransmitter values and confidence")
    a = fetch.raw_fafb_annotations()
    m = fetch.raw_malecns_neurons()[0]
    parts = [_vc(a["top_nt"], "fafb.top_nt"), _vc(m["predictedNt"], "malecns.predictedNt (used)"),
             _vc(m["consensusNt"], "malecns.consensusNt"), _vc(m["celltypePredictedNt"], "malecns.celltypePredictedNt")]
    table(pd.concat(parts), "D4_nt_values")
    rows = []
    for name, s in (("fafb.top_nt_conf", a["top_nt_conf"]), ("malecns.predictedNtConfidence", m["predictedNtConfidence"]),
                    ("malecns.celltypePredictedNtConfidence", m["celltypePredictedNtConfidence"])):
        s = pd.to_numeric(s, errors="coerce")
        q = s.quantile([0, .05, .25, .5, .75, .95, 1]).round(3)
        rows.append({"column": name, "n": int(s.notna().sum()), "null": int(s.isna().sum()),
                     **{f"q{int(k * 100)}": v for k, v in q.items()},
                     "share<0.5": round(float((s < .5).sum() / s.notna().sum()), 4),
                     "share<0.7": round(float((s < .7).sum() / s.notna().sum()), 4)})
    table(pd.DataFrame(rows), "D4_nt_confidence")


def d5_nulls():
    h("D5. Null counts per raw and per mapped column")
    rows = []
    for ds in DS:
        raw = raw_neuron_table(ds)
        if ds == "fafb":   # restrict raw annotations to the 783 proofread ids used downstream
            raw = raw[raw["root_id"].isin(fetch.raw_fafb_root_ids())]
        for c in raw.columns:
            rows.append({"dataset": ds, "level": "raw", "column": c, "rows": len(raw), "nulls": int(raw[c].isna().sum())})
        n = neurons(ds)
        for c in n.columns:
            nn = int(n[c].isna().sum())
            if c in ("side", "nt_type", "super_class", "region"):
                nn_unknown = int((n[c].astype(str) == "unknown").sum())
                rows.append({"dataset": ds, "level": "mapped", "column": c, "rows": len(n), "nulls": nn,
                             "value 'unknown'": nn_unknown})
            else:
                rows.append({"dataset": ds, "level": "mapped", "column": c, "rows": len(n), "nulls": nn})
    df = pd.DataFrame(rows)
    table(df, "D5_nulls")
    ann = fetch.raw_fafb_annotations()
    ids = set(fetch.raw_fafb_root_ids())
    note(f"FAFB: annotation rows {len(ann):,}; 783 proofread ids {len(ids):,}; annotation ids not in 783 list: "
         f"{len(set(ann.root_id) - ids)}; 783 ids without annotation: {len(ids - set(ann.root_id))} "
         f"(these get super_class/side/nt = 'unknown').")


def d6_releases():
    h("D6. Available releases")
    from neuprint import Client
    C.load_env()
    import os
    c = fetch._neuprint_client()
    rows = []
    for k, v in c.fetch_datasets().items():
        rows.append({"server": "neuPrint", "dataset": k, "last_mod": str(v.get("last-mod", ""))[:60],
                     "description (first line)": str(v.get("description", "")).split("\n")[0][:140]})
    try:
        from caveclient import CAVEclient
        cv = CAVEclient(C.FAFB_DATASTACK, auth_token=os.environ.get("CAVE_TOKEN"))
        for v in cv.materialize.get_versions():
            md = cv.materialize.get_version_metadata(v)
            rows.append({"server": "CAVE flywire_fafb_public", "dataset": f"materialization {v}",
                         "last_mod": str(md.get("time_stamp"))[:25], "description (first line)": ""})
    except Exception as e:   # never echo tokens; error type only
        rows.append({"server": "CAVE", "dataset": "error", "last_mod": "", "description (first line)": type(e).__name__})
    table(pd.DataFrame(rows), "D6_releases")
    note("Pinned: FAFB materialization 783 (public versions: 630 and 783; 783 is the version of the 2024 "
         "Nature papers, 630 the 2023 preprints) and neuPrint male-cns:v1.0 (v0.9 is the preprint release; "
         "v1.0 adds Google Connectomics to the description and a 2026 segment-property update).")


def d8_aggregation():
    h("D8. Aggregation over neuropil/ROI and self-connections")
    raw = fetch.raw_fafb_connections(columns=("pre_pt_root_id", "post_pt_root_id", "syn_count"))
    rows = [{"dataset": "fafb", "raw rows (pre, post, neuropil)": len(raw),
             "raw min syn_count": int(raw["syn_count"].min())}]
    w = fetch.raw_malecns_weights()
    rows.append({"dataset": "malecns", "raw rows (pre, post, neuropil)": "n/a: flat export already summed over ROIs",
                 "raw min syn_count": int(w["weight"].min())})
    for r, ds in zip(rows, DS):
        e = edges(ds, 1)
        self_ = e[e["pre_id"] == e["post_id"]]
        r.update({"edges after aggregation (min1)": len(e), "total synapses": int(e["syn_count"].sum()),
                  "self-connections (min1)": len(self_), "self-connection synapses": int(self_["syn_count"].sum()),
                  "self-connections >=5": int((self_["syn_count"] >= 5).sum())})
    table(pd.DataFrame(rows), "D8_aggregation")
    note("Self-connections are kept in the cached tables; whether to drop them is recorded as "
         "'to be agreed in group'. They do not change BFS depth (a self-loop never shortens a path).")


def d8b_neuprint_crosscheck():
    """One page of the paged Cypher route vs the flat export (same release)."""
    h("D8b. neuPrint paged Cypher vs Janelia flat export (one page)")
    ids = np.sort(fetch.raw_malecns_neurons()[0]["bodyId"].to_numpy())[:C.NEUPRINT_PAGE_SIZE]
    t0 = time.perf_counter()
    q = fetch.fetch_malecns_edges_neuprint(1, body_ids=ids)
    dt = time.perf_counter() - t0
    b = edges("malecns", 1)
    b = b[b["pre_id"].isin(ids)]
    m = q.merge(b, on=["pre_id", "post_id"], how="outer", suffixes=("_np", "_flat"), indicator=True)
    both = m[m["_merge"] == "both"]
    pages = -(-165_122 // C.NEUPRINT_PAGE_SIZE)
    table(pd.DataFrame([{"source neurons in page": len(ids), "neuPrint rows": len(q), "flat rows": len(b),
                         "only neuPrint": int((m["_merge"] == "left_only").sum()),
                         "only flat": int((m["_merge"] == "right_only").sum()),
                         "weight equal": int((both["syn_count_np"] == both["syn_count_flat"]).sum()),
                         "page seconds": round(dt, 1),
                         "extrapolated full min1 minutes": round(dt * pages / 60, 1)}]), "D8b_neuprint_crosscheck")


def d9_timings():
    h("D9. Timings (cold build from local raw files, cache hits, parquet size)")
    t = pd.read_csv(C.REPORTS_DIR / "timings.csv")
    table(t, "D9_timings")
    log = json.loads((C.RAW_DIR / "download_log.json").read_text())
    rows = [{"download": k, "MB": round(v.get("bytes", 0) / 1e6, 1) if "bytes" in v else "",
             "rows": v.get("rows", ""), "seconds": v["seconds"]} for k, v in log.items() if not k.startswith("_")]
    table(pd.DataFrame(rows), "D9_downloads")
    note("Cold time for an edge table = download (once, shared by all thresholds) + build. "
         "Second calls are served from data/cache (cache_hit=True).")


def summary_section():
    h("Summary vs published figures")
    s = pd.read_csv(C.REPORTS_DIR / "summary.csv")
    table(s, "summary")


# ---------------------------------------------------------------------------
# Part 2: graph evidence
# ---------------------------------------------------------------------------
class Graph:
    """Index neurons 0..N-1 and hold the min1 edge list with relative weights."""

    def __init__(self, ds: str):
        self.ds = ds
        self.n = neurons(ds).sort_values("neuron_id").reset_index(drop=True)
        self.ids = self.n["neuron_id"].to_numpy()
        e = edges(ds, 1)
        self.pre = np.searchsorted(self.ids, e["pre_id"].to_numpy())
        self.post = np.searchsorted(self.ids, e["post_id"].to_numpy())
        self.w = e["syn_count"].to_numpy().astype(np.int64)
        tot_in = np.bincount(self.post, weights=self.w, minlength=len(self.ids))
        self.rel = self.w / tot_in[self.post]        # share of target's total input (min1 table)
        self.central = self.n["region_group"].eq("central_brain").to_numpy()

    def mask(self, thr) -> np.ndarray:
        if isinstance(thr, float):
            return self.rel >= thr
        return self.w >= thr

    def adj(self, keep: np.ndarray, nodes: np.ndarray | None = None) -> sp.csr_matrix:
        if nodes is not None:
            keep = keep & nodes[self.pre] & nodes[self.post]
        N = len(self.ids)
        return sp.csr_matrix((np.ones(keep.sum(), dtype=np.int8), (self.pre[keep], self.post[keep])), shape=(N, N))


def thr_label(t):
    return f"{t * 100:.1f}%" if isinstance(t, float) else f">={t}"


THRESHOLDS = list(C.ABS_THRESHOLDS) + list(C.REL_THRESHOLDS)
DEPTH_THRESHOLDS = (1, 5, 10, 0.009, 0.01, 0.026)


def threshold_tables(G: Graph):
    rows, tops = [], []
    for scope, nodes in (("all", np.ones(len(G.ids), bool)), ("central_brain", G.central)):
        for t in THRESHOLDS:
            k = G.mask(t) & nodes[G.pre] & nodes[G.post]
            out_deg = np.bincount(G.pre[k], minlength=len(G.ids))[nodes]
            in_deg = np.bincount(G.post[k], minlength=len(G.ids))[nodes]
            connected = int(((out_deg + in_deg) > 0).sum())
            rows.append({"dataset": G.ds, "scope": scope, "threshold": thr_label(t), "neurons": int(nodes.sum()),
                         "edges": int(k.sum()), "synapses": int(G.w[k].sum()), "neurons connected": connected,
                         "neurons isolated": int(nodes.sum()) - connected,
                         "median out-degree": float(np.median(out_deg)), "p99 out-degree": float(np.percentile(out_deg, 99)),
                         "max out-degree": int(out_deg.max())})
        k = G.mask(10) & nodes[G.pre] & nodes[G.post]
        od = pd.Series(np.bincount(G.pre[k], minlength=len(G.ids)))[nodes].nlargest(10)
        t10 = G.n.loc[od.index, ["neuron_id", "cell_type", "super_class", "region"]].assign(out_degree_ge10=od.values)
        tops.append(t10.assign(dataset=G.ds, scope=scope))
    return pd.DataFrame(rows), pd.concat(tops)


def crosstabs(G: Graph):
    rows = []
    for rel in (0.009, 0.01):
        a, b = G.w >= 10, G.rel >= rel
        rows.append({"dataset": G.ds, "comparison": f">=10 syn vs >={rel * 100:.1f}% input",
                     "both": int((a & b).sum()), "only >=10 syn": int((a & ~b).sum()),
                     "only relative": int((~a & b).sum()), "neither": int((~a & ~b).sum()),
                     "jaccard": round(float((a & b).sum() / (a | b).sum()), 4)})
    return pd.DataFrame(rows)


def optic_section(G: Graph):
    n = G.n
    counts = fetch.neuropil_counts(G.ds)
    counts["region"] = counts["neuropil"].map(lambda r: C.normalize_region(r, G.ds))
    counts["group"] = counts["region"].map(C.region_group)
    syn = (counts["n_pre"] + counts["n_post"])
    of = n["optic_fraction"]
    return {"dataset": G.ds, "neurons": len(n),
            "region_group optic": int(n["is_optic"].sum()),
            "optic share of neurons": round(float(n["is_optic"].mean()), 4),
            "central_brain": int((n["region_group"] == "central_brain").sum()),
            "vnc": int((n["region_group"] == "vnc").sum()), "unknown region": int((n["region_group"] == "unknown").sum()),
            "optic share of synapses": round(float(syn[counts["group"] == "optic"].sum() / syn.sum()), 4),
            "split 0.2<optic_frac<0.8": int(((of > .2) & (of < .8)).sum()),
            "split 0.1<optic_frac<0.9": int(((of > .1) & (of < .9)).sum())}


def d15_ties(ds: str):
    c = fetch.neuropil_counts(ds)
    c = c[c["neuropil"].notna()].copy()
    c["region"] = c["neuropil"].map(lambda r: C.normalize_region(r, ds))
    c["s"] = c["n_pre"] + c["n_post"]
    c = c.groupby(["neuron_id", "region"], as_index=False)["s"].sum()
    c = c[c["s"] > 0].sort_values(["neuron_id", "s"], ascending=[True, False])
    c["rank"] = c.groupby("neuron_id").cumcount()
    top = c[c["rank"] == 0].set_index("neuron_id")["s"]
    sec = c[c["rank"] == 1].set_index("neuron_id")["s"].reindex(top.index).fillna(0)
    r = sec / top
    return {"dataset": ds, "neurons with >=1 neuropil": len(top), "with >=2 neuropils": int((sec > 0).sum()),
            "second >= 80% of top": int((r >= .8).sum()), "second >= 90% of top": int((r >= .9).sum()),
            "exact ties": int((r == 1).sum())}


def d16_zero(G: Graph):
    rows = []
    for t in (1, 5, 10, 0.009, 0.01):
        k = G.mask(t)
        od = np.bincount(G.pre[k], minlength=len(G.ids))
        idg = np.bincount(G.post[k], minlength=len(G.ids))
        for scope, nodes in (("all", np.ones(len(G.ids), bool)), ("central_brain", G.central)):
            rows.append({"dataset": G.ds, "threshold": thr_label(t), "scope": scope, "neurons": int(nodes.sum()),
                         "zero inputs": int((idg[nodes] == 0).sum()), "zero outputs": int((od[nodes] == 0).sum()),
                         "zero inputs AND outputs": int(((idg == 0) & (od == 0))[nodes].sum())})
    return rows


def d18_pairs(ds: str):
    n = neurons(ds)
    side = n["side"].astype(str)
    typed = n["cell_type"].notna()
    lr = side.isin(["left", "right"])
    sides_per_type = n[typed & lr].groupby("cell_type")["side"].agg(lambda s: set(s.astype(str)))
    has_opp = n["cell_type"].map(lambda t: sides_per_type.get(t, set()) if pd.notna(t) else set())
    opp = side.map({"left": "right", "right": "left"})
    paired = typed & lr & pd.Series([o in s for o, s in zip(opp, has_opp)], index=n.index)
    return {"dataset": ds, "neurons": len(n), "untyped": int((~typed).sum()),
            "typed, side unknown": int((typed & (side == "unknown")).sum()),
            "typed, side center": int((typed & (side == "center")).sum()),
            "typed L/R with opposite-side partner": int(paired.sum()),
            "typed L/R unpaired": int((typed & lr & ~paired).sum()),
            "unpaired cell types": int(n[typed & lr & ~paired]["cell_type"].nunique())}


def d18b_group_pairs() -> dict:
    """MaleCNS `group`: shared id for left/right homologues. Compare with the cell_type rule."""
    m = fetch.raw_malecns_neurons()[0][["bodyId", "group"]].rename(columns={"bodyId": "neuron_id"})
    n = neurons("malecns")[["neuron_id", "cell_type", "side"]].merge(m, on="neuron_id")
    side = n["side"].astype(str)
    lr = side.isin(["left", "right"])
    g_sides = n[lr & n["group"].notna()].groupby("group")["side"].agg(lambda s: set(s.astype(str)))
    bilateral = g_sides[g_sides.map(lambda x: {"left", "right"} <= x)].index
    in_bilat = n["group"].isin(bilateral) & lr
    paired_ct = n["neuron_id"].isin(_ct_paired_ids("malecns"))
    sizes = n[n["group"].notna()].groupby("group").size()
    return {"dataset": "malecns", "neurons with group": int(n["group"].notna().sum()), "distinct groups": int(len(sizes)),
            "group size median": float(sizes.median()), "groups with >=1 L and >=1 R": int(len(bilateral)),
            "L/R neurons in such groups": int(in_bilat.sum()),
            "L/R neurons paired by cell_type rule": int(paired_ct.sum()),
            "paired by both": int((in_bilat & paired_ct).sum()),
            "group-paired only": int((in_bilat & ~paired_ct).sum()),
            "cell_type-paired only": int((~in_bilat & paired_ct).sum()),
            "group-paired but untyped": int((in_bilat & n["cell_type"].isna()).sum())}


def _ct_paired_ids(ds: str) -> set:
    n = neurons(ds)
    side = n["side"].astype(str)
    sel = n["cell_type"].notna() & side.isin(["left", "right"])
    sides = n[sel].groupby("cell_type")["side"].agg(lambda s: set(s.astype(str)))
    both = set(sides[sides.map(lambda x: {"left", "right"} <= x)].index)
    return set(n.loc[sel & n["cell_type"].isin(both), "neuron_id"])


def sets_section(G: Graph):
    n = G.n
    sens = n["super_class"].isin(SENSORY_CLASSES)
    mot = n["super_class"].isin(MOTOR_CLASSES)
    rows = []
    for name, m in (("sensory", sens), ("motor", mot)):
        vc = n.loc[m, "region_group"].astype(str).value_counts()
        rows.append({"dataset": G.ds, "set": name, "total": int(m.sum()),
                     **{f"in {g}": int(vc.get(g, 0)) for g in C.REGION_GROUPS}})
    by_cls = (n.loc[sens | mot].assign(set=np.where(sens[sens | mot], "sensory", "motor"))
              .groupby(["set", "super_class", "cell_class"], observed=True, dropna=False).size()
              .rename("n").reset_index().assign(dataset=G.ds))
    top_regions = (n.loc[sens].groupby("region", observed=True).size().nlargest(12).rename("sensory neurons")
                   .reset_index().assign(dataset=G.ds))
    return rows, by_cls, top_regions


def bfs_depth(A: sp.csr_matrix, sources: np.ndarray) -> np.ndarray:
    """Multi-source unweighted shortest path length (pre->post). inf = unreachable."""
    d = csgraph.dijkstra(A, directed=True, indices=sources, unweighted=True, min_only=True)
    return d


def depth_section(G: Graph):
    n = G.n
    sens = n["super_class"].isin(SENSORY_CLASSES).to_numpy()
    reach_rows, dist_rows, cls_rows, checks = [], [], [], []
    for t in DEPTH_THRESHOLDS:
        k = G.mask(t)
        for scope, nodes in (("full graph", None), ("central-brain subgraph", G.central)):
            A = G.adj(k, nodes)
            src = np.flatnonzero(sens & (nodes if nodes is not None else True))
            d = bfs_depth(A, src)
            # sanity: sources depth 0; direct non-source targets depth 1
            tgt = np.unique(A[src].indices)
            tgt = tgt[~sens[tgt]]
            checks.append({"dataset": G.ds, "threshold": thr_label(t), "scope": scope,
                           "sources at depth 0": bool((d[src] == 0).all()),
                           "direct targets at depth 1": bool((d[tgt] == 1).all()), "n direct targets": len(tgt)})
            views = [("all neurons", np.ones(len(d), bool) if nodes is None else nodes)]
            if nodes is None:
                views.append(("central-brain neurons", G.central))
            for view, vm in views:
                dv = d[vm]
                unr = ~np.isfinite(dv)
                fin = dv[~unr]
                reach_rows.append({"dataset": G.ds, "threshold": thr_label(t), "graph": scope, "neurons": view,
                                   "n": int(vm.sum()), "sources": int((sens & vm).sum()),
                                   "reachable": int((~unr).sum()), "unreachable": int(unr.sum()),
                                   "unreachable share": round(float(unr.mean()), 4),
                                   "unreachable & isolated": int((unr & (np.asarray(A.sum(0)).ravel()[vm] == 0)
                                                                  & (np.asarray(A.sum(1)).ravel()[vm] == 0)).sum()),
                                   "mean depth (reached, non-source)": round(float(fin[fin > 0].mean()), 3) if (fin > 0).any() else None,
                                   "max depth": int(fin.max()) if len(fin) else None})
                vc = pd.Series(np.where(unr, -1, dv).astype(int)).value_counts().sort_index()
                for depth, cnt in vc.items():
                    dist_rows.append({"dataset": G.ds, "threshold": thr_label(t), "graph": scope, "neurons": view,
                                      "depth": "unreachable" if depth == -1 else int(depth), "count": int(cnt)})
                if view == "all neurons":
                    sc = n.loc[vm, "super_class"].astype(str).to_numpy()
                    df = pd.DataFrame({"super_class": sc, "unr": unr})
                    g = df.groupby("super_class")["unr"].agg(["size", "sum", "mean"]).reset_index()
                    g.columns = ["super_class", "n", "unreachable", "unreachable share"]
                    cls_rows.append(g.assign(dataset=G.ds, threshold=thr_label(t), graph=scope))
    return (pd.DataFrame(reach_rows), pd.DataFrame(dist_rows),
            pd.concat(cls_rows, ignore_index=True), pd.DataFrame(checks))


# ---------------------------------------------------------------------------
def part_data():
    h("Part 1: data evidence (D1-D9)", 1)
    d1_columns(); d2_ids(); d3_side(); d3b_superclass(); d4_nt(); d5_nulls(); d6_releases(); d8_aggregation()
    d8b_neuprint_crosscheck(); d9_timings(); summary_section()


def part_graph():
    h("Part 2: thresholds, regions and sensory depth", 1)
    note("Relative thresholds: edge kept if syn_count / (target's total input in the min1 table) >= x. "
         "Central-brain scope = both endpoints have region_group 'central_brain' (top neuropil not optic, not VNC). "
         f"Optic neuropils: {sorted(C.OPTIC_NEUROPILS)} (each side).")
    graphs = {ds: Graph(ds) for ds in DS}
    thr, tops, xt, opt, d16 = [], [], [], [], []
    for G in graphs.values():
        a, b = threshold_tables(G); thr.append(a); tops.append(b)
        xt.append(crosstabs(G)); opt.append(optic_section(G)); d16 += d16_zero(G)
    h("Threshold table"); table(pd.concat(thr), "threshold_table")
    h("Top-10 out-degree neurons at >=10 synapses"); table(pd.concat(tops), "top10_outdegree_ge10")
    h("Crosstabs: absolute vs relative thresholds (edges of the min1 table)"); table(pd.concat(xt), "crosstabs")
    h("Optic lobe share and optic/central split neurons"); table(pd.DataFrame(opt), "optic_share")
    h("D15. Top-neuropil near-ties"); table(pd.DataFrame([d15_ties(ds) for ds in DS]), "D15_ties")
    h("D16. Neurons with zero inputs / outputs"); table(pd.DataFrame(d16), "D16_zero_degree")
    h("D18. Pairing rule 'same cell_type, opposite side'"); table(pd.DataFrame([d18_pairs(ds) for ds in DS]), "D18_pairs")
    h("D18b. MaleCNS `group` (left/right homologue id) vs cell_type rule"); table(pd.DataFrame([d18b_group_pairs()]), "D18b_group_pairs")
    h("Project 4: sensory and motor sets")
    note(f"Sensory set = super_class in {SENSORY_CLASSES}; motor set = super_class in {MOTOR_CLASSES}.")
    s_rows, s_cls, s_reg = [], [], []
    for G in graphs.values():
        r, c, g = sets_section(G); s_rows += r; s_cls.append(c); s_reg.append(g)
    table(pd.DataFrame(s_rows), "sets_sizes"); table(pd.concat(s_cls), "sets_by_class"); table(pd.concat(s_reg), "sensory_top_regions")
    h("Project 4: multi-source BFS depth from the sensory set (edge direction pre->post)")
    reach, dist, cls, chk = [], [], [], []
    for G in graphs.values():
        a, b, c, d = depth_section(G); reach.append(a); dist.append(b); cls.append(c); chk.append(d)
    h("Sanity checks", 3); table(pd.concat(chk), "depth_sanity")
    h("Reachable vs unreachable", 3); table(pd.concat(reach), "depth_reachability")
    dist = pd.concat(dist)
    dist.to_csv(OUT / "depth_distribution.csv", index=False)
    order = [thr_label(t) for t in DEPTH_THRESHOLDS]
    dist["threshold"] = pd.Categorical(dist["threshold"], order, ordered=True)
    piv = dist.pivot_table(index=["dataset", "graph", "neurons", "threshold"], columns="depth", values="count",
                           aggfunc="sum", fill_value=0, observed=True)
    piv.columns = [str(c) for c in piv.columns]
    h("Depth distribution", 3); table(piv.reset_index(), None)
    cls = pd.concat(cls)
    cls.to_csv(OUT / "unreachable_by_superclass.csv", index=False)
    cls["threshold"] = pd.Categorical(cls["threshold"], order, ordered=True)
    pc = cls.pivot_table(index=["dataset", "graph", "super_class"], columns="threshold", values="unreachable share",
                         observed=True)
    pc.columns = [str(c) for c in pc.columns]
    h("Unreachable share by super_class", 3); table(pc.round(4).reset_index(), None)


def main(part: str):
    OUT.mkdir(parents=True, exist_ok=True)
    if part in ("data", "all"):
        part_data()
    if part in ("graph", "all"):
        part_graph()
    path = C.REPORTS_DIR / ("evidence.md" if part == "all" else f"evidence_{part}.md")
    path.write_text("# Evidence report (generated by src/evidence.py)\n" + "\n".join(MD), encoding="utf-8")
    print(f"\nwritten {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["data", "graph", "all"], default="all")
    main(ap.parse_args().part)
