"""Which filter could explain the ~10k gap in MaleCNS edges >= 5 (ours 6.236M vs Project 1's 6.25M)?

    python -m src.gap_check     -> reports/malecns_edges_ge5_variants.csv

Counts edges with syn_count >= 5 under alternative definitions, all from Janelia's v1.0
flat export at synapse confidence >= 0.5.
"""
import pandas as pd

from . import fetch
from .sources import MALECNS_BUCKET_URL, MALECNS_RAW, ROOT, download

FULL = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
SIGNIFICANT = "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"


def main():
    traced = set(fetch.load("malecns", ["neurons"])["neurons"]["neuron_id"])
    rows = []

    def add(name, w, note=""):
        rows.append({"variant": name, "edges_ge5": int((w["weight"] >= 5).sum()),
                     "edges_ge1": len(w), "note": note})

    t = pd.read_feather(MALECNS_RAW / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
                        columns=["body_pre", "body_post", "weight"])
    add("traced-only file, self-loops kept", t, "week 2 value")
    add("traced-only file, self-loops dropped", t[t.body_pre != t.body_post], "ours (Conventions)")
    full = pd.read_feather(download(MALECNS_BUCKET_URL.format(name=FULL), MALECNS_RAW / FULL),
                           columns=["body_pre", "body_post", "weight"])
    pre_t, post_t = full.body_pre.isin(traced), full.body_post.isin(traced)
    add("full file, both ends Traced", full[pre_t & post_t])
    add("full file, pre Traced (any post)", full[pre_t])
    add("full file, post Traced (any pre)", full[post_t])
    add("full file, either end Traced", full[pre_t | post_t])
    add("full file, all segments", full)
    ANN = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    ann = pd.read_feather(download(MALECNS_BUCKET_URL.format(name=ANN), MALECNS_RAW / ANN), columns=["bodyId", "status"])
    no_self = full[full.body_pre != full.body_post]
    for extra in (["Anchor"], ["Assign"], ["Anchor", "Assign"], ["Orphan"]):
        ids = set(ann.loc[ann.status.isin(["Traced"] + extra), "bodyId"])
        add(f"full file, both ends status in Traced+{'+'.join(extra)}, self-loops dropped",
            no_self[no_self.body_pre.isin(ids) & no_self.body_post.isin(ids)], f"{len(ids):,} bodies")
    sig = pd.read_feather(download(MALECNS_BUCKET_URL.format(name=SIGNIFICANT), MALECNS_RAW / SIGNIFICANT),
                          columns=["body_pre", "body_post", "weight"])
    add("significant-only file, all rows", sig)
    add("significant-only file, both ends Traced", sig[sig.body_pre.isin(traced) & sig.body_post.isin(traced)])
    out = pd.DataFrame(rows)
    out["minus_ours"] = out.edges_ge5 - out.loc[1, "edges_ge5"]
    out.to_csv(ROOT / "reports" / "malecns_edges_ge5_variants.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
