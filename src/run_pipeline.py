"""Step 0: build (or read) the six cached tables, validate them and write the count check.

    python -m src.run_pipeline [--force-refresh]

Writes reports/step0.md, reports/step0.csv, reports/build_log.json,
reports/malecns_nerve_cord_rois.txt and reports/source_columns.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

import config
from . import fetch
from .provisional import flag, nerve_cord_rois
from .sources import ROOT
from .validate import assert_interchangeable, validate_edges, validate_neurons

REPORTS = ROOT / "reports"
EXPECTED = {
    "fafb neurons": 139_248,
    "fafb edges with syn_count >= 5": 2_700_513,
    "malecns neurons": 165_122,
    "malecns edges with syn_count >= 5": None,
    "self-connections dropped (fafb / malecns)": "0 / 101",
}


def malecns_brain(neurons: pd.DataFrame) -> pd.DataFrame:
    return flag(neurons, nerve_cord=nerve_cord_rois(fetch.roi_hierarchy()))


def main(force_refresh: bool = False) -> dict:
    t0 = time.perf_counter()
    tabs = {"fafb": fetch.fafb_tables(force_refresh), "malecns": fetch.malecns_tables(force_refresh)}
    build_seconds = round(time.perf_counter() - t0, 1)

    # All validators on every table (also when read from cache) + assert_interchangeable.
    checks = []
    for ds, t in tabs.items():
        checks.append((ds, "neurons", validate_neurons(t["neurons"])))
        checks.append((ds, "edges", validate_edges(t["edges"], t["neurons"])))
        checks.append((ds, "edgeroi", validate_edges(t["edgeroi"], t["neurons"], roi=True)))
    for table in ("neurons", "edges", "edgeroi"):
        assert_interchangeable(tabs["fafb"][table], tabs["malecns"][table])

    log = json.loads(fetch.LOG_PATH.read_text())
    flagged = {"fafb": flag(tabs["fafb"]["neurons"]), "malecns": malecns_brain(tabs["malecns"]["neurons"])}
    vnc = nerve_cord_rois(fetch.roi_hierarchy())
    (REPORTS / "malecns_nerve_cord_rois.txt").write_text("\n".join(vnc) + "\n")

    m = flagged["malecns"]
    unspecified = m["region"].astype(str)
    ours = {
        "fafb neurons": len(tabs["fafb"]["neurons"]),
        "fafb edges with syn_count >= 5": int((tabs["fafb"]["edges"]["syn_count"] >= 5).sum()),
        "malecns neurons": len(tabs["malecns"]["neurons"]),
        "malecns edges with syn_count >= 5": int((tabs["malecns"]["edges"]["syn_count"] >= 5).sum()),
        "self-connections dropped (fafb / malecns)":
            f"{log['fafb']['self_connections_dropped_edges']} / {log['malecns']['self_connections_dropped_edges']}",
    }
    rows = [{"table": k, "expected": EXPECTED[k], "ours": v} for k, v in ours.items()]
    extra = {
        "fafb edges (syn_count >= 1)": len(tabs["fafb"]["edges"]),
        "fafb edgeroi rows": len(tabs["fafb"]["edgeroi"]),
        "malecns edges (syn_count >= 1)": len(tabs["malecns"]["edges"]),
        "malecns edgeroi rows": len(tabs["malecns"]["edgeroi"]),
        "malecns brain set (in_brain)": int(m["in_brain"].sum()),
        "malecns region = CentralBrain-unspecified": int((unspecified == "CentralBrain-unspecified").sum()),
        "malecns region = Optic-unspecified(L)": int((unspecified == "Optic-unspecified(L)").sum()),
        "malecns region = Optic-unspecified(R)": int((unspecified == "Optic-unspecified(R)").sum()),
        "malecns region = VNC-unspecified (nerve cord)": int((unspecified == "VNC-unspecified").sum()),
        "malecns region = unknown (no primary-ROI synapse)": int((unspecified == "unknown").sum()),
        "fafb region = UNASGD": int((tabs["fafb"]["neurons"]["region"] == "UNASGD").sum()),
        "fafb region = unknown (no synapse)": int((tabs["fafb"]["neurons"]["region"] == "unknown").sum()),
        "fafb region ties (alphabetical)": log["fafb"]["region_ties_alphabetical"],
        "malecns region ties (alphabetical)": log["malecns"]["region_ties_alphabetical"],
    }
    pd.DataFrame(rows).to_csv(REPORTS / "step0.csv", index=False)
    lines = ["# Step 0 count check", "",
             f"Built/read in {build_seconds} s. All 8 validators passed on neurons, edges and edgeroi "
             "of both datasets; assert_interchangeable passed for all three table pairs.", "",
             "| Table | Expected | Ours |", "|---|---|---|"]
    lines += [f"| {r['table']} | {r['expected'] if r['expected'] is not None else 'one agreed number'} | "
              f"{r['ours']:,} |" if isinstance(r["ours"], int) else
              f"| {r['table']} | {r['expected']} | {r['ours']} |" for r in rows]
    lines += ["", "| Other count | Value |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in extra.items()]
    lines += ["", "## Checks run", ""]
    lines += [f"- {ds} {table}: {', '.join(c)}" for ds, table, c in checks]
    lines += ["", "## Build log", "", "```json", json.dumps(log, indent=2, default=str), "```", ""]
    (REPORTS / "step0.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return tabs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-refresh", action="store_true")
    main(ap.parse_args().force_refresh)
