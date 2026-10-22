"""Provisional EDA flags (is_optic, region_group, in_brain) on top of eda_shared v0.1.1.

TEMPORARY - delete this module's two extra rows when a new eda_shared version arrives.

Group decision (b), 14-15 Oct: when a MaleCNS neuron's region is Optic-unspecified(L/R)
or CentralBrain-unspecified, it keeps that label. eda_shared v0.1.1 does not know these
names and must not be edited, so we pass it a table = load_neuropil_table() plus two rows:
    Optic-unspecified          is_optic True,  region_group unknown
    CentralBrain-unspecified   is_optic False, region_group unknown
(eda_shared strips the "(L)"/"(R)" side marker, so one row covers both optic sides.)

Nerve cord (open in the group; our default): every primary ROI under the VNC branch of
the neuPrint male-cns:v1.0 ROI hierarchy, including VNC-unspecified. The list is written
to reports/malecns_nerve_cord_rois.txt and to the README.
"""
from __future__ import annotations

import pandas as pd

from eda_shared.eda_helpers import add_provisional_flags, load_neuropil_table

# --- delete from here when eda_shared knows the unspecified ROIs --------------------
EXTRA_ROWS = pd.DataFrame({
    "neuropil": ["Optic-unspecified", "CentralBrain-unspecified"],
    "full_name": ["optic lobe, outside named neuropils", "central brain, outside named neuropils"],
    "is_optic": [True, False],
    "region_group": ["unknown", "unknown"],
    "status": ["provisional", "provisional"],
    "basis": ["group decision 14-15 Oct (b)", "group decision 14-15 Oct (b)"],
}).astype({"neuropil": "string", "region_group": "string"}).set_index("neuropil")


def neuropil_table() -> pd.DataFrame:
    return pd.concat([load_neuropil_table(), EXTRA_ROWS])
# --- to here -------------------------------------------------------------------------


def nerve_cord_rois(hierarchy: dict) -> list[str]:
    """Primary ROIs ('*' in the neuPrint hierarchy) anywhere under the 'VNC' branch."""
    def find(d):
        for k, v in d.items():
            if k.rstrip("*") == "VNC":
                return v
            hit = find(v or {})
            if hit is not None:
                return hit
        return None

    found = []

    def walk(d):
        for k, v in (d or {}).items():
            if k.endswith("*"):
                found.append(k[:-1])
            walk(v)
    walk(find(hierarchy))
    return sorted(found)


def flag(neurons: pd.DataFrame, nerve_cord: list[str] | None = None) -> pd.DataFrame:
    """add_provisional_flags with our table; pass nerve_cord for malecns only."""
    return add_provisional_flags(neurons, table=neuropil_table(),
                                 nerve_cord=frozenset(nerve_cord or ()))
