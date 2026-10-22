"""Project configuration: pinned releases, paths, thresholds and the data schema as code.

Everything that decides *which* data we analyse lives here, so a result can be
reproduced by reading one file. Tokens are never stored here; they are read
from ``.env`` at runtime (see ``load_env``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded source files, untouched
CACHE_DIR = DATA_DIR / "cache"      # schema-conformant parquet tables
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_env() -> None:
    """Load API tokens from PROJECT_ROOT/.env into os.environ (never printed)."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Pinned releases
# ---------------------------------------------------------------------------
FAFB_RELEASE = "783"                 # FlyWire FAFB public materialization 783
FAFB_DATASTACK = "flywire_fafb_public"
FAFB_ZENODO_RECORD = "10676866"      # "FlyWire Whole-brain Connectome Connectivity Data 783.0"
FAFB_ZENODO_URL = f"https://zenodo.org/api/records/{FAFB_ZENODO_RECORD}/files/{{name}}/content"
# flywire_annotations repo (Schlegel et al. 2024), pinned to a commit for reproducibility.
FAFB_ANNOT_COMMIT = "a83b2776d60d5764cef36b927f5f9679c16c47a2"   # repo HEAD on 2026-10-08 (file last changed 2026-09-29)
FAFB_ANNOT_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                  f"{FAFB_ANNOT_COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
FAFB_FILES = {
    "connections": "proofread_connections_783.feather",
    "root_ids": "proofread_root_ids_783.npy",
    "neuropil_pre": "per_neuron_neuropil_count_pre_783.feather",
    "neuropil_post": "per_neuron_neuropil_count_post_783.feather",
}

MALECNS_SERVER = "neuprint.janelia.org"
MALECNS_DATASET = "male-cns:v1.0"    # newest public MaleCNS on neuPrint (checked 2026-10-08)
MALECNS_RELEASE = "v1.0"
MALECNS_STATUS = "Traced"            # neurons table = all bodies with status 'Traced'
MALECNS_BUCKET_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/"
                      "connectome-data/flat-connectome/{name}")
MALECNS_FILES = {
    # Janelia's official flat export of the same release; traced-to-traced only.
    "weights_traced": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}

RELEASES = {"fafb": FAFB_RELEASE, "malecns": MALECNS_RELEASE}

# Published reference figures used by summary.py
PUBLISHED = {
    "fafb": {"neurons": 139_255, "source": "Dorkenwald et al. 2024, Nature (FlyWire v783)"},
    "malecns": {"neurons": 167_000, "source": "neuPrint male-cns:v1.0 dataset description ('167k neurons')"},
}

# ---------------------------------------------------------------------------
# Fetch thresholds
# ---------------------------------------------------------------------------
MIN_SYN_DEFAULT = 1          # cache the full graph once; higher thresholds are filters of it
MIN_SYN_CACHED = (1, 5)      # thresholds written to cache by run_pipeline
ABS_THRESHOLDS = (1, 2, 3, 5, 10, 31)
REL_THRESHOLDS = (0.009, 0.01, 0.026)   # fraction of the target neuron's total input
NEUPRINT_PAGE_SIZE = 2_000   # source neurons per paged Cypher edge query

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------
DATASETS = ["fafb", "malecns", "manc", "hemibrain"]
SIDES = ["left", "right", "center", "unknown"]
# histamine added beyond the minimal list: MaleCNS predicts it for >2k neurons
# (photoreceptors etc.). Mapping it to 'unknown' would destroy real information.
NT_TYPES = ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin",
            "octopamine", "histamine", "unknown"]
# FlyWire super_class values (observed in v783 annotations) + values needed for
# the MaleCNS VNC and a few MaleCNS-only classes. Same list for both datasets.
SUPER_CLASSES = ["optic", "central", "sensory", "visual_projection", "ascending",
                 "descending", "sensory_ascending", "visual_centrifugal", "motor",
                 "endocrine",
                 # MaleCNS-only (no FAFB equivalent)
                 "vnc_intrinsic", "efferent", "enteric", "unknown"]
REGION_GROUPS = ["optic", "central_brain", "vnc", "unknown"]

# ---------------------------------------------------------------------------
# Source -> schema mappings (strict: recode() raises on anything unmapped)
# ---------------------------------------------------------------------------
SIDE_MAP = {
    "fafb": {"left": "left", "right": "right", "center": "center", "na": "unknown",
             None: "unknown"},
    # MaleCNS: somaSide first, rootSide as fallback for neurons without a soma (sensory)
    "malecns": {"L": "left", "R": "right", "M": "center", "unknown": "unknown",
                None: "unknown"},
}
NT_MAP = {
    "fafb": {"acetylcholine": "acetylcholine", "glutamate": "glutamate", "gaba": "gaba",
             "dopamine": "dopamine", "serotonin": "serotonin", "octopamine": "octopamine",
             None: "unknown"},
    "malecns": {"acetylcholine": "acetylcholine", "glutamate": "glutamate", "gaba": "gaba",
                "dopamine": "dopamine", "serotonin": "serotonin", "octopamine": "octopamine",
                "histamine": "histamine", "unclear": "unknown", None: "unknown"},
}
SUPERCLASS_MAP = {
    "fafb": {v: v for v in SUPER_CLASSES[:10]} | {None: "unknown"},
    "malecns": {
        "ol_intrinsic": "optic", "cb_intrinsic": "central", "vnc_intrinsic": "vnc_intrinsic",
        "visual_projection": "visual_projection", "visual_projection_tbc": "visual_projection",
        "cb_sensory": "sensory", "ol_sensory": "sensory", "vnc_sensory": "sensory",
        "cb_sensory_tbc": "sensory", "vnc_sensory_tbc": "sensory",
        "sensory_descending": "sensory",
        "sensory_ascending": "sensory_ascending", "sensory_ascending_tbc": "sensory_ascending",
        "ascending_neuron": "ascending", "descending_neuron": "descending",
        "visual_centrifugal": "visual_centrifugal",
        "cb_motor": "motor", "vnc_motor": "motor",
        "cb_endocrine": "endocrine", "vnc_endocrine": "endocrine",
        "cb_efferent": "efferent", "vnc_efferent": "efferent",
        "efferent_ascending": "efferent", "efferent_descending": "efferent",
        "ENS": "enteric", "vnc_tbc": "unknown", None: "unknown",
    },
}

# Neuropil / ROI names -> one shared vocabulary (FlyWire style: NAME_L / NAME_R).
OPTIC_NEUROPILS = {"LA", "ME", "AME", "LO", "LOP", "Optic-unspecified"}
MALECNS_MB_MAP = {"CA": "MB_CA", "PED": "MB_PED", "aL": "MB_VL", "a'L": "MB_VL",
                  "bL": "MB_ML", "b'L": "MB_ML", "gL": "MB_ML"}
VNC_PREFIXES = ("LegNp", "NTct", "WTct", "HTct", "IntTct", "LTct", "ANm", "mVAC", "Ov",
                "VNC-unspecified", "CV-unspecified", "ADMN", "AbN", "CvN", "DMetaN",
                "DProN", "MesoAN", "MesoLN", "MetaLN", "PDMN", "PrN", "ProAN", "ProCN",
                "ProLN", "VProN")


def normalize_region(name: str | None, dataset: str) -> str:
    """Map a raw neuropil/ROI name to the shared region vocabulary."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return "unknown"
    if dataset == "fafb":
        return name                       # already FlyWire style, e.g. 'ME_L', 'GNG'
    m = re.fullmatch(r"(.+?)(?:\((UTct-T\d)\))?(?:\((T\d)\))?(?:\(([LR])\))?", name)
    base, _, seg, side = m.groups()
    base = MALECNS_MB_MAP.get(base, base)
    if seg:
        base = f"{base}_{seg}"
    return f"{base}_{side}" if side else base


def region_group(region: str) -> str:
    if region == "unknown":
        return "unknown"
    base = re.sub(r"_[LR]$", "", region)
    if base in OPTIC_NEUROPILS:
        return "optic"
    if region.startswith(VNC_PREFIXES):
        return "vnc"
    return "central_brain"


# ---------------------------------------------------------------------------
# Schema as code
# ---------------------------------------------------------------------------
# dtype strings are pandas dtypes; "category" columns carry a fixed vocabulary.
NEURONS_SCHEMA = {
    "neuron_id":     {"dtype": "int64",    "nullable": False, "key": True},
    "dataset":       {"dtype": "category", "nullable": False, "vocab": DATASETS},
    "cell_type":     {"dtype": "string",   "nullable": True},
    "super_class":   {"dtype": "category", "nullable": False, "vocab": SUPER_CLASSES},
    "side":          {"dtype": "category", "nullable": False, "vocab": SIDES},
    "region":        {"dtype": "category", "nullable": False},   # open vocab, 'unknown' if no synapses
    "is_optic":      {"dtype": "bool",     "nullable": False},
    "nt_type":       {"dtype": "category", "nullable": False, "vocab": NT_TYPES},
    "nt_confidence": {"dtype": "float32",  "nullable": True},
    "n_pre":         {"dtype": "int32",    "nullable": False},   # output synapse count
    "n_post":        {"dtype": "int32",    "nullable": False},   # input synapse count
    "flywire_type":  {"dtype": "string",   "nullable": True},
    "manc_type":     {"dtype": "string",   "nullable": True},
    "hemibrain_type": {"dtype": "string",  "nullable": True},
    "release":       {"dtype": "string",   "nullable": False},
    # extra (nullable) columns
    "cell_class":     {"dtype": "string",   "nullable": True},
    "cell_sub_class": {"dtype": "string",   "nullable": True},
    "super_class_raw": {"dtype": "string",  "nullable": True},
    "region_group":   {"dtype": "category", "nullable": False, "vocab": REGION_GROUPS},
    "optic_fraction": {"dtype": "float32",  "nullable": True},   # share of synapses in optic neuropils
}
EDGES_SCHEMA = {
    "pre_id":          {"dtype": "int64",    "nullable": False, "key": True},
    "post_id":         {"dtype": "int64",    "nullable": False, "key": True},
    "syn_count":       {"dtype": "int32",    "nullable": False},
    "dataset":         {"dtype": "category", "nullable": False, "vocab": DATASETS},
    "release":         {"dtype": "string",   "nullable": False},
    "min_syn_fetched": {"dtype": "int32",    "nullable": False},
}
SCHEMA = {"neurons": NEURONS_SCHEMA, "edges": EDGES_SCHEMA}


class UnmappedValueError(ValueError):
    pass


def recode(series: pd.Series, mapping: dict, name: str = "") -> pd.Series:
    """Map raw values through ``mapping``; raise if any value has no mapping.

    Missing values (None/NaN/pd.NA) are looked up under the ``None`` key.
    """
    s = series.astype(object).where(series.notna(), None)
    observed = set(s.unique())
    missing = {v for v in observed if v not in mapping}
    if missing:
        raise UnmappedValueError(f"{name}: unmapped values {sorted(map(str, missing))}")
    return s.map(lambda v: mapping[v])


def apply_schema(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Order columns and cast dtypes exactly as the schema says."""
    spec = SCHEMA[table]
    out = pd.DataFrame(index=df.index)
    for col, s in spec.items():
        x = df[col]
        if s["dtype"] == "category":
            cats = s.get("vocab") or sorted(x.dropna().unique())
            out[col] = pd.Categorical(x, categories=cats)
        else:
            out[col] = x.astype(s["dtype"])
    return out.reset_index(drop=True)
