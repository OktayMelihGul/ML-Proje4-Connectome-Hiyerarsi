"""Where the raw files come from (exact bulk-download URLs) and how they are fetched.

The pinned releases themselves (783, v3.2.0, buhmann, male-cns:v1.0, Traced, 0.5)
live in the root config.py; this module only turns them into addresses.
Tokens are read from the git-ignored .env and never printed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FAFB_RAW = RAW_DIR / f"fafb_{config.FAFB_MATERIALIZATION}"
MALECNS_RAW = RAW_DIR / "malecns_v1.0"
DOWNLOAD_LOG = RAW_DIR / "download_log.json"

# FAFB 783: Zenodo record "FlyWire Whole-brain Connectome Connectivity Data 783.0".
# proofread_connections_783 is the Codex per-neuropil table on Buhmann synapses
# (one row per pre, post, neuropil; unthresholded, min syn_count 1).
FAFB_ZENODO_RECORD = "10676866"
FAFB_ZENODO_URL = "https://zenodo.org/records/10676866/files/{name}?download=1"
FAFB_FILES = {
    "connections": "proofread_connections_783.feather",
    "root_ids": "proofread_root_ids_783.npy",
}
# flywire_annotations, tag v3.2.0 (= commit a83b2776d60d5764cef36b927f5f9679c16c47a2).
FAFB_ANNOT_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                  f"{config.FAFB_ANNOTATIONS_TAG}/supplemental_files/"
                  "Supplemental_file1_neuron_annotations.tsv")
FAFB_ANNOT_FILE = "Supplemental_file1_neuron_annotations.tsv"

# MaleCNS v1.0: Janelia's flat-connectome export of the same release.
MALECNS_SERVER = "neuprint.janelia.org"
MALECNS_BUCKET_URL = ("https://storage.googleapis.com/flyem-male-cns/v1.0/"
                      "connectome-data/flat-connectome/{name}")
MALECNS_FILES = {
    # Traced -> Traced neuron weights, synapse confidence >= 0.5, summed over ROIs.
    "weights_traced": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    # One row per synaptic partner pair (Traced -> Traced, conf >= 0.5) with primary_post ROI.
    "partners_traced": "syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather",
}


def bulk_urls() -> dict[str, str]:
    """Every bulk-download address the pipeline uses (printed into the README)."""
    out = {name: FAFB_ZENODO_URL.format(name=name) for name in FAFB_FILES.values()}
    out[FAFB_ANNOT_FILE] = FAFB_ANNOT_URL
    out |= {name: MALECNS_BUCKET_URL.format(name=name) for name in MALECNS_FILES.values()}
    return out


def download(url: str, dest: Path, chunk: int = 1 << 22) -> Path:
    """Stream `url` to `dest` once (skipped if the file exists); log size and time."""
    import requests

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
    logs = json.loads(DOWNLOAD_LOG.read_text()) if DOWNLOAD_LOG.exists() else {}
    logs[dest.name] = {"url": url, "bytes": dest.stat().st_size,
                       "seconds": round(time.perf_counter() - t0, 1)}
    DOWNLOAD_LOG.write_text(json.dumps(logs, indent=2))
    return dest


def fafb_file(key: str) -> Path:
    name = FAFB_FILES[key]
    return download(FAFB_ZENODO_URL.format(name=name), FAFB_RAW / name)


def fafb_annotations_file() -> Path:
    return download(FAFB_ANNOT_URL, FAFB_RAW / FAFB_ANNOT_FILE)


def malecns_file(key: str) -> Path:
    name = MALECNS_FILES[key]
    return download(MALECNS_BUCKET_URL.format(name=name), MALECNS_RAW / name)


def neuprint_client():
    """neuPrint client for male-cns:v1.0; the token comes from .env."""
    from dotenv import load_dotenv
    from neuprint import Client

    load_dotenv(ROOT / ".env", override=False)
    return Client(MALECNS_SERVER, dataset=config.MALECNS_DATASET,
                  token=os.environ["NEUPRINT_APPLICATION_CREDENTIALS"])
