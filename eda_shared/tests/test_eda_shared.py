"""Run with:  pytest tests/  (from the folder holding the shared files)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eda_helpers import (add_provisional_flags, basic_features,  # noqa: E402
                         canonical_neuropil, load_neuropil_table, relative_weight)
from randomiser import randomise_edges, reciprocity  # noqa: E402

# The 78 FlyWire neuropils (fafbseg volume_name_dict, without UNASGD).
FLYWIRE = """AL_L AL_R AME_L AME_R AMMC_L AMMC_R AOTU_L AOTU_R ATL_L ATL_R AVLP_L AVLP_R
BU_L BU_R CAN_L CAN_R CRE_L CRE_R EB EPA_L EPA_R FB FLA_L FLA_R GA_L GA_R GNG GOR_L GOR_R
IB_L IB_R ICL_L ICL_R IPS_L IPS_R LA_L LA_R LAL_L LAL_R LH_L LH_R LO_L LO_R LOP_L LOP_R
MB_CA_L MB_CA_R MB_ML_L MB_ML_R MB_PED_L MB_PED_R MB_VL_L MB_VL_R ME_L ME_R NO OCG PB
PLP_L PLP_R PRW PVLP_L PVLP_R SAD SCL_L SCL_R SIP_L SIP_R SLP_L SLP_R SMP_L SMP_R SPS_L
SPS_R VES_L VES_R WED_L WED_R""".split()


# ---------- neuropil table ----------

def test_table_covers_every_flywire_neuropil():
    table = load_neuropil_table()
    assert len(FLYWIRE) == 78
    mapped = canonical_neuropil(pd.Series(FLYWIRE), table)
    assert mapped.isin(table.index).all()


def test_optic_lobe_is_exactly_five_neuropils():
    table = load_neuropil_table()
    assert set(table.index[table["is_optic"]]) == {"LA", "ME", "AME", "LO", "LOP"}


def test_region_group_vocabulary():
    table = load_neuropil_table()
    assert set(table["region_group"]) == {"sensory_periphery", "motor_periphery", "higher_order"}


def test_neuprint_style_and_aliases():
    table = load_neuropil_table()
    out = canonical_neuropil(pd.Series(["ME(R)", "a'L(L)", "PED(R)", "unknown", "UNASGD"]), table)
    assert list(out) == ["ME", "MB_VL", "MB_PED", "unknown", "unknown"]


def test_unmapped_value_raises():
    table = load_neuropil_table()
    with pytest.raises(ValueError, match="unmapped"):
        canonical_neuropil(pd.Series(["ME_L", "XYZ(R)"]), table)


def test_nerve_cord_values_are_explicit():
    table = load_neuropil_table()
    neurons = pd.DataFrame({"neuron_id": [1, 2, 3], "region": ["LO_R", "VNC_X", "unknown"]})
    flagged = add_provisional_flags(neurons, table, nerve_cord={"VNC_X"})
    assert list(flagged["is_optic"]) == [True, False, False]
    assert list(flagged["in_brain"]) == [True, False, False]
    assert list(flagged["region_group"]) == ["sensory_periphery", "unknown", "unknown"]


# ---------- four features, toy graph computed by hand ----------
#   1 -> 2 (12 syn), 1 -> 3 (3 syn), 2 -> 3 (10 syn), 3 -> 1 (1 syn); neuron 4 isolated.
#   tau = 10 keeps 1->2 and 2->3.
#   in_degree:  1:0 2:1 3:1 4:0      out_degree: 1:1 2:1 3:0 4:0
#   total_in:   1:1 2:12 3:13 4:0    total_out:  1:15 2:10 3:1 4:0

def toy():
    neurons = pd.DataFrame({"neuron_id": [1, 2, 3, 4]})
    edges = pd.DataFrame({"pre_id": [1, 1, 2, 3], "post_id": [2, 3, 3, 1],
                          "syn_count": [12, 3, 10, 1], "min_syn_fetched": 1})
    return neurons, edges


def test_basic_features_by_hand():
    neurons, edges = toy()
    f = basic_features(neurons, edges, tau=10).set_index("neuron_id")
    assert f["in_degree"].to_dict() == {1: 0, 2: 1, 3: 1, 4: 0}
    assert f["out_degree"].to_dict() == {1: 1, 2: 1, 3: 0, 4: 0}
    assert f["total_in_syn"].to_dict() == {1: 1, 2: 12, 3: 13, 4: 0}
    assert f["total_out_syn"].to_dict() == {1: 15, 2: 10, 3: 1, 4: 0}


def test_relative_weight_by_hand():
    _, edges = toy()
    w = relative_weight(edges)
    assert np.allclose(w, [12 / 12, 3 / 13, 10 / 13, 1 / 1])


def test_totals_refuse_a_thresholded_table():
    neurons, edges = toy()
    edges["min_syn_fetched"] = 5
    with pytest.raises(ValueError):
        basic_features(neurons, edges)


# ---------- randomiser ----------

def random_graph(n_nodes=3000, n_edges=30000, seed=1):
    rng = np.random.default_rng(seed)
    # heavy-tailed out- and in-propensities, like a connectome
    w_out = rng.pareto(1.5, n_nodes) + 1
    w_in = rng.pareto(1.5, n_nodes) + 1
    pre = rng.choice(n_nodes, 3 * n_edges, p=w_out / w_out.sum())
    post = rng.choice(n_nodes, 3 * n_edges, p=w_in / w_in.sum())
    df = pd.DataFrame({"pre_id": pre, "post_id": post})
    df = df[df.pre_id != df.post_id].drop_duplicates().head(n_edges)
    df[["pre_id", "post_id"]] += 720575940600000000      # FlyWire-sized int64 ids
    df["syn_count"] = rng.integers(10, 200, len(df))
    return df.reset_index(drop=True)


def degrees(df):
    return (df.groupby("pre_id").size().sort_index(), df.groupby("post_id").size().sort_index())


def test_degrees_preserved_and_graph_valid():
    g = random_graph()
    r, stats = randomise_edges(g, seed=0, return_stats=True)
    assert stats["reached_target"]
    assert degrees(r)[0].equals(degrees(g)[0]) and degrees(r)[1].equals(degrees(g)[1])
    assert (r.pre_id != r.post_id).all()
    assert not r.duplicated(["pre_id", "post_id"]).any()
    assert r["pre_id"].dtype == np.int64 and r["post_id"].dtype == np.int64


def test_out_strength_preserved():
    g = random_graph()
    r = randomise_edges(g, seed=0)
    assert g.groupby("pre_id")["syn_count"].sum().equals(r.groupby("pre_id")["syn_count"].sum())


def test_seed_determinism():
    g = random_graph()
    assert randomise_edges(g, seed=7).equals(randomise_edges(g, seed=7))
    assert not randomise_edges(g, seed=7).equals(randomise_edges(g, seed=8))


def test_graph_is_actually_mixed():
    g = random_graph()
    r = randomise_edges(g, seed=0)
    same = pd.merge(g, r, on=["pre_id", "post_id"]).shape[0] / len(g)
    assert same < 0.2


def test_warns_when_graph_cannot_mix():
    # a complete directed triangle has exactly one graph with its degrees
    tri = pd.DataFrame({"pre_id": [1, 2, 3, 2, 3, 1], "post_id": [2, 3, 1, 1, 2, 3]})
    with pytest.warns(UserWarning, match="swaps accepted"):
        randomise_edges(tri, seed=0)


def test_reciprocity_falls_to_null_level():
    g = random_graph()
    back = g.rename(columns={"pre_id": "post_id", "post_id": "pre_id"}).sample(frac=0.3, random_state=0)
    g = pd.concat([g, back]).drop_duplicates(["pre_id", "post_id"]).reset_index(drop=True)
    assert reciprocity(g) > 0.3
    assert reciprocity(randomise_edges(g, seed=0)) < reciprocity(g) / 2
