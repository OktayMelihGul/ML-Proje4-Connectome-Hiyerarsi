import numpy as np
import pandas as pd
import pytest

from src.region_null import randomise_edges_by_block, region_blocks


def toy_graph(n=300, p=0.04, seed=1):
    rng = np.random.default_rng(seed)
    region = pd.Series(rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2]), index=np.arange(n) + 1000)
    pre, post = np.nonzero(rng.random((n, n)) < p)
    keep = pre != post
    edges = pd.DataFrame({"pre_id": pre[keep] + 1000, "post_id": post[keep] + 1000, "syn_count": 10})
    return edges, region


def per_region_degrees(edges, region):
    e = edges.assign(rpre=region.reindex(edges.pre_id).to_numpy(), rpost=region.reindex(edges.post_id).to_numpy())
    out_to = e.groupby(["pre_id", "rpost"]).size().sort_index()
    in_from = e.groupby(["post_id", "rpre"]).size().sort_index()
    return out_to, in_from


@pytest.fixture(scope="module")
def run():
    edges, region = toy_graph()
    block = region_blocks(edges, region)
    out, stats, per_block = randomise_edges_by_block(edges, block, seed=0, return_stats=True)
    return edges, region, block, out, stats, per_block


def test_per_region_degrees_unchanged(run):
    edges, region, _, out, _, _ = run
    o1, i1 = per_region_degrees(edges, region)
    o2, i2 = per_region_degrees(out, region)
    pd.testing.assert_series_equal(o1, o2)
    pd.testing.assert_series_equal(i1, i2)


def test_total_degrees_and_blocks_unchanged(run):
    edges, region, block, out, _, _ = run
    pd.testing.assert_series_equal(edges.pre_id.value_counts().sort_index(), out.pre_id.value_counts().sort_index())
    pd.testing.assert_series_equal(edges.post_id.value_counts().sort_index(), out.post_id.value_counts().sort_index())
    assert (region_blocks(out, region).to_numpy() == block.to_numpy()).all()


def test_no_self_loops_or_duplicates(run):
    _, _, _, out, _, _ = run
    assert (out.pre_id != out.post_id).all()
    assert not out.duplicated(["pre_id", "post_id"]).any()


def test_edges_actually_rewired_and_target_reached(run):
    edges, _, _, out, stats, per_block = run
    old = set(zip(edges.pre_id, edges.post_id))
    new = set(zip(out.pre_id, out.post_id))
    assert len(old & new) / len(old) < 0.5
    assert stats["edges_unreached"] == 0 and per_block.loc[per_block.swappable, "reached_target"].all()


def test_same_seed_same_output(run):
    edges, _, block, out, _, _ = run
    again = randomise_edges_by_block(edges, block, seed=0)
    other = randomise_edges_by_block(edges, block, seed=1)
    pd.testing.assert_frame_equal(out, again)
    assert not out.equals(other)


def test_rejects_bad_input():
    edges = pd.DataFrame({"pre_id": [1, 2, 3], "post_id": [2, 2, 1]})
    with pytest.raises(ValueError):
        randomise_edges_by_block(edges, ["x"] * 3, seed=0)
    with pytest.raises(ValueError):
        randomise_edges_by_block(edges.iloc[:2], ["x"], seed=0)


def test_block_diagnostics(run):
    from src.region_null import block_diagnostics
    edges, _, block, out, _, _ = run
    diag = block_diagnostics(edges, out, block)
    assert diag.edges.sum() == len(edges)
    assert ((diag.density > 0) & (diag.density <= 1)).all()
    assert diag.share_changed.between(0, 1).all() and (block_diagnostics(edges, edges, block).share_changed == 0).all()
