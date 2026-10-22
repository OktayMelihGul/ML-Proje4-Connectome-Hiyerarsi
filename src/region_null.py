"""Region-preserving null model for reciprocity (EDA D-7b).

Each edge belongs to a block = (region of pre neuron, region of post neuron). Swaps pair only edges
of the same block (a->b, c->d become a->d, c->b), so every neuron keeps its exact out-degree towards
each target region and its exact in-degree from each source region. Partners are re-drawn only among
neurons of those regions. Same batched swap logic and checks as eda_shared.randomiser.randomise_edges
(no self-connection, no existing edge, no edge proposed twice), checked against the global edge set.
eda_shared is not changed; this is an additional null beside the plan's degree-preserving one.
"""
import warnings

import numpy as np
import pandas as pd


def _contains(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_keys, keys)
    idx[idx == len(sorted_keys)] = 0
    return sorted_keys[idx] == keys


def region_blocks(edges: pd.DataFrame, region: pd.Series) -> pd.Series:
    """Block label 'pre_region|post_region' per edge; region is indexed by neuron_id."""
    rp = region.reindex(edges["pre_id"]).astype(str).to_numpy()
    rq = region.reindex(edges["post_id"]).astype(str).to_numpy()
    return pd.Series(np.char.add(np.char.add(rp, "|"), rq), index=edges.index, name="block")


def randomise_edges_by_block(edges: pd.DataFrame, block, seed: int, swaps_per_edge: float = 10.0,
                             max_rounds: int | None = None, return_stats: bool = False):
    """Return a copy of `edges` (pre_id, post_id, ...) with post_id swapped only within blocks.

    `block`: one label per edge row. Each block with at least two edges runs until about
    `swaps_per_edge` x (its edge count) swaps are accepted or `max_rounds` is reached.
    """
    if edges["pre_id"].isna().any() or edges["post_id"].isna().any():
        raise ValueError("edges: missing ids")
    pre_ids = edges["pre_id"].to_numpy(np.int64)
    post_ids = edges["post_id"].to_numpy(np.int64)
    n_edges = len(edges)
    if len(block) != n_edges:
        raise ValueError("block: one label per edge")
    if (pre_ids == post_ids).any():
        raise ValueError("edges: drop self-connections first")

    nodes, inverse = np.unique(np.concatenate([pre_ids, post_ids]), return_inverse=True)
    n = np.int64(len(nodes))
    pre = inverse[:n_edges].astype(np.int64)
    post = inverse[n_edges:].astype(np.int64)
    if len(np.unique(pre * n + post)) != n_edges:
        raise ValueError("edges: duplicate (pre_id, post_id) rows")

    labels, blk = np.unique(np.asarray(block).astype(str), return_inverse=True)
    size = np.bincount(blk, minlength=len(labels))
    # a block whose edges all share one sender or one receiver has no other arrangement
    n_pre = pd.Series(pre).groupby(blk).nunique().reindex(range(len(labels))).to_numpy()
    n_post = pd.Series(post).groupby(blk).nunique().reindex(range(len(labels))).to_numpy()
    fixed = (n_pre == 1) | (n_post == 1)
    target = np.floor(swaps_per_edge * size).astype(np.int64)
    swappable = size >= 2
    accepted = np.zeros(len(labels), np.int64)
    rng = np.random.default_rng(seed)
    max_rounds = max_rounds or int(20 * swaps_per_edge + 20)
    rounds = 0
    while rounds < max_rounds:
        active_blocks = swappable & (accepted < target)
        if not active_blocks.any():
            break
        rounds += 1
        act = np.flatnonzero(active_blocks[blk])
        order = act[np.lexsort((rng.random(len(act)), blk[act]))]   # random order within each block
        m = len(order) // 2
        a, b = order[0:2 * m:2], order[1:2 * m:2]
        same = blk[a] == blk[b]                                     # drop the pair straddling two blocks
        a, b = a[same], b[same]
        keys = np.sort(pre * n + post)
        k1 = pre[a] * n + post[b]
        k2 = pre[b] * n + post[a]
        ok = ((pre[a] != post[b]) & (pre[b] != post[a])
              & (pre[a] != pre[b]) & (post[a] != post[b]))
        ok &= ~_contains(keys, k1) & ~_contains(keys, k2)
        proposed = np.concatenate([k1[ok], k2[ok]])
        uniq, counts = np.unique(proposed, return_counts=True)
        clash = uniq[counts > 1]
        if clash.size:
            ok[ok] = ~(np.isin(k1[ok], clash) | np.isin(k2[ok], clash))
        a, b = a[ok], b[ok]
        post[a], post[b] = post[b], post[a].copy()
        accepted += np.bincount(blk[a], minlength=len(labels))

    reached = accepted >= target
    per_block = pd.DataFrame({"block": labels, "edges": size, "accepted_swaps": accepted,
                              "target_swaps": target, "reached_target": reached & swappable,
                              "swappable": swappable, "senders": n_pre, "receivers": n_post,
                              "no_alternative": fixed})
    unreached_edges = int(size[swappable & ~reached].sum())
    unreached_fixed = int(size[swappable & ~reached & fixed].sum())
    if unreached_edges:
        warnings.warn(f"randomise_edges_by_block: {int((swappable & ~reached).sum())} blocks "
                      f"({unreached_edges:,} edges, {unreached_edges / n_edges:.1%}) did not reach their swap "
                      f"target in {rounds} rounds. Report this with the result.", stacklevel=2)
    out = edges.copy()
    out["post_id"] = nodes[post]
    if return_stats:
        stats = {"rounds": rounds, "accepted_swaps": int(accepted.sum()), "target_swaps": int(target[swappable].sum()),
                 "blocks": len(labels), "blocks_unswappable": int((~swappable).sum()),
                 "edges_unswappable": int(size[~swappable].sum()),
                 "blocks_unreached": int((swappable & ~reached).sum()), "edges_unreached": unreached_edges,
                 "share_edges_unreached": unreached_edges / n_edges,
                 "edges_unreached_no_alternative": unreached_fixed}
        return out, stats, per_block
    return out


def block_diagnostics(real: pd.DataFrame, randomised: pd.DataFrame, block) -> pd.DataFrame:
    """Per block: edges, distinct senders and receivers, density = edges / (senders x receivers),
    and the share of the block's randomised edges that are not edges of the real graph."""
    real_keys = pd.MultiIndex.from_arrays([real["pre_id"], real["post_id"]])
    new_keys = pd.MultiIndex.from_arrays([randomised["pre_id"], randomised["post_id"]])
    d = pd.DataFrame({"block": np.asarray(block).astype(str), "pre": real["pre_id"].to_numpy(),
                      "post": real["post_id"].to_numpy(), "changed": ~new_keys.isin(real_keys)})
    g = d.groupby("block")
    out = pd.DataFrame({"edges": g.size(), "senders": g.pre.nunique(), "receivers": g.post.nunique(),
                        "share_changed": g.changed.mean()})
    out["density"] = out.edges / (out.senders * out.receivers)
    return out.reset_index()
