"""Degree-preserving randomiser for a directed edge table (v0.1-eda).

Used by C6 (motifs in the mushroom-body subgraph) and D-7 (reciprocity).
Every neuron keeps its exact in-degree and out-degree; the result has no
self-connections and no duplicate edges; the same seed gives the same graph.

Method: double-edge swaps (Maslov & Sneppen 2002). Two edges a->b and c->d
become a->d and c->b. Swaps are proposed in rounds: each round pairs all
edges at random and accepts every swap that creates no self-connection and
no edge that already exists or is proposed twice. This batched scheme is
fast on whole-brain graphs; it is a standard practical null model, not an
exactly uniform sampler of all graphs with these degrees.

Weights: every column other than post_id stays on its row, so syn_count
follows the presynaptic neuron. Out-strength is preserved; in-strength is
not. For binary analyses (C6, D-7) this does not matter.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def randomise_edges(edges: pd.DataFrame, seed: int, swaps_per_edge: float = 10.0,
                    max_rounds: int | None = None, return_stats: bool = False):
    """Return a randomised copy of `edges` (columns pre_id, post_id, ...).

    Threshold first, then randomise: pass the edge table at the threshold
    you analyse. Runs until about `swaps_per_edge` x E swaps are accepted.
    """
    if edges["pre_id"].isna().any() or edges["post_id"].isna().any():
        raise ValueError("edges: missing ids")
    pre_ids = edges["pre_id"].to_numpy(np.int64)
    post_ids = edges["post_id"].to_numpy(np.int64)
    n_edges = len(edges)
    if n_edges < 2:
        raise ValueError("edges: need at least two edges")
    if (pre_ids == post_ids).any():
        raise ValueError("edges: drop self-connections first")

    nodes, inverse = np.unique(np.concatenate([pre_ids, post_ids]), return_inverse=True)
    n = np.int64(len(nodes))
    pre = inverse[:n_edges].astype(np.int64)
    post = inverse[n_edges:].astype(np.int64)
    if len(np.unique(pre * n + post)) != n_edges:
        raise ValueError("edges: duplicate (pre_id, post_id) rows")

    rng = np.random.default_rng(seed)
    target = int(swaps_per_edge * n_edges)
    max_rounds = max_rounds or int(20 * swaps_per_edge + 20)
    m = n_edges // 2
    accepted = 0
    rounds = 0
    while accepted < target and rounds < max_rounds:
        rounds += 1
        keys = np.sort(pre * n + post)
        perm = rng.permutation(n_edges)
        a, b = perm[:m], perm[m:2 * m]
        k1 = pre[a] * n + post[b]          # a's source -> b's target
        k2 = pre[b] * n + post[a]          # b's source -> a's target
        ok = ((pre[a] != post[b]) & (pre[b] != post[a])          # no self-connection
              & (pre[a] != pre[b]) & (post[a] != post[b]))        # a real change
        ok &= ~_contains(keys, k1) & ~_contains(keys, k2)          # no existing edge
        proposed = np.concatenate([k1[ok], k2[ok]])
        uniq, counts = np.unique(proposed, return_counts=True)
        clash = uniq[counts > 1]
        if clash.size:                                              # no edge proposed twice
            ok[ok] = ~(np.isin(k1[ok], clash) | np.isin(k2[ok], clash))
        a, b = a[ok], b[ok]
        post[a], post[b] = post[b], post[a].copy()
        accepted += int(ok.sum())
        if ok.sum() == 0:
            break

    if accepted < target:
        warnings.warn(f"randomise_edges: only {accepted:,} of {target:,} swaps accepted "
                      f"in {rounds} rounds; the graph may be too dense to mix fully. "
                      f"Report this with the result.", stacklevel=2)
    out = edges.copy()
    out["post_id"] = nodes[post]
    if return_stats:
        return out, {"accepted_swaps": accepted, "target_swaps": target,
                     "rounds": rounds, "reached_target": accepted >= target}
    return out


def _contains(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_keys, query)
    idx[idx == len(sorted_keys)] = 0
    return sorted_keys[idx] == query


def reciprocity(edges: pd.DataFrame) -> float:
    """Share of edges i->j for which j->i also exists (used in D-7)."""
    fwd = pd.MultiIndex.from_arrays([edges["pre_id"], edges["post_id"]])
    rev = pd.MultiIndex.from_arrays([edges["post_id"], edges["pre_id"]])
    return float(rev.isin(fwd).mean())
