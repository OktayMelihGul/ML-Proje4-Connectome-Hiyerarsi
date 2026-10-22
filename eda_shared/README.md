# EDA shared files — v0.1.1-eda

Everything the five workstreams must share before the week 3 session. Copy this folder unchanged into the root of your repository, next to `config.py`. Do not edit it locally: report a problem to the group, and the fix comes back as a new version for everyone.

```
<your-repo>/
  config.py                # Data Conventions, unchanged
  eda_shared/              # this folder, unchanged
  src/                     # your fetch, cache and validator code
  data/cache/              # in .gitignore
  notebooks/
    00_integration.ipynb
    eda_<A-E>.ipynb        # your workstream
  figures/eda/             # committed
  reports/eda_<A-E>.md     # one-page summary
  README.md
```

Notebooks run from `notebooks/`, so their first cell puts the repository root on the import path. The same cell makes `import config` work:

```python
import sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
```

These files are temporary. The real `features.py` is written after the session, once D16 (zero-input behaviour) and D19 (feature column list) are decided.

## Files

| File | What it gives you | Used in |
| --- | --- | --- |
| `neuropils_provisional.csv` | Each FlyWire neuropil with `is_optic` and `region_group` | Every item that uses the populations or `region_group` |
| `eda_helpers.py` | `add_provisional_flags`, `basic_features`, `relative_weight` | All workstreams |
| `randomiser.py` | `randomise_edges` (degree-preserving, seeded), `reciprocity` | C6, D-7 |
| `tests/test_eda_shared.py` | 15 tests, including the toy graph worked by hand | Run once after copying, from the repository root: `pytest eda_shared/tests` |

## Use

```python
from eda_shared.eda_helpers import add_provisional_flags, basic_features, relative_weight
from eda_shared.randomiser import randomise_edges, reciprocity

neurons = add_provisional_flags(neurons)                      # fafb
neurons = add_provisional_flags(neurons, nerve_cord=VNC)      # malecns, see below
feats = basic_features(neurons, edges, tau=10)                # one dataset at a time
edges["frac"] = relative_weight(edges)                        # before any filtering

kept = edges[edges.syn_count >= 10]
null = [reciprocity(randomise_edges(kept, seed=s)) for s in range(20)]
```

## The neuropil table

- **Keys** are FlyWire base names (`ME`, `MB_CA` …). The 78 FlyWire neuropils were checked against fafbseg 3.2.2. Side markers (`_L`, `(R)`) are stripped automatically.
- **`is_optic`** is True for exactly five neuropils: LA, ME, AME, LO, LOP. A neuron is optic if its `region` is one of them.
- **`region_group`** is drawn from anatomy, not from any result. Rows marked `clear` follow textbook roles; rows marked `debatable` are first guesses. Project 1 reviews the debatable rows and sends changes to the group before the EDA starts; A2 then checks the grouping against the data.
- **`unknown`** and FlyWire's `UNASGD` give `region_group = unknown` and `is_optic = False`.
- **malecns names.** Hemibrain-style names (`CA`, `PED`, `aL`, `a'L`, `bL`, `b'L`, `gL`, `ROB`, `RUB`, `AB`) are mapped in `ALIASES`. Print the malecns `region` values once (Data Conventions, "Source column mapping"). Any name the code does not know raises an error naming it; report it, do not add it privately. Pass the nerve-cord neuropils explicitly as `nerve_cord`; their neurons get `in_brain = False`, and every comparison with `fafb` keeps `in_brain == True` only.

## The four features

`basic_features` returns `in_degree` and `out_degree` at the threshold you give, and `total_in_syn` and `total_out_syn` over all cached edges (τ = 1). The totals are the fixed denominators from the EDA Plan's Definitions, so the function refuses a table not cached at 1 synapse. Neurons without edges get 0; D16 is still open, so this choice holds for the EDA only.

## The randomiser

- Keeps every neuron's in-degree and out-degree exactly; no self-connections, no duplicate edges; the same seed gives the same graph.
- Double-edge swaps (Maslov & Sneppen 2002), proposed in batches: about 10 accepted swaps per edge. It is a standard practical null model, not an exactly uniform sampler.
- Threshold first, then randomise.
- Weights stay with the presynaptic neuron: out-strength is preserved, in-strength is not. C6 and D-7 use the binary graph, so this does not affect them.
- Speed on a synthetic graph shaped like a connectome: 1.5M edges in about 9 s, 3M edges in about 20 s per seed.
- Very dense subgraphs mix less, and the function warns if it cannot reach the target. Report the warning with the result.
