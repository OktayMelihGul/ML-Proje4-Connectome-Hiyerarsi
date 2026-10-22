# EDA Workstream D — Degrees, paths and reachability (Project 4): one-page summary

fafb = FlyWire 783, annotations v3.2.0, Buhmann synapses (139,255 neurons); malecns = male-cns v1.0, Traced, conf ≥ 0.5,
brain only (142,757 neurons). Central = is_optic False. Default τ = 10. Notebook `notebooks/eda_D.ipynb`, figures
`figures/eda/D-*`, tables `reports/eda_D_tables/`. Full detail, Step 0 counts and group decisions: [eda_D_appendix.md](eda_D_appendix.md).

- **D-1 Degrees (τ = 10):** median in/out central 4/5 (fafb) and 11/14 (malecns) vs optic 3/3 and 5/7; central neurons are 4.4–5.4×
  as likely as optic ones to have ≥ 100 input partners. → The population choice changes every degree statistic; malecns degrees are ~2× fafb's.
- **D-2 Hubs (top 1% of central by in + out):** fafb 445 (degree ≥ 208), malecns 440 (≥ 335); 90% super_class central, mostly input hubs
  (in > out for 58% / 66%), top regions AVLP and AL. → Hubs are central integrators, not giant optic cells; they shorten paths into AVLP/AL.
- **D-3 No inputs / no outputs (central):** at τ = 10 no input 20.7% (fafb) / 7.2% (malecns), no output 15.8% / 3.1%; no input
  62.8% / 35.7% at τ = 31, ≤ 1.8% at τ = 1. → The zero-vs-null choice (D16) touches one central neuron in five in fafb, four times fewer in malecns.
- **D-4 Seed sets:** sensory 17,516 (fafb) vs 9,142 (malecns brain), visual 11,391 vs 4,107; motor 110 vs 107, almost all in GNG.
  → Depths are counted from different starting sets, mostly because of the photoreceptors.
- **D-5 Reachability (central non-sensory, BFS from all sensory):** fafb 99.9 / 96.5 / 87.7 / 40.6% at τ = 1 / 5 / 10 / 31, malecns
  99.9 / 99.7 / 98.2 / 70.3%; median depth 3 (fafb) / 2 (malecns) hops at τ = 10. → Reachability hinges on weak edges: τ 5 → 10 makes 3,310 more fafb neurons unreachable.
- **D-6 Who is unreachable (τ = 10):** fafb 4,615 neurons, 24.1% ascending (vs 1.9% of reachable), median degree 2 vs 14; malecns 713,
  20.9% ascending (vs 2.4%), median degree 4 vs 30. → Not a random subset, so dropping them biases depth: Project 4's claim, confirmed descriptively.
- **D-7 Reciprocity vs degree-preserving null (τ = 10, 20 seeds):** whole graph 0.115 vs 0.0026 (fafb, 44×), 0.127 vs 0.0021 (malecns, 60×);
  all 20 top regions above the 97.5% bound in both. → Far above what degrees explain, highest in EB, MB_ML_L and AL (recurrent circuits).
  - **D-7b vs region-preserving null (added after the session):** 0.115 vs 0.031 (3.7×), 0.127 vs 0.029 (4.4×); above the bound in 18 of 19
    informative fafb regions and 20/20 malecns. → Region structure gives about a quarter of the reciprocity; the excess is ~4×, not 44–60×, yet above chance almost everywhere.

**Surprise:** in brain-only malecns 14.2% of ascending neurons are unreachable at τ = 10, against 1.3% on the full CNS (fafb: 63.5%).
Cutting the nerve cord removes their inputs, so unreachability is partly an artefact of the imaged volume, not only of the threshold.
