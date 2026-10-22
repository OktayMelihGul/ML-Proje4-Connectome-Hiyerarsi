# Step 0 count check

Built/read in 126.8 s. All 8 validators passed on neurons, edges and edgeroi of both datasets; assert_interchangeable passed for all three table pairs.

| Table | Expected | Ours |
|---|---|---|
| fafb neurons | 139248 | 139,255 |
| fafb edges with syn_count >= 5 | 2700513 | 2,700,513 |
| malecns neurons | 165122 | 165,122 |
| malecns edges with syn_count >= 5 | one agreed number | 6,235,649 |
| self-connections dropped (fafb / malecns) | 0 / 101 | 0 / 101 |

| Other count | Value |
|---|---|
| fafb edges (syn_count >= 1) | 15,091,983 |
| fafb edgeroi rows | 16,847,997 |
| malecns edges (syn_count >= 1) | 25,563,096 |
| malecns edgeroi rows | 29,436,294 |
| malecns brain set (in_brain) | 142,757 |
| malecns region = CentralBrain-unspecified | 365 |
| malecns region = Optic-unspecified(L) | 359 |
| malecns region = Optic-unspecified(R) | 37 |
| malecns region = VNC-unspecified (nerve cord) | 230 |
| malecns region = unknown (no primary-ROI synapse) | 338 |
| fafb region = UNASGD | 9 |
| fafb region = unknown (no synapse) | 616 |
| fafb region ties (alphabetical) | 319 |
| malecns region ties (alphabetical) | 81 |

## Checks run

- fafb neurons: columns, dtypes, nulls, keys, vocabulary
- fafb edges: columns, dtypes, nulls, keys, references, vocabulary, threshold, plausibility (skipped: n_post missing)
- fafb edgeroi: columns, dtypes, nulls, keys, references, vocabulary, threshold
- malecns neurons: columns, dtypes, nulls, keys, vocabulary
- malecns edges: columns, dtypes, nulls, keys, references, vocabulary, threshold, plausibility (164,463 neurons checked)
- malecns edgeroi: columns, dtypes, nulls, keys, references, vocabulary, threshold

## Build log

```json
{
  "fafb": {
    "self_connections_dropped_edgeroi": 0,
    "proofread_ids_783": 139255,
    "annotation_rows_v3.2.0": 139248,
    "proofread_ids_without_annotation": [
      720575940606792881,
      720575940607862706,
      720575940613336473,
      720575940613437206,
      720575940619169349,
      720575940623450077,
      720575940623913933,
      720575940625523749,
      720575940626741265,
      720575940628159081,
      720575940633242449,
      720575940636488667,
      720575940639387763,
      720575940642560987
    ],
    "proofread_ids_without_annotation_in_edges": 14,
    "annotation_ids_not_in_783": [
      720575940616847814,
      720575940618455453,
      720575940622062913,
      720575940624154153,
      720575940626850833,
      720575940628215433,
      720575940637590670
    ],
    "annotation_ids_not_in_783_in_edges": 0,
    "region_ties_alphabetical": 319,
    "self_connections_dropped_edges": 0,
    "self_connection_synapses_dropped": 0,
    "self_connections_dropped_syn_ge5": 0
  },
  "malecns": {
    "region_ties_alphabetical": 81,
    "neurons_without_primary_roi_synapses": 338,
    "nt_confidence_blanked_where_nt_unknown": 13508,
    "partner_rows": 124025046,
    "partner_min_confidence": 0.5,
    "partner_rows_with_non_traced_body": 0,
    "partner_rows_outside_primary_rois": 251,
    "self_connections_dropped_edgeroi": 131,
    "weights_file_rows": 25563197,
    "self_connections_dropped_edges": 101,
    "self_connection_synapses_dropped": 473,
    "self_connections_dropped_syn_ge5": 33,
    "weights_file_vs_edgeroi_sum": {
      "only_in_file": 0,
      "only_in_edgeroi": 0,
      "weight_differs": 0
    }
  }
}
```
