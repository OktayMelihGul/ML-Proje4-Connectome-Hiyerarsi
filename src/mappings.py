"""Source value -> vocabulary mappings (Data Conventions, "Vocabularies").

Built from the values the pinned sources actually print (README, "Source columns").
recode() raises on any unmapped value; a missing source value becomes "unknown".
"""
from __future__ import annotations

import pandas as pd


def recode(series: pd.Series, mapping: dict, column: str) -> pd.Series:
    """The Conventions' recode(): unmapped non-null values raise, missing -> unknown."""
    out = series.map(mapping)
    unmapped = series[out.isna() & series.notna()].unique()
    if len(unmapped):
        raise ValueError(f"{column}: unmapped values {unmapped[:10]}")
    return out.fillna("unknown")


FAFB_SIDE = {"left": "left", "right": "right", "center": "center", "na": "unknown"}
MALECNS_SIDE = {"L": "left", "R": "right", "M": "center", "unknown": "unknown"}

_NT = ["acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine"]
FAFB_NT = {v: v for v in _NT}                                    # FAFB has no histamine class
MALECNS_NT = {v: v for v in _NT} | {"histamine": "histamine", "unclear": "unknown"}

# FAFB super_class. sensory_ascending (612 neurons, flow = afferent) has no level in
# config.SUPER_CLASS_VALUES; default (logged, open for the group): -> sensory, as in week 2,
# so these afferents stay in the sensory seed set.
FAFB_SUPER = {v: v for v in ["sensory", "ascending", "descending", "motor", "endocrine",
                             "visual_projection", "visual_centrifugal", "optic", "central"]}
FAFB_SUPER["sensory_ascending"] = "sensory"

# MaleCNS superclass mapped onto FlyWire's levels. Levels with no FlyWire counterpart
# (vnc_intrinsic, *_efferent, ENS, vnc_tbc) become unknown until the group adds
# MaleCNS-only levels to config.SUPER_CLASS_VALUES ("after EDA"); they are almost all
# nerve-cord neurons and leave the brain set anyway.
MALECNS_SUPER = {
    "ol_intrinsic": "optic", "cb_intrinsic": "central",
    "visual_projection": "visual_projection", "visual_projection_tbc": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "cb_sensory": "sensory", "ol_sensory": "sensory", "vnc_sensory": "sensory",
    "cb_sensory_tbc": "sensory", "vnc_sensory_tbc": "sensory", "sensory_descending": "sensory",
    "sensory_ascending": "sensory", "sensory_ascending_tbc": "sensory",
    "ascending_neuron": "ascending", "descending_neuron": "descending",
    "cb_motor": "motor", "vnc_motor": "motor",
    "cb_endocrine": "endocrine", "vnc_endocrine": "endocrine",
    "vnc_intrinsic": "unknown", "cb_efferent": "unknown", "vnc_efferent": "unknown",
    "efferent_ascending": "unknown", "efferent_descending": "unknown",
    "ENS": "unknown", "vnc_tbc": "unknown",
}
