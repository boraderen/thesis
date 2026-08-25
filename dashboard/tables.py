"""Styled-table helpers — pandas Styler with per-group background tints."""
from __future__ import annotations

from typing import Mapping

import pandas as pd
from pandas.io.formats.style import Styler

# Rotating tints for feature groups; resolution falls back through this cycle.
TINT_CYCLE = ["#EEEDFE", "#E1F5EE", "#FAEEDA", "#FAECE7", "#E7F0FA", "#F4EAF6",
              "#EFF4E4", "#FBE9F0", "#E6F1F1", "#F2EFE3"]

GROUP_TINTS: dict[str, str] = {
    "activity_freq": "#EEEDFE",
    "bigram": "#E1F5EE",
    "vocab": "#FAEEDA",
    "progress": "#EEEEEE",
    "current": "#FAECE7",
    "history": "#EFF4E4",
    "events": "#FAEEDA",
    "active": "#E1F5EE",
    "duration": "#EEEDFE",
    "wait": "#FAECE7",
    "activity_events": "#E7F0FA",
    "ho": "#FFF6CC",
    "active_cases": "#E1F5EE",
    "events_per_case": "#E1F5EE",
    "new_arrivals": "#EEEDFE",
    "completions": "#EEEDFE",
    "mean_delta_t": "#FFF6CC",
    "std_delta_t": "#FFF6CC",
    "stalled_cases": "#FAEEDA",
}


def _tint_for(group: str, index: int) -> str:
    if group in GROUP_TINTS:
        return GROUP_TINTS[group]
    if group.startswith("attr_"):
        return "#E7F0FA"
    return TINT_CYCLE[index % len(TINT_CYCLE)]


def _darken(hex_color: str, factor: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02X}{:02X}{:02X}".format(int(r * factor), int(g * factor), int(b * factor))


def _binary_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    out: list[str] = []
    for c in candidates:
        if c not in df.columns:
            continue
        vals = df[c].dropna().unique()
        if len(vals) and set(vals).issubset({0, 0.0, 1, 1.0}):
            out.append(c)
    return out


def styled_feature_table(
    df: pd.DataFrame,
    groups: Mapping[str, list[str]],
    max_rows: int = 30,
    precision: int = 3,
) -> Styler:
    """Style a feature DataFrame: per-group tint + bold dark highlight on one-hot 1s."""
    head = df.head(max_rows).copy()
    styler = head.style.format(precision=precision)
    for i, (group, cols) in enumerate(groups.items()):
        color = _tint_for(group, i)
        present = [c for c in cols if c in head.columns]
        if not present:
            continue
        styler = styler.set_properties(subset=present, **{"background-color": color})
        highlight = _darken(color)
        for col in _binary_columns(head, present):
            styler = styler.map(
                lambda v, hi=highlight: (
                    f"background-color: {hi}; color: white; font-weight: 600" if v == 1 else ""
                ),
                subset=[col],
            )
    return styler
