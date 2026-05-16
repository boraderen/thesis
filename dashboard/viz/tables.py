"""Reusable styled-table helpers — pandas Styler with per-group background tints."""
from __future__ import annotations

from typing import Mapping

import pandas as pd
from pandas.io.formats.style import Styler

from viz.palette import GROUP_TINTS


def _resolve_tints(groups: Mapping[str, list[str]], overrides: Mapping[str, str] | None) -> dict[str, str]:
    """Combine the default GROUP_TINTS with per-call overrides."""
    tints = dict(GROUP_TINTS)
    if overrides:
        tints.update(overrides)
    return tints


def _darken(hex_color: str, factor: float = 0.55) -> str:
    """Return a darker hex color for highlighting active one-hot cells."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02X}{:02X}{:02X}".format(int(r * factor), int(g * factor), int(b * factor))


def _binary_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return columns whose values are all in {0, 1}."""
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
    overrides: Mapping[str, str] | None = None,
    max_rows: int = 30,
    precision: int = 3,
) -> Styler:
    """Style a feature DataFrame: per-group tint + bold dark highlight on one-hot 1s."""
    tints = _resolve_tints(groups, overrides)
    head = df.head(max_rows).copy()
    styler = head.style.format(precision=precision)
    for group, cols in groups.items():
        color = tints.get(group)
        present = [c for c in cols if c in head.columns]
        if not (color and present):
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


def styled_pca_preview(transformed_cols: list[str], df: pd.DataFrame) -> Styler:
    """Lightly format a PCA-projection preview table."""
    return df.head(10).style.format(precision=3).set_properties(
        subset=transformed_cols, **{"background-color": "#F5F5F5"}
    )
