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


def styled_feature_table(
    df: pd.DataFrame,
    groups: Mapping[str, list[str]],
    overrides: Mapping[str, str] | None = None,
    max_rows: int = 30,
    precision: int = 3,
) -> Styler:
    """Style a feature DataFrame by colouring each column group with a faint background."""
    tints = _resolve_tints(groups, overrides)
    head = df.head(max_rows).copy()
    styler = head.style.format(precision=precision)
    for group, cols in groups.items():
        color = tints.get(group)
        present = [c for c in cols if c in head.columns]
        if color and present:
            styler = styler.set_properties(subset=present, **{"background-color": color})
    return styler


def styled_pca_preview(transformed_cols: list[str], df: pd.DataFrame) -> Styler:
    """Lightly format a PCA-projection preview table."""
    return df.head(10).style.format(precision=3).set_properties(
        subset=transformed_cols, **{"background-color": "#F5F5F5"}
    )
