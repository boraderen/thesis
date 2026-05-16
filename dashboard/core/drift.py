"""Per-window state frequency distributions from the intra-case SOM."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.windows import floor_to_window


@st.cache_data(show_spinner=False)
def intra_state_distribution(
    feat: pd.DataFrame, n_states: int, window_minutes: int
) -> pd.DataFrame:
    """Aggregate per-event SOM states into per-window frequency vectors."""
    df = feat.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    origin = df["timestamp"].min()
    df["__win__"] = floor_to_window(df["timestamp"], origin, window_minutes)
    counts = (
        df.groupby(["__win__", "state_id"]).size().unstack(fill_value=0)
        .reindex(columns=range(n_states), fill_value=0)
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    fractions = counts.div(totals, axis=0).fillna(0.0)
    fractions.columns = [f"intra_S{s}" for s in fractions.columns]
    fractions.index.name = "window_start"
    return fractions.reset_index()


@st.cache_data(show_spinner=False)
def joint_signal(
    intra_dist: pd.DataFrame,
    resource_states: pd.DataFrame | None,
    inter_states: pd.DataFrame | None,
) -> pd.DataFrame:
    """Outer-join intra distribution with resource/inter state labels on window_start."""
    out = intra_dist.copy()
    if resource_states is not None and not resource_states.empty:
        out = out.merge(resource_states, on="window_start", how="outer")
    if inter_states is not None and not inter_states.empty:
        out = out.merge(inter_states, on="window_start", how="outer")
    return out.sort_values("window_start").reset_index(drop=True)
