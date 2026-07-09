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
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    origin = df["time:timestamp"].min()
    df["__win__"] = floor_to_window(df["time:timestamp"], origin, window_minutes)
    counts = (
        df.groupby(["__win__", "state_id"]).size().unstack(fill_value=0)
        .reindex(columns=range(n_states), fill_value=0)
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    fractions = counts.div(totals, axis=0).fillna(0.0)
    fractions.columns = [f"intra_S{s}" for s in fractions.columns]
    fractions.index.name = "window_start"
    return fractions.reset_index()
