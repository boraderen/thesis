"""Detect state transitions in a trajectory and summarise their boundary conditions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _top_changes(before: pd.Series, after: pd.Series, top_n: int) -> str:
    """Return a 'name: a→b (Δ)' string for the top_n features that moved most."""
    delta = (after - before).abs()
    top = delta.sort_values(ascending=False).head(top_n).index
    parts = []
    for col in top:
        a = float(before[col])
        b = float(after[col])
        if a == b:
            continue
        parts.append(f"{col}: {a:.2f} → {b:.2f} ({b - a:+.2f})")
    return "; ".join(parts) if parts else "—"


@st.cache_data(show_spinner=False)
def find_transitions(
    timestamps: pd.Series,
    state_ids: np.ndarray,
    cell_labels: list[str],
    features: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Return a row per state change with timestamp, labels, and dominant feature deltas."""
    if len(state_ids) < 2:
        return pd.DataFrame(columns=["timestamp", "from", "to", "top_changes"])
    changes = np.where(np.diff(state_ids) != 0)[0]
    rows = []
    feat = features.reset_index(drop=True)
    times = pd.to_datetime(timestamps).reset_index(drop=True)
    for idx in changes:
        before, after = feat.iloc[idx], feat.iloc[idx + 1]
        rows.append({
            "timestamp": times.iloc[idx + 1],
            "from_idx": int(state_ids[idx]),
            "to_idx": int(state_ids[idx + 1]),
            "from": cell_labels[int(state_ids[idx])] if int(state_ids[idx]) < len(cell_labels) else f"S{int(state_ids[idx])}",
            "to": cell_labels[int(state_ids[idx + 1])] if int(state_ids[idx + 1]) < len(cell_labels) else f"S{int(state_ids[idx + 1])}",
            "top_changes": _top_changes(before, after, top_n),
        })
    df = pd.DataFrame(rows)
    return df[["timestamp", "from", "to", "from_idx", "to_idx", "top_changes"]] if len(df) else df
