"""State-based perspective attribution: how much each SOM's state distribution
shifts away from its full-log baseline. Stays within the state-based method —
no raw activity / resource / attribute distributions are read here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with epsilon smoothing; both vectors normalised internally."""
    eps = 1e-9
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    return float(np.sum((p / p.sum()) * np.log((p / p.sum()) / (q / q.sum()))))


@st.cache_data(show_spinner=False)
def intra_state_shift(intra_dist: pd.DataFrame) -> pd.DataFrame:
    """Per-window KL of intra-case state fractions against the full-log mean."""
    cols = [c for c in intra_dist.columns if c.startswith("intra_S")]
    if not cols or intra_dist.empty:
        return pd.DataFrame({"window_start": intra_dist.get("window_start", []), "score": []})
    baseline = intra_dist[cols].mean(axis=0).to_numpy()
    scores = [_kl(row[cols].to_numpy(), baseline) for _, row in intra_dist.iterrows()]
    return pd.DataFrame({"window_start": intra_dist["window_start"].values, "score": scores})


@st.cache_data(show_spinner=False)
def state_id_shift(
    states_df: pd.DataFrame, n_states: int, smoothing_window: int = 5
) -> pd.DataFrame:
    """Rolling-window KL of one-hot state IDs against the full-log mean fraction.

    Resource and inter-case SOMs assign one state per calendar window, so a
    single-window distribution is degenerate. A short rolling window smooths
    the indicator: a sustained shift in state mix produces a high KL.
    """
    if states_df.empty:
        return pd.DataFrame({"window_start": [], "score": []})
    df = states_df.sort_values("window_start").reset_index(drop=True)
    ids = df["state_id"].astype(int).to_numpy()
    one_hot = np.zeros((len(df), n_states), dtype=float)
    one_hot[np.arange(len(df)), np.clip(ids, 0, n_states - 1)] = 1.0
    smoothed = pd.DataFrame(one_hot).rolling(smoothing_window, min_periods=1).mean().to_numpy()
    baseline = one_hot.mean(axis=0)
    scores = [_kl(smoothed[i], baseline) for i in range(len(smoothed))]
    return pd.DataFrame({"window_start": df["window_start"].values, "score": scores})
