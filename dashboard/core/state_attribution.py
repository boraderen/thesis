"""State-based drift signals: how far each window sits from the window before it.

Stays within the state-based method — no raw activity / resource / attribute
distributions are read here. Every score is window-to-window, never against a
full-log baseline, so a shift shows up where it happens instead of being spread
over every window that differs from the average.
"""
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
    """KL of each window's intra-case state distribution against the previous window's."""
    cols = [c for c in intra_dist.columns if c.startswith("intra_S")]
    if not cols or intra_dist.empty:
        return pd.DataFrame({"window_start": intra_dist.get("window_start", []), "score": []})
    dist = intra_dist.sort_values("window_start")
    rows = dist[cols].to_numpy(dtype=float)
    scores = [0.0] + [_kl(rows[i], rows[i - 1]) for i in range(1, len(rows))]
    return pd.DataFrame({"window_start": dist["window_start"].values, "score": scores})


@st.cache_data(show_spinner=False)
def window_vector_shift(window_starts: pd.Series, vectors: np.ndarray) -> pd.DataFrame:
    """Euclidean distance between each window's compressed vector and the previous one's.

    Resource and inter-case windows are represented by one PCA-compressed
    vector each, so consecutive windows are compared directly in that space.
    """
    if len(vectors) == 0:
        return pd.DataFrame({"window_start": [], "score": []})
    mat = np.asarray(vectors, dtype=float)
    scores = np.zeros(len(mat))
    if len(mat) > 1:
        scores[1:] = np.linalg.norm(np.diff(mat, axis=0), axis=1)
    return pd.DataFrame({"window_start": pd.Series(window_starts).values, "score": scores})
