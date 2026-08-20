"""State-based drift signals: how far each window sits from a reference window.

Stays within the state-based method — no raw activity / resource / attribute
distributions are read here. The default reference is the window before, so a
shift shows up where it happens instead of being spread over every window that
differs from the average; the other references trade that locality for a
steadier comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# What a window's state distribution is compared against.
REFERENCE_LABELS = {
    "previous": "Previous window",
    "recent": "Average of the last l windows",
    "baseline": "Full-log baseline",
}

# Ways to compare two state *distributions* (intra-case windows).
DIVERGENCE_LABELS = {
    "kl": "KL divergence",
    "js": "Jensen–Shannon",
    "tv": "Total variation",
    "hellinger": "Hellinger",
}

_EPS = 1e-9


def _normalised(vector: np.ndarray) -> np.ndarray:
    """A smoothed probability vector, safe to divide by and to take logs of."""
    smoothed = np.asarray(vector, dtype=float) + _EPS
    return smoothed / smoothed.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q); both vectors normalised internally."""
    p, q = _normalised(p), _normalised(q)
    return float(np.sum(p * np.log(p / q)))


def _divergence(p: np.ndarray, q: np.ndarray, kind: str) -> float:
    """Compare two state distributions under the chosen divergence."""
    if kind == "kl":
        return _kl(p, q)
    if kind == "js":
        mean = (_normalised(p) + _normalised(q)) / 2.0
        return 0.5 * _kl(p, mean) + 0.5 * _kl(q, mean)
    if kind == "tv":
        return float(0.5 * np.abs(_normalised(p) - _normalised(q)).sum())
    if kind == "hellinger":
        root_diff = np.sqrt(_normalised(p)) - np.sqrt(_normalised(q))
        return float(np.sqrt(np.sum(root_diff**2) / 2.0))
    raise ValueError(f"Unknown divergence: {kind}")


def _vector_distance(later: np.ndarray, earlier: np.ndarray, metric: str) -> np.ndarray:
    """Row-wise distance between two aligned blocks of window vectors."""
    diff = later - earlier
    if metric == "euclidean":
        return np.linalg.norm(diff, axis=1)
    if metric == "manhattan":
        return np.abs(diff).sum(axis=1)
    if metric == "chebyshev":
        return np.abs(diff).max(axis=1)
    if metric == "cosine":
        norms = np.linalg.norm(later, axis=1) * np.linalg.norm(earlier, axis=1)
        return 1.0 - np.sum(later * earlier, axis=1) / np.where(norms > 0, norms, np.nan)
    raise ValueError(f"Unknown metric: {metric}")


def _reference_rows(rows: np.ndarray, reference: str, lookback: int) -> list[np.ndarray | None]:
    """The distribution each window is compared against; None means "no score yet"."""
    if reference == "baseline":
        base = rows.mean(axis=0)
        return [base] * len(rows)
    if reference == "recent":
        span = max(1, int(lookback))
        return [None] + [rows[max(0, i - span):i].mean(axis=0) for i in range(1, len(rows))]
    return [None] + [rows[i - 1] for i in range(1, len(rows))]


@st.cache_data(show_spinner=False)
def intra_state_shift(
    intra_dist: pd.DataFrame,
    divergence: str = "kl",
    reference: str = "previous",
    lookback: int = 5,
) -> pd.DataFrame:
    """Compare each window's intra-case state distribution against its reference.

    The reference is the window before it, the mean of the `lookback` windows
    before it, or the mean over every window in the log.
    """
    cols = [c for c in intra_dist.columns if c.startswith("intra_S")]
    if not cols or intra_dist.empty:
        return pd.DataFrame({"window_start": intra_dist.get("window_start", []), "score": []})
    dist = intra_dist.sort_values("window_start")
    rows = dist[cols].to_numpy(dtype=float)
    scores = [
        0.0 if against is None else _divergence(rows[i], against, divergence)
        for i, against in enumerate(_reference_rows(rows, reference, lookback))
    ]
    return pd.DataFrame({"window_start": dist["window_start"].values, "score": scores})


@st.cache_data(show_spinner=False)
def window_vector_shift(
    window_starts: pd.Series, vectors: np.ndarray, metric: str = "euclidean"
) -> pd.DataFrame:
    """Distance between each window's compressed vector and the previous one's.

    Resource and inter-case windows are represented by one PCA-compressed
    vector each, so consecutive windows are compared directly in that space.
    """
    if len(vectors) == 0:
        return pd.DataFrame({"window_start": [], "score": []})
    mat = np.asarray(vectors, dtype=float)
    scores = np.zeros(len(mat))
    if len(mat) > 1:
        scores[1:] = np.nan_to_num(_vector_distance(mat[1:], mat[:-1], metric))
    return pd.DataFrame({"window_start": pd.Series(window_starts).values, "score": scores})
