"""Drift signals from state behaviour over time.

Stays within the state-based method — no raw activity / resource / attribute
distributions are read here. The default reference is the window before, so a
shift shows up where it happens instead of being spread over every window that
differs from the average; the other references trade that locality for a
steadier comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# What a window's state distribution is compared against.
REFERENCES = {
    "previous": "Previous window",
    "recent": "Average of the last l windows",
    "baseline": "Full-log baseline",
}

# Ways to compare two state distributions.
DIVERGENCES = {
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
    p, q = _normalised(p), _normalised(q)
    return float(np.sum(p * np.log(p / q)))


def divergence(p: np.ndarray, q: np.ndarray, kind: str = "js") -> float:
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


def drift_signal(
    distribution: pd.DataFrame,
    divergence_kind: str = "js",
    reference: str = "previous",
    lookback: int = 5,
) -> pd.DataFrame:
    """Compare each window's state distribution against its reference.

    `distribution` is the output of ``state_distribution`` (columns ``S*`` plus
    ``window_start``). The reference is the window before, the mean of the
    `lookback` windows before, or the mean over every window in the log.
    """
    cols = [c for c in distribution.columns if c.startswith("S")]
    if not cols or distribution.empty:
        return pd.DataFrame({"window_start": distribution.get("window_start", []), "score": []})
    dist = distribution.sort_values("window_start")
    rows = dist[cols].to_numpy(dtype=float)
    scores = [
        0.0 if against is None else divergence(rows[i], against, divergence_kind)
        for i, against in enumerate(_reference_rows(rows, reference, lookback))
    ]
    return pd.DataFrame({"window_start": dist["window_start"].values, "score": scores})


def window_vector_shift(
    window_starts: pd.Series,
    vectors: np.ndarray,
    metric: str = "euclidean",
    reference: str = "previous",
    lookback: int = 5,
) -> pd.DataFrame:
    """Distance between each window's compressed vector and its reference vector.

    Resource and inter-case windows are represented by one compressed vector
    each, so windows are compared directly in that space. The reference is the
    window before, the mean of the `lookback` windows before, or the mean over
    every window in the log — the same three choices the distribution-based
    ``drift_signal`` offers, and with the same trade-off: `previous` keeps a
    shift local to where it happens, `baseline` spreads it over every window
    that differs from the average.
    """
    if len(vectors) == 0:
        return pd.DataFrame({"window_start": [], "score": []})
    mat = np.asarray(vectors, dtype=float)
    scores = np.zeros(len(mat))
    against = _reference_rows(mat, reference, lookback)
    scored = [i for i, row in enumerate(against) if row is not None]
    if scored:
        earlier = np.vstack([against[i] for i in scored])
        scores[scored] = np.nan_to_num(_vector_distance(mat[scored], earlier, metric))
    return pd.DataFrame({"window_start": pd.Series(window_starts).values, "score": scores})


def top_drift_windows(signal: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """The k windows with the highest drift score, strongest first."""
    if signal.empty:
        return signal
    return signal.sort_values("score", ascending=False).head(k).reset_index(drop=True)
