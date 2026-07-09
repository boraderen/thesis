"""Per-perspective drift attribution scores against the full-log baseline.

The four scores (control-flow / resource / inter-case / data) are computed
per calendar window. The window-vs-baseline divergence is:

- Control-flow: KL(activity-dist_window || activity-dist_log)
- Resource: mean over activities of KL(resource-dist_window | a || baseline | a)
- Inter-case: absolute z-score of the per-window event count
- Data: |z(mean)| for each numeric case attribute + KL for each categorical,
  averaged
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.windows import floor_to_window, window_index


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with epsilon smoothing; both vectors normalised internally."""
    eps = 1e-9
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    return float(np.sum((p / p.sum()) * np.log((p / p.sum()) / (q / q.sum()))))


def _dist(values: pd.Series, universe: list[str]) -> np.ndarray:
    """Counts of `values` over `universe`, unnormalised (smoothed downstream)."""
    counts = values.value_counts()
    return np.array([counts.get(v, 0) for v in universe], dtype=float)


def _cf_score(bucket: pd.DataFrame, activities: list[str], baseline: np.ndarray) -> float:
    """Window activity-distribution KL divergence vs. baseline."""
    return _kl(_dist(bucket["concept:name"], activities), baseline)


def _resource_score(
    bucket: pd.DataFrame, activities: list[str], resources: list[str],
    baseline_by_activity: dict[str, np.ndarray],
) -> float:
    """Mean over activities of KL(resource-dist in window for a || baseline for a)."""
    scores: list[float] = []
    for a in activities:
        sub = bucket[bucket["concept:name"] == a]
        if len(sub) == 0:
            continue
        scores.append(_kl(_dist(sub["org:resource"].astype(str), resources), baseline_by_activity[a]))
    return float(np.mean(scores)) if scores else 0.0


def _data_score(
    bucket: pd.DataFrame,
    numeric_attrs: tuple[str, ...], num_baseline: dict[str, tuple[float, float]],
    categorical_attrs: tuple[str, ...], cat_baseline: dict[str, tuple[list[str], np.ndarray]],
) -> float:
    """Average of |z(mean)| over numeric attrs + KL over categorical attrs."""
    parts: list[float] = []
    for col in numeric_attrs:
        if col not in bucket.columns or bucket[col].empty:
            continue
        mean, std = num_baseline[col]
        parts.append(abs(float(bucket[col].mean()) - mean) / max(std, 1e-9))
    for col in categorical_attrs:
        if col not in bucket.columns or bucket[col].empty:
            continue
        universe, base = cat_baseline[col]
        parts.append(_kl(_dist(bucket[col].astype(str), universe), base))
    return float(np.mean(parts)) if parts else 0.0


def _build_baseline(
    df: pd.DataFrame,
    numeric_attrs: tuple[str, ...],
    categorical_attrs: tuple[str, ...],
) -> dict:
    """Pre-compute distributions and moments from the full log."""
    activities = sorted(df["concept:name"].unique().tolist())
    has_resource = "org:resource" in df.columns
    resources = sorted(df["org:resource"].astype(str).unique().tolist()) if has_resource else []
    return {
        "activities": activities,
        "cf": _dist(df["concept:name"], activities),
        "resources": resources,
        "resource_by_activity": {
            a: _dist(df.loc[df["concept:name"] == a, "org:resource"].astype(str), resources)
            for a in activities
        } if has_resource else {},
        "num": {c: (float(df[c].mean()), float(df[c].std())) for c in numeric_attrs},
        "cat": {
            c: (sorted(df[c].astype(str).unique().tolist()),
                _dist(df[c].astype(str), sorted(df[c].astype(str).unique().tolist())))
            for c in categorical_attrs
        },
    }


@st.cache_data(show_spinner=False)
def per_window_scores(
    log: pd.DataFrame,
    window_minutes: int,
    numeric_attrs: tuple[str, ...] = (),
    categorical_attrs: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return one row per window with cf/resource/inter/data scores."""
    df = log.copy()
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    origin = df["time:timestamp"].min()
    df["__win__"] = floor_to_window(df["time:timestamp"], origin, window_minutes)
    win_idx = window_index(origin, df["time:timestamp"].max(), window_minutes)
    base = _build_baseline(df, numeric_attrs, categorical_attrs)

    counts = df.groupby("__win__").size().reindex(win_idx, fill_value=0)
    ic_mean, ic_std = float(counts.mean()), max(float(counts.std()), 1e-9)
    has_resource = "org:resource" in df.columns

    rows: list[dict] = []
    for win in win_idx:
        bucket = df[df["__win__"] == win]
        rows.append({
            "window_start": win,
            "cf_score": _cf_score(bucket, base["activities"], base["cf"]) if len(bucket) else 0.0,
            "resource_score": _resource_score(bucket, base["activities"], base["resources"], base["resource_by_activity"])
                if has_resource and len(bucket) else 0.0,
            "inter_score": abs(len(bucket) - ic_mean) / ic_std,
            "data_score": _data_score(bucket, numeric_attrs, base["num"], categorical_attrs, base["cat"])
                if len(bucket) else 0.0,
        })
    return pd.DataFrame(rows)
