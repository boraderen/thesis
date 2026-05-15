"""Inter-case feature extraction: one row per calendar window, 7 system-level features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

_CELL_NAMES = {
    "new_arrivals": "Burst opening",
    "completions": "Burst closing",
    "stalled_cases": "Stalling",
    "active_cases": "High load",
    "total_events": "High throughput",
    "mean_delta_t": "Irregular timing",
    "std_delta_t": "Irregular timing",
}


@dataclass(frozen=True)
class InterSpec:
    """Column metadata for the inter-case feature matrix."""

    columns: list[str]
    groups: dict[str, list[str]]
    window_minutes: int
    stall_minutes: int
    window_starts: pd.DatetimeIndex


def _zlog_minutes(seconds: np.ndarray) -> np.ndarray:
    """Seconds → ln(minutes), zeros preserved."""
    return np.log1p(np.clip(seconds / 60.0, 0, None))


def _stalled_count(
    case_last: pd.Series, win_starts: pd.DatetimeIndex, win_minutes: int, stall_minutes: int
) -> np.ndarray:
    """For each window end, count open cases whose last event is older than `stall_minutes`."""
    threshold = pd.Timedelta(minutes=stall_minutes)
    win_end = win_starts + pd.Timedelta(minutes=win_minutes)
    out = np.zeros(len(win_starts), dtype=int)
    for i, w_end in enumerate(win_end):
        active = case_last[case_last <= w_end]
        out[i] = int(((w_end - active) > threshold).sum())
    return out


def _describe_cell(features: np.ndarray, columns: list[str]) -> str:
    """Map a centroid feature vector to a short descriptive name."""
    if not columns:
        return "—"
    top_col = columns[int(np.argmax(features))]
    return _CELL_NAMES.get(top_col, "Steady flow")


def _delta_stats(df: pd.DataFrame, win_index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of ln-minute gaps between consecutive events in each window."""
    means = np.zeros(len(win_index), dtype=float)
    stds = np.zeros(len(win_index), dtype=float)
    for i, w_start in enumerate(win_index):
        bucket = df[df["__win__"] == w_start]
        if len(bucket) < 2:
            continue
        seconds = bucket["timestamp"].sort_values().diff().dt.total_seconds().dropna().to_numpy()
        if not len(seconds):
            continue
        log_min = _zlog_minutes(seconds)
        means[i] = float(np.mean(log_min))
        stds[i] = float(np.std(log_min))
    return means, stds


def _window_counts(df: pd.DataFrame, win_index: pd.DatetimeIndex, win_minutes: int) -> pd.DataFrame:
    """Compute active/arrivals/completions/totals per window."""
    first = df.groupby("case_id")["timestamp"].min().dt.floor(f"{win_minutes}min")
    last = df.groupby("case_id")["timestamp"].max().dt.floor(f"{win_minutes}min")
    return pd.DataFrame(
        {
            "active_cases": df.groupby("__win__")["case_id"].nunique().reindex(win_index, fill_value=0),
            "new_arrivals": first.value_counts().reindex(win_index, fill_value=0),
            "completions": last.value_counts().reindex(win_index, fill_value=0),
            "total_events": df.groupby("__win__").size().reindex(win_index, fill_value=0),
        },
        index=win_index,
    ).astype(float)


@st.cache_data(show_spinner=False)
def build_features(
    log: pd.DataFrame, window_minutes: int = 60, stall_minutes: int = 60
) -> tuple[pd.DataFrame, InterSpec]:
    """Compute the 7 per-window inter-case features."""
    df = log.sort_values("timestamp").copy()
    df["__win__"] = df["timestamp"].dt.floor(f"{window_minutes}min")
    win_index = pd.date_range(df["__win__"].min(), df["__win__"].max(), freq=f"{window_minutes}min")

    counts = _window_counts(df, win_index, window_minutes)
    deltas_mean, deltas_std = _delta_stats(df, win_index)
    case_last = df.groupby("case_id")["timestamp"].max()
    stalled = _stalled_count(case_last, win_index, window_minutes, stall_minutes)

    out = counts.assign(mean_delta_t=deltas_mean, std_delta_t=deltas_std, stalled_cases=stalled.astype(float))
    out.index.name = "window_start"
    groups = {
        "counts": ["active_cases", "total_events"],
        "rates": ["new_arrivals", "completions"],
        "gaps": ["mean_delta_t", "std_delta_t"],
        "stall": ["stalled_cases"],
    }
    spec = InterSpec(
        columns=list(out.columns),
        groups=groups,
        window_minutes=window_minutes,
        stall_minutes=stall_minutes,
        window_starts=pd.DatetimeIndex(out.index),
    )
    return out.reset_index(), spec


def describe_cells(centroids: np.ndarray, columns: list[str]) -> list[str]:
    """Convert SOM cell centroids to short descriptive labels."""
    return [_describe_cell(c, columns) for c in centroids]
