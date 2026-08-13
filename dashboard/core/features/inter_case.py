"""Inter-case feature extraction: one row per calendar window, 7 system-level features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from core.windows import floor_to_window, window_index

_CELL_NAMES = {
    "new_arrivals": "Burst opening",
    "completions": "Burst closing",
    "stalled_cases": "Stalling",
    "active_cases": "High load",
    "events_per_case": "Busy cases",
    "mean_delta_t": "Slow cases",
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


def _epoch_ns(values) -> np.ndarray:
    """Timestamps as int64 nanoseconds, whatever resolution the log carries."""
    return pd.DatetimeIndex(values).as_unit("ns").view("int64")


def _stalled_count(
    df: pd.DataFrame, win_starts: pd.DatetimeIndex, win_minutes: int, stall_minutes: int
) -> np.ndarray:
    """For each window end, count the still-running cases idle for longer than τ.

    A case waits between two of its consecutive events, and its last event
    completes it — so only the gaps between consecutive events can stall, and a
    completed case is never counted (same reading of "completion" as the
    completions feature). Of those gaps only the ones longer than τ can stall;
    such a gap contributes to every window end that falls more than τ after its
    earlier event and before its later one. Counting both bounds with
    searchsorted over the sorted endpoints does all windows at once.
    """
    threshold = pd.Timedelta(minutes=stall_minutes)
    ordered = df.sort_values(["case:concept:name", "time:timestamp"])
    current = ordered["time:timestamp"]
    following = ordered.groupby("case:concept:name")["time:timestamp"].shift(-1)
    stalls = (following - current) > threshold  # a case's last event has no gap: NaT → False
    onset = np.sort(_epoch_ns(current[stalls] + threshold))
    resumes = np.sort(_epoch_ns(following[stalls]))
    win_end = _epoch_ns(win_starts + pd.Timedelta(minutes=win_minutes))
    return (
        np.searchsorted(onset, win_end, side="left")
        - np.searchsorted(resumes, win_end, side="right")
    ).astype(int)


def _describe_cell(features: np.ndarray, columns: list[str]) -> str:
    """Map a centroid feature vector to a short descriptive name."""
    if not columns:
        return "—"
    top_col = columns[int(np.argmax(features))]
    return _CELL_NAMES.get(top_col, "Steady flow")


def _delta_stats(df: pd.DataFrame, win_idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the gaps in minutes to the previous event *of the same case*, per window.

    Each gap is credited to the window of its later event, so a window measures
    how fast the cases that were running in it moved — not how densely the
    system's events happened to interleave.
    """
    ordered = df.sort_values(["case:concept:name", "time:timestamp"])
    gap_s = ordered.groupby("case:concept:name")["time:timestamp"].diff().dt.total_seconds()
    mask = gap_s.notna()
    minutes = (gap_s[mask] / 60.0).clip(lower=0)
    grouped = minutes.groupby(ordered.loc[mask, "__win__"])
    agg = grouped.agg(["mean", "std"]).reindex(win_idx).fillna(0.0)
    return agg["mean"].to_numpy(dtype=float), agg["std"].to_numpy(dtype=float)


def _values(df: pd.DataFrame, col: str) -> list[str]:
    """The distinct values of a categorical attribute, in a stable order."""
    return sorted(df[col].dropna().astype(str).unique().tolist())


def attribute_features(
    log: pd.DataFrame, numeric_attrs: Iterable[str], categorical_attrs: Iterable[str]
) -> dict[str, tuple[str, list[str]]]:
    """Selectable case-attribute features: key -> (label, matrix columns).

    A numeric attribute offers its mean and its standard deviation separately.
    A categorical one offers a single entry covering the share columns of all
    its values, since the shares are one distribution and only mean something
    together.
    """
    features: dict[str, tuple[str, list[str]]] = {}
    for col in numeric_attrs:
        features[f"attr_mean:{col}"] = (f"Mean {col}", [f"attr_mean:{col}"])
        features[f"attr_std:{col}"] = (f"Std {col}", [f"attr_std:{col}"])
    for col in categorical_attrs:
        columns = [f"attr_share:{col}={value}" for value in _values(log, col)]
        features[f"attr_share:{col}"] = (f"{col} value shares", columns)
    return features


def _attribute_columns(
    df: pd.DataFrame,
    win_idx: pd.DatetimeIndex,
    numeric_attrs: Iterable[str],
    categorical_attrs: Iterable[str],
) -> pd.DataFrame:
    """Per-window mean/std of numeric attributes and per-value shares of categorical ones."""
    out = pd.DataFrame(index=win_idx)
    for col in numeric_attrs:
        agg = (
            pd.to_numeric(df[col], errors="coerce")
            .groupby(df["__win__"])
            .agg(["mean", "std"])
            .reindex(win_idx)
            .fillna(0.0)
        )
        out[f"attr_mean:{col}"] = agg["mean"].to_numpy(dtype=float)
        out[f"attr_std:{col}"] = agg["std"].to_numpy(dtype=float)
    for col in categorical_attrs:
        values = _values(df, col)
        counts = (
            pd.crosstab(df["__win__"], df[col].astype(str))
            .reindex(index=win_idx, fill_value=0)
            .reindex(columns=values, fill_value=0)
        )
        shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        for value in values:
            out[f"attr_share:{col}={value}"] = shares[value].to_numpy(dtype=float)
    return out


def _window_counts(
    df: pd.DataFrame, win_idx: pd.DatetimeIndex, origin: pd.Timestamp, win_minutes: int
) -> pd.DataFrame:
    """Compute active cases, arrivals, completions and events per active case per window."""
    first = floor_to_window(df.groupby("case:concept:name")["time:timestamp"].min(), origin, win_minutes)
    last = floor_to_window(df.groupby("case:concept:name")["time:timestamp"].max(), origin, win_minutes)
    active = df.groupby("__win__")["case:concept:name"].nunique().reindex(win_idx, fill_value=0)
    events = df.groupby("__win__").size().reindex(win_idx, fill_value=0)
    return pd.DataFrame(
        {
            "active_cases": active,
            "new_arrivals": first.value_counts().reindex(win_idx, fill_value=0),
            "completions": last.value_counts().reindex(win_idx, fill_value=0),
            "events_per_case": events.div(active.replace(0, np.nan)).fillna(0.0),
        },
        index=win_idx,
    ).astype(float)


@st.cache_data(show_spinner=False)
def build_features(
    log: pd.DataFrame,
    window_minutes: int = 60,
    stall_minutes: int = 60,
    numeric_attrs: tuple[str, ...] = (),
    categorical_attrs: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, InterSpec]:
    """Compute the 7 per-window inter-case features plus the case-attribute ones."""
    df = log.sort_values("time:timestamp").copy()
    origin = df["time:timestamp"].min()
    df["__win__"] = floor_to_window(df["time:timestamp"], origin, window_minutes)
    win_idx = window_index(origin, df["time:timestamp"].max(), window_minutes)

    counts = _window_counts(df, win_idx, origin, window_minutes)
    deltas_mean, deltas_std = _delta_stats(df, win_idx)
    stalled = _stalled_count(df, win_idx, window_minutes, stall_minutes)

    out = counts.assign(mean_delta_t=deltas_mean, std_delta_t=deltas_std, stalled_cases=stalled.astype(float))
    attrs = _attribute_columns(df, win_idx, numeric_attrs, categorical_attrs)
    out = out.join(attrs)
    out.index.name = "window_start"
    groups = {
        "counts": ["active_cases", "events_per_case"],
        "rates": ["new_arrivals", "completions"],
        "gaps": ["mean_delta_t", "std_delta_t"],
        "stall": ["stalled_cases"],
        "attributes": list(attrs.columns),
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
