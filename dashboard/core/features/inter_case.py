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
    """For each window end, count cases whose last event is older than `stall_minutes`.

    Vectorised via searchsorted on case_last sorted in ns. We count cases with
    `case_last < (win_end - threshold)` — under threshold > 0 this already implies
    the case has started by win_end, so a separate filter is unnecessary.
    """
    threshold_ns = pd.Timedelta(minutes=stall_minutes).value
    win_end = win_starts + pd.Timedelta(minutes=win_minutes)
    cutoff_ns = win_end.view("int64") - threshold_ns
    last_ns = np.sort(pd.DatetimeIndex(case_last).view("int64"))
    return np.searchsorted(last_ns, cutoff_ns, side="left").astype(int)


def _describe_cell(features: np.ndarray, columns: list[str]) -> str:
    """Map a centroid feature vector to a short descriptive name."""
    if not columns:
        return "—"
    top_col = columns[int(np.argmax(features))]
    return _CELL_NAMES.get(top_col, "Steady flow")


def _delta_stats(df: pd.DataFrame, win_idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of ln-minute gaps between consecutive in-window events, per window."""
    ordered = df.sort_values(["__win__", "time:timestamp"])
    gap_s = ordered.groupby("__win__")["time:timestamp"].diff().dt.total_seconds()
    mask = gap_s.notna()
    log_min = pd.Series(_zlog_minutes(gap_s[mask].to_numpy()), index=gap_s.index[mask])
    grouped = log_min.groupby(ordered.loc[mask, "__win__"])
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
    """Compute active/arrivals/completions/totals per window."""
    first = floor_to_window(df.groupby("case:concept:name")["time:timestamp"].min(), origin, win_minutes)
    last = floor_to_window(df.groupby("case:concept:name")["time:timestamp"].max(), origin, win_minutes)
    return pd.DataFrame(
        {
            "active_cases": df.groupby("__win__")["case:concept:name"].nunique().reindex(win_idx, fill_value=0),
            "new_arrivals": first.value_counts().reindex(win_idx, fill_value=0),
            "completions": last.value_counts().reindex(win_idx, fill_value=0),
            "total_events": df.groupby("__win__").size().reindex(win_idx, fill_value=0),
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
    case_last = df.groupby("case:concept:name")["time:timestamp"].max()
    stalled = _stalled_count(case_last, win_idx, window_minutes, stall_minutes)

    out = counts.assign(mean_delta_t=deltas_mean, std_delta_t=deltas_std, stalled_cases=stalled.astype(float))
    attrs = _attribute_columns(df, win_idx, numeric_attrs, categorical_attrs)
    out = out.join(attrs)
    out.index.name = "window_start"
    groups = {
        "counts": ["active_cases", "total_events"],
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
