"""Intra-case feature extraction: one row per event with a windowed view of the case."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class IntraSpec:
    """Names and column-group ranges for the intra-case feature matrix."""

    columns: list[str]
    groups: dict[str, list[str]]
    activities: list[str]
    window: int


def _zlog_minutes(seconds: np.ndarray) -> np.ndarray:
    """Convert seconds to ln(minutes), preserving zeros."""
    mins = np.clip(seconds / 60.0, 0, None)
    return np.log1p(mins)


def _windowed_blocks(df: pd.DataFrame, activities: list[str], window: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill the per-event activity one-hot and Δt gap blocks for a sorted log."""
    n = len(df)
    act_to_idx = {a: i for i, a in enumerate(activities)}
    act_block = np.zeros((n, window * len(activities)), dtype=float)
    gap_block = np.zeros((n, max(0, window - 1)), dtype=float)
    for _, idxs in df.groupby("case_id", sort=False).indices.items():
        acts = df["activity"].iloc[idxs].to_numpy()
        ts = df["timestamp"].iloc[idxs].to_numpy()
        for pos, row_idx in enumerate(idxs):
            for w in range(window):
                src = pos - (window - 1 - w)
                if src >= 0:
                    act_block[row_idx, w * len(activities) + act_to_idx[acts[src]]] = 1.0
            for g in range(window - 1):
                src_next = pos - (window - 2 - g)
                src_prev = src_next - 1
                if src_prev < 0 or src_next < 0:
                    continue
                delta_s = (ts[src_next] - ts[src_prev]) / np.timedelta64(1, "s")
                gap_block[row_idx, g] = _zlog_minutes(np.array([delta_s]))[0]
    return act_block, gap_block


def _name_window_cols(activities: list[str], window: int) -> tuple[list[str], list[str]]:
    """Generate the column names for activity one-hot and Δt blocks."""
    act_names = [f"act[t-{window - 1 - w}]={a}" for w in range(window) for a in activities]
    delta_names = [f"Δt[t-{window - 2 - g}]" for g in range(window - 1)]
    return act_names, delta_names


def _attr_blocks(
    df: pd.DataFrame, numeric_attrs: tuple[str, ...], categorical_attrs: tuple[str, ...]
) -> tuple[list[np.ndarray], list[str]]:
    """Return blocks + column names for normalised numeric and one-hot categorical case attributes."""
    blocks: list[np.ndarray] = []
    names: list[str] = []
    if numeric_attrs:
        num = df[list(numeric_attrs)].astype(float).to_numpy()
        denom = np.where(np.ptp(num, axis=0) == 0, 1.0, np.ptp(num, axis=0))
        blocks.append((num - num.min(axis=0)) / denom)
        names.extend(f"num:{c}" for c in numeric_attrs)
    if categorical_attrs:
        dummies = pd.get_dummies(
            df[list(categorical_attrs)].astype(str), prefix=list(categorical_attrs)
        )
        blocks.append(dummies.to_numpy(dtype=float))
        names.extend(f"cat:{c}" for c in dummies.columns)
    return blocks, names


@st.cache_data(show_spinner=False)
def build_features(
    log: pd.DataFrame,
    window: int = 3,
    numeric_attrs: tuple[str, ...] = (),
    categorical_attrs: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, IntraSpec]:
    """Return one row per event with windowed activity, gap, and case features."""
    df = log.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    activities = sorted(df["activity"].unique().tolist())

    act_block, gap_block = _windowed_blocks(df, activities, window)
    case_starts = df.groupby("case_id")["timestamp"].transform("min")
    elapsed = _zlog_minutes((df["timestamp"] - case_starts).dt.total_seconds().to_numpy()).reshape(-1, 1)
    attr_blocks, attr_cols = _attr_blocks(df, numeric_attrs, categorical_attrs)

    act_names, delta_names = _name_window_cols(activities, window)
    columns = act_names + delta_names + ["elapsed"] + attr_cols
    groups = {
        "activity": list(act_names),
        "delta": list(delta_names),
        "elapsed": ["elapsed"],
        "case_attr": list(attr_cols),
    }

    matrix = np.hstack([act_block, gap_block, elapsed, *attr_blocks])
    feat = pd.DataFrame(matrix, columns=columns)
    feat.insert(0, "case_id", df["case_id"].values)
    feat.insert(1, "activity", df["activity"].values)
    feat.insert(2, "timestamp", df["timestamp"].values)
    spec = IntraSpec(columns=columns, groups=groups, activities=activities, window=window)
    return feat, spec
