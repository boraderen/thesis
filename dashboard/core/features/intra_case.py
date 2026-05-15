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
    act_to_idx = {a: i for i, a in enumerate(activities)}

    n = len(df)
    act_block = np.zeros((n, window * len(activities)), dtype=float)
    gap_block = np.zeros((n, window - 1), dtype=float) if window > 1 else np.zeros((n, 0), dtype=float)
    elapsed = np.zeros(n, dtype=float)

    case_starts = df.groupby("case_id")["timestamp"].transform("min")
    elapsed = _zlog_minutes((df["timestamp"] - case_starts).dt.total_seconds().to_numpy())

    case_groups = df.groupby("case_id", sort=False).indices
    for _, idxs in case_groups.items():
        acts = df["activity"].iloc[idxs].to_numpy()
        ts = df["timestamp"].iloc[idxs].to_numpy()
        for pos, row_idx in enumerate(idxs):
            for w in range(window):
                src = pos - (window - 1 - w)
                if src < 0:
                    continue
                a = acts[src]
                act_block[row_idx, w * len(activities) + act_to_idx[a]] = 1.0
            if window > 1:
                for g in range(window - 1):
                    src_next = pos - (window - 2 - g)
                    src_prev = src_next - 1
                    if src_prev < 0 or src_next < 0:
                        continue
                    delta = (ts[src_next] - ts[src_prev]) / np.timedelta64(1, "s")
                    gap_block[row_idx, g] = _zlog_minutes(np.array([delta]))[0]

    blocks = [act_block, gap_block, elapsed.reshape(-1, 1)]
    columns: list[str] = []
    groups: dict[str, list[str]] = {"activity": [], "delta": [], "elapsed": [], "case_attr": []}
    for w in range(window):
        for a in activities:
            name = f"act[t-{window - 1 - w}]={a}"
            columns.append(name)
            groups["activity"].append(name)
    for g in range(window - 1):
        name = f"Δt[t-{window - 2 - g}]"
        columns.append(name)
        groups["delta"].append(name)
    columns.append("elapsed")
    groups["elapsed"].append("elapsed")

    if numeric_attrs:
        num = df[list(numeric_attrs)].astype(float).to_numpy()
        denom = np.where(np.ptp(num, axis=0) == 0, 1.0, np.ptp(num, axis=0))
        norm = (num - num.min(axis=0)) / denom
        blocks.append(norm)
        for c in numeric_attrs:
            name = f"num:{c}"
            columns.append(name)
            groups["case_attr"].append(name)

    if categorical_attrs:
        dummies = pd.get_dummies(df[list(categorical_attrs)].astype(str), prefix=list(categorical_attrs))
        blocks.append(dummies.to_numpy(dtype=float))
        for c in dummies.columns:
            name = f"cat:{c}"
            columns.append(name)
            groups["case_attr"].append(name)

    matrix = np.hstack(blocks)
    feat = pd.DataFrame(matrix, columns=columns)
    feat.insert(0, "case_id", df["case_id"].values)
    feat.insert(1, "activity", df["activity"].values)
    feat.insert(2, "timestamp", df["timestamp"].values)
    spec = IntraSpec(columns=columns, groups=groups, activities=activities, window=window)
    return feat, spec
