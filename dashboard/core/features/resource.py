"""Resource feature extraction: one row per calendar window of size W minutes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from core.windows import floor_to_window, window_index


@dataclass(frozen=True)
class ResourceSpec:
    """Column metadata for the resource feature matrix."""

    columns: list[str]
    groups: dict[str, list[str]]
    resources: list[str]
    window_minutes: int
    window_starts: pd.DatetimeIndex


def _zlog_minutes(seconds: np.ndarray) -> np.ndarray:
    """Convert seconds to ln(minutes), zeros preserved."""
    return np.log1p(np.clip(seconds / 60.0, 0, None))


def _handover_counts(log: pd.DataFrame, resources: list[str]) -> pd.DataFrame:
    """Count r→r' handovers within each window."""
    df = log.sort_values(["case_id", "timestamp"]).copy()
    df["prev_res"] = df.groupby("case_id")["__res__"].shift(1)
    crossings = df.dropna(subset=["prev_res"]).query("prev_res != __res__")
    cols = [f"ho:{a}→{b}" for a in resources for b in resources if a != b]
    out = pd.DataFrame(0, index=df["__win__"].unique(), columns=cols, dtype=float)
    if crossings.empty:
        return out.sort_index()
    grouped = crossings.groupby(["__win__", "prev_res", "__res__"]).size().reset_index(name="n")
    for _, row in grouped.iterrows():
        col = f"ho:{row['prev_res']}→{row['__res__']}"
        if col in out.columns:
            out.at[row["__win__"], col] = row["n"]
    return out.sort_index()


def _resource_mean_wait(
    df: pd.DataFrame, resources: list[str], win_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """For each window and resource r, mean ln-minutes wait into r within the window."""
    df = df.sort_values(["case_id", "timestamp"]).copy()
    df["prev_res"] = df.groupby("case_id")["__res__"].shift(1)
    df["prev_ts"] = df.groupby("case_id")["timestamp"].shift(1)
    crossings = df.dropna(subset=["prev_res"]).query("prev_res != __res__").copy()
    crossings["wait_s"] = (crossings["timestamp"] - crossings["prev_ts"]).dt.total_seconds()
    crossings["wait_lnmin"] = _zlog_minutes(crossings["wait_s"].to_numpy())
    pivot = (
        crossings.groupby(["__win__", "__res__"])["wait_lnmin"].mean().unstack(fill_value=0)
        if not crossings.empty
        else pd.DataFrame(0.0, index=win_idx, columns=resources)
    )
    pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    pivot.columns = [f"wait:{r}" for r in resources]
    return pivot


@st.cache_data(show_spinner=False)
def build_features(log: pd.DataFrame, window_minutes: int = 60) -> tuple[pd.DataFrame, ResourceSpec]:
    """Build per-window resource workload features. Returns (DataFrame, spec)."""
    if "resource" not in log.columns:
        raise ValueError("No 'resource' column in the log")
    df = log.copy()
    df["__res__"] = df["resource"].astype(str)
    origin = df["timestamp"].min()
    df["__win__"] = floor_to_window(df["timestamp"], origin, window_minutes)
    resources = sorted(df["__res__"].dropna().unique().tolist())

    win_idx = window_index(origin, df["timestamp"].max(), window_minutes)

    events_by = df.groupby(["__win__", "__res__"]).size().unstack(fill_value=0).reindex(win_idx, fill_value=0)
    events_by = events_by.reindex(columns=resources, fill_value=0)
    events_by.columns = [f"events:{r}" for r in resources]

    active = df.groupby(["__win__", "__res__"])["case_id"].nunique().unstack(fill_value=0)
    active = active.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    active.columns = [f"active:{r}" for r in resources]

    wait = _resource_mean_wait(df, resources, win_idx)
    ho = _handover_counts(df, resources).reindex(win_idx, fill_value=0)

    matrix = pd.concat([events_by, active, wait, ho], axis=1)
    matrix.index.name = "window_start"

    groups: dict[str, list[str]] = {}
    for r in resources:
        tint = f"resource_{chr(ord('a') + (resources.index(r) % 3))}"
        groups.setdefault(tint, []).extend([f"events:{r}", f"active:{r}", f"wait:{r}"])
    groups["handover"] = [c for c in matrix.columns if c.startswith("ho:")]

    spec = ResourceSpec(
        columns=matrix.columns.tolist(),
        groups=groups,
        resources=resources,
        window_minutes=window_minutes,
        window_starts=pd.DatetimeIndex(matrix.index),
    )
    return matrix.reset_index(), spec
