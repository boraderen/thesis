"""Resource feature extraction: one row per calendar window of size W minutes."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from core.windows import floor_to_window, window_index


@dataclass(frozen=True)
class ResourceSpec:
    """Column metadata for the resource feature matrix."""

    columns: list[str]
    groups: dict[str, list[str]]
    resources: list[str]
    activities: list[str]
    window_minutes: int
    window_starts: pd.DatetimeIndex


def _handover_shares(log: pd.DataFrame, resources: list[str]) -> pd.DataFrame:
    """Per window: share of r1's handovers that go to r2.

    ho:r1→r2 = (handovers r1→r2 in the window) / (all handovers from r1 in the
    window); 0 when r1 hands nothing over in the window.
    """
    df = log.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df["prev_res"] = df.groupby("case:concept:name")["__res__"].shift(1)
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
    for source in resources:
        source_cols = [f"ho:{source}→{b}" for b in resources if b != source]
        totals = out[source_cols].sum(axis=1)
        out[source_cols] = out[source_cols].div(totals.where(totals > 0), axis=0).fillna(0.0)
    return out.sort_index()


def _resource_mean_wait(
    df: pd.DataFrame, resources: list[str], win_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """For each window and resource r, mean minutes since the previous case event,
    counting only events whose previous event was executed by a different resource."""
    df = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    df["prev_res"] = df.groupby("case:concept:name")["__res__"].shift(1)
    df["prev_ts"] = df.groupby("case:concept:name")["time:timestamp"].shift(1)
    crossings = df.dropna(subset=["prev_res"]).query("prev_res != __res__").copy()
    crossings["wait_min"] = (
        (crossings["time:timestamp"] - crossings["prev_ts"]).dt.total_seconds().clip(lower=0) / 60.0
    )
    pivot = (
        crossings.groupby(["__win__", "__res__"])["wait_min"].mean().unstack(fill_value=0)
        if not crossings.empty
        else pd.DataFrame(0.0, index=win_idx, columns=resources)
    )
    pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    pivot.columns = [f"wait:{r}" for r in resources]
    return pivot


def _duration_minutes(df: pd.DataFrame) -> pd.Series | None:
    if "event:duration_min" not in df.columns:
        return None
    return pd.to_numeric(df["event:duration_min"], errors="coerce").clip(lower=0)


def _resource_mean_duration(
    df: pd.DataFrame, resources: list[str], win_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """For each window and resource r, mean event duration in minutes."""
    durations = _duration_minutes(df)
    if durations is None:
        pivot = pd.DataFrame(0.0, index=win_idx, columns=resources)
    else:
        duration_df = df.assign(__duration_min__=durations).dropna(subset=["__duration_min__"])
        pivot = (
            duration_df.groupby(["__win__", "__res__"])["__duration_min__"].mean().unstack(fill_value=0)
            if not duration_df.empty
            else pd.DataFrame(0.0, index=win_idx, columns=resources)
        )
        pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    pivot.columns = [f"duration:{r}" for r in resources]
    return pivot


def _activity_resource_counts(
    df: pd.DataFrame,
    activities: list[str],
    resources: list[str],
    win_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Per window: share of each activity's events executed by each resource.

    activity_events:a:r = (a-events by r in the window) / (all a-events in the
    window); 0 when the activity does not occur in the window.
    """
    if not activities or not resources:
        return pd.DataFrame(index=win_idx)
    cols = pd.MultiIndex.from_product([activities, resources], names=["__act__", "__res__"])
    counts = df.groupby(["__win__", "__act__", "__res__"]).size()
    pivot = counts.unstack(["__act__", "__res__"], fill_value=0)
    pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=cols, fill_value=0).astype(float)
    for activity in activities:
        totals = pivot[activity].sum(axis=1)
        pivot[activity] = pivot[activity].div(totals.where(totals > 0), axis=0).fillna(0.0)
    pivot.columns = [f"activity_events:{activity}:{resource}" for activity, resource in pivot.columns]
    return pivot


@st.cache_data(show_spinner=False)
def build_features(log: pd.DataFrame, window_minutes: int = 60) -> tuple[pd.DataFrame, ResourceSpec]:
    """Build per-window resource workload features (wait counts only events whose
    previous case event was by a different resource; activity-resource and
    handover features are within-window shares). Returns (DataFrame, spec)."""
    if "org:resource" not in log.columns:
        raise ValueError("No 'org:resource' column in the log")
    df = log.copy()
    df["__res__"] = df["org:resource"].astype(str)
    df["__act__"] = df["concept:name"].astype(str)
    origin = df["time:timestamp"].min()
    df["__win__"] = floor_to_window(df["time:timestamp"], origin, window_minutes)
    resources = sorted(df["__res__"].dropna().unique().tolist())
    activities = sorted(df["__act__"].dropna().unique().tolist())

    win_idx = window_index(origin, df["time:timestamp"].max(), window_minutes)

    events_by = df.groupby(["__win__", "__res__"]).size().unstack(fill_value=0).reindex(win_idx, fill_value=0)
    events_by = events_by.reindex(columns=resources, fill_value=0)
    events_by.columns = [f"events:{r}" for r in resources]

    active = df.groupby(["__win__", "__res__"])["case:concept:name"].nunique().unstack(fill_value=0)
    active = active.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    active.columns = [f"active:{r}" for r in resources]

    duration = _resource_mean_duration(df, resources, win_idx)
    wait = _resource_mean_wait(df, resources, win_idx)
    ho = _handover_shares(df, resources).reindex(win_idx, fill_value=0)
    activity_resource = _activity_resource_counts(df, activities, resources, win_idx)

    matrix = pd.concat([events_by, active, duration, wait, activity_resource, ho], axis=1)
    matrix.index.name = "window_start"

    groups: dict[str, list[str]] = {}
    for r in resources:
        tint = f"resource_{chr(ord('a') + (resources.index(r) % 3))}"
        groups.setdefault(tint, []).extend([f"events:{r}", f"active:{r}", f"duration:{r}", f"wait:{r}"])
    groups["activity_resource"] = [c for c in matrix.columns if c.startswith("activity_events:")]
    groups["handover"] = [c for c in matrix.columns if c.startswith("ho:")]

    spec = ResourceSpec(
        columns=matrix.columns.tolist(),
        groups=groups,
        resources=resources,
        activities=activities,
        window_minutes=window_minutes,
        window_starts=pd.DatetimeIndex(matrix.index),
    )
    return matrix.reset_index(), spec
