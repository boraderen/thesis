"""Resource features: one row per calendar window of size W minutes.

Each feature kind is its own function; `build` calls only the requested ones.
Share features (handovers, activity-resource shares) always use the full
resource pool in their denominators — a `resources` filter only selects which
columns come back, it never changes what a share means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import ACTIVITY, CASE, DURATION, RESOURCE, TIMESTAMP
from ..data.windows import assign_windows
from . import FeatureSet


def events_per_resource(
    df: pd.DataFrame, resources: list[str], win_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per window: number of events each resource executed."""
    pivot = df.groupby(["__win__", "__res__"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    pivot.columns = [f"events:{r}" for r in resources]
    return pivot.astype(float)


def active_cases_per_resource(
    df: pd.DataFrame, resources: list[str], win_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per window: number of distinct cases each resource touched."""
    pivot = df.groupby(["__win__", "__res__"])[CASE].nunique().unstack(fill_value=0)
    pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    pivot.columns = [f"active:{r}" for r in resources]
    return pivot.astype(float)


def mean_duration_per_resource(
    df: pd.DataFrame, resources: list[str], win_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per window: mean event duration in minutes per resource (0 without durations)."""
    if DURATION in df.columns:
        durations = pd.to_numeric(df[DURATION], errors="coerce").clip(lower=0)
        duration_df = df.assign(__duration_min__=durations).dropna(subset=["__duration_min__"])
        pivot = (
            duration_df.groupby(["__win__", "__res__"])["__duration_min__"].mean().unstack(fill_value=0)
            if not duration_df.empty
            else pd.DataFrame(0.0, index=win_idx, columns=resources)
        )
        pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    else:
        pivot = pd.DataFrame(0.0, index=win_idx, columns=resources)
    pivot.columns = [f"duration:{r}" for r in resources]
    return pivot.astype(float)


def mean_wait_per_resource(
    df: pd.DataFrame, resources: list[str], win_idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """Per window and resource r: mean minutes since the previous case event,
    counting only events whose previous event was executed by a different resource."""
    ordered = df.sort_values([CASE, TIMESTAMP]).copy()
    ordered["prev_res"] = ordered.groupby(CASE)["__res__"].shift(1)
    ordered["prev_ts"] = ordered.groupby(CASE)[TIMESTAMP].shift(1)
    crossings = ordered.dropna(subset=["prev_res"]).query("prev_res != __res__").copy()
    crossings["wait_min"] = (
        (crossings[TIMESTAMP] - crossings["prev_ts"]).dt.total_seconds().clip(lower=0) / 60.0
    )
    pivot = (
        crossings.groupby(["__win__", "__res__"])["wait_min"].mean().unstack(fill_value=0)
        if not crossings.empty
        else pd.DataFrame(0.0, index=win_idx, columns=resources)
    )
    pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=resources, fill_value=0)
    pivot.columns = [f"wait:{r}" for r in resources]
    return pivot.astype(float)


def activity_resource_shares(
    df: pd.DataFrame,
    activities: list[str],
    resources: list[str],
    win_idx: pd.DatetimeIndex,
    all_resources: list[str] | None = None,
) -> pd.DataFrame:
    """Per window: the share of each activity's events executed by each resource.

    ``activity_events:a:r`` = (a-events by r in the window) / (all a-events in
    the window, by anyone); 0 when the activity does not occur in the window.
    """
    pool = all_resources or resources
    if not activities or not pool:
        return pd.DataFrame(index=win_idx)
    cols = pd.MultiIndex.from_product([activities, pool], names=["__act__", "__res__"])
    counts = df.groupby(["__win__", "__act__", "__res__"]).size()
    pivot = counts.unstack(["__act__", "__res__"], fill_value=0)
    pivot = pivot.reindex(win_idx, fill_value=0).reindex(columns=cols, fill_value=0).astype(float)
    for activity in activities:
        totals = pivot[activity].sum(axis=1)
        pivot[activity] = pivot[activity].div(totals.where(totals > 0), axis=0).fillna(0.0)
    pivot.columns = [f"activity_events:{a}:{r}" for a, r in pivot.columns]
    keep = [f"activity_events:{a}:{r}" for a in activities for r in resources]
    return pivot[keep]


def handover_shares(
    df: pd.DataFrame,
    resources: list[str],
    win_idx: pd.DatetimeIndex,
    all_resources: list[str] | None = None,
) -> pd.DataFrame:
    """Per window: the share of r1's within-case handovers that go to r2.

    ``ho:r1→r2`` = (handovers r1→r2 in the window) / (all handovers from r1 in
    the window, to anyone); 0 when r1 hands nothing over in the window.
    """
    pool = all_resources or resources
    ordered = df.sort_values([CASE, TIMESTAMP]).copy()
    ordered["prev_res"] = ordered.groupby(CASE)["__res__"].shift(1)
    crossings = ordered.dropna(subset=["prev_res"]).query("prev_res != __res__")
    cols = [f"ho:{a}→{b}" for a in pool for b in pool if a != b]
    out = pd.DataFrame(0.0, index=win_idx, columns=cols)
    if not crossings.empty:
        grouped = crossings.groupby(["__win__", "prev_res", "__res__"]).size().reset_index(name="n")
        for _, row in grouped.iterrows():
            col = f"ho:{row['prev_res']}→{row['__res__']}"
            if col in out.columns and row["__win__"] in out.index:
                out.at[row["__win__"], col] = row["n"]
        for source in pool:
            source_cols = [f"ho:{source}→{b}" for b in pool if b != source]
            totals = out[source_cols].sum(axis=1)
            out[source_cols] = out[source_cols].div(totals.where(totals > 0), axis=0).fillna(0.0)
    keep = [f"ho:{a}→{b}" for a in resources for b in resources if a != b]
    return out[keep]


BUILDERS = ("events", "active", "duration", "wait", "activity_events", "ho")


def build(
    log: pd.DataFrame,
    features: tuple[str, ...] | list[str] | None = None,
    window_minutes: int = 60,
    resources: tuple[str, ...] | list[str] | None = None,
    activities: tuple[str, ...] | list[str] | None = None,
) -> FeatureSet:
    """One row per calendar window with the requested resource features.

    `resources` / `activities` restrict which columns are produced; share
    denominators always cover the full pool.
    """
    if RESOURCE not in log.columns:
        raise ValueError(f"No '{RESOURCE}' column in the log")
    requested = list(BUILDERS) if features is None else [f for f in BUILDERS if f in features]
    if not requested:
        raise ValueError("No known resource feature requested")
    df, win_idx = assign_windows(log, window_minutes)
    df["__res__"] = df[RESOURCE].astype(str)
    df["__act__"] = df[ACTIVITY].astype(str)
    all_resources = sorted(df["__res__"].dropna().unique().tolist())
    all_activities = sorted(df["__act__"].dropna().unique().tolist())
    picked_resources = [r for r in (resources or all_resources) if r in all_resources]
    picked_activities = [a for a in (activities or all_activities) if a in all_activities]

    parts: list[pd.DataFrame] = []
    groups: dict[str, list[str]] = {}
    for key in requested:
        if key == "events":
            part = events_per_resource(df, picked_resources, win_idx)
        elif key == "active":
            part = active_cases_per_resource(df, picked_resources, win_idx)
        elif key == "duration":
            part = mean_duration_per_resource(df, picked_resources, win_idx)
        elif key == "wait":
            part = mean_wait_per_resource(df, picked_resources, win_idx)
        elif key == "activity_events":
            part = activity_resource_shares(df, picked_activities, picked_resources, win_idx, all_resources)
        else:
            part = handover_shares(df, picked_resources, win_idx, all_resources)
        parts.append(part)
        groups[key] = part.columns.tolist()

    matrix = pd.concat(parts, axis=1)
    matrix.index.name = "window_start"
    index = matrix.index.to_frame(index=False)
    meta = {
        "resources": all_resources,
        "activities": all_activities,
        "picked_resources": picked_resources,
        "picked_activities": picked_activities,
        "window_minutes": int(window_minutes),
        "features": requested,
    }
    return FeatureSet(
        perspective="resource",
        matrix=matrix.reset_index(drop=True),
        index=index,
        groups=groups,
        meta=meta,
    )
