"""Inter-case features: one row per calendar window with system-level signals.

Each feature is its own function; `build` calls only the requested ones. Mapped
case attributes contribute extra features (mean / std of numeric ones, value
shares of categorical ones).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import CASE, TIMESTAMP
from ..data.windows import assign_windows, floor_to_window
from . import FeatureSet

# Descriptive names for state centroids, keyed by the dominant feature.
CELL_NAMES = {
    "new_arrivals": "Burst opening",
    "completions": "Burst closing",
    "stalled_cases": "Stalling",
    "active_cases": "High load",
    "events_per_case": "Busy cases",
    "mean_delta_t": "Slow cases",
    "std_delta_t": "Irregular timing",
}


def active_cases(df: pd.DataFrame, win_idx: pd.DatetimeIndex) -> pd.Series:
    """Per window: distinct cases with at least one event in it."""
    return df.groupby("__win__")[CASE].nunique().reindex(win_idx, fill_value=0).astype(float)


def new_arrivals(df: pd.DataFrame, win_idx: pd.DatetimeIndex) -> pd.Series:
    """Per window: cases whose first event falls inside it."""
    origin, minutes = win_idx[0], _window_minutes(win_idx)
    first = floor_to_window(df.groupby(CASE)[TIMESTAMP].min(), origin, minutes)
    return first.value_counts().reindex(win_idx, fill_value=0).astype(float)


def completions(df: pd.DataFrame, win_idx: pd.DatetimeIndex) -> pd.Series:
    """Per window: cases whose last event falls inside it."""
    origin, minutes = win_idx[0], _window_minutes(win_idx)
    last = floor_to_window(df.groupby(CASE)[TIMESTAMP].max(), origin, minutes)
    return last.value_counts().reindex(win_idx, fill_value=0).astype(float)


def events_per_case(df: pd.DataFrame, win_idx: pd.DatetimeIndex) -> pd.Series:
    """Per window: events divided by the number of cases active in it."""
    active = df.groupby("__win__")[CASE].nunique().reindex(win_idx, fill_value=0)
    events = df.groupby("__win__").size().reindex(win_idx, fill_value=0)
    return events.div(active.replace(0, np.nan)).fillna(0.0).astype(float)


def _case_gaps_minutes(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Within-case gaps in minutes and the window of each gap's later event."""
    ordered = df.sort_values([CASE, TIMESTAMP])
    gap_s = ordered.groupby(CASE)[TIMESTAMP].diff().dt.total_seconds()
    mask = gap_s.notna()
    minutes = (gap_s[mask] / 60.0).clip(lower=0)
    return minutes, ordered.loc[mask, "__win__"]


def mean_delta_t(df: pd.DataFrame, win_idx: pd.DatetimeIndex) -> pd.Series:
    """Per window: mean gap to the previous event of the same case, in minutes.

    Each gap is credited to the window of its later event, so a window measures
    how fast the cases running in it moved.
    """
    minutes, wins = _case_gaps_minutes(df)
    return minutes.groupby(wins).mean().reindex(win_idx).fillna(0.0).astype(float)


def std_delta_t(df: pd.DataFrame, win_idx: pd.DatetimeIndex) -> pd.Series:
    """Per window: standard deviation of the within-case gaps, in minutes."""
    minutes, wins = _case_gaps_minutes(df)
    return minutes.groupby(wins).std().reindex(win_idx).fillna(0.0).astype(float)


def _epoch_ns(values) -> np.ndarray:
    """Timestamps as int64 nanoseconds, whatever resolution the log carries."""
    return pd.DatetimeIndex(values).as_unit("ns").view("int64")


def _window_minutes(win_idx: pd.DatetimeIndex) -> int:
    if len(win_idx) > 1:
        return int((win_idx[1] - win_idx[0]).total_seconds() / 60)
    return 1


def stalled_cases(
    df: pd.DataFrame, win_idx: pd.DatetimeIndex, stall_minutes: int = 60
) -> pd.Series:
    """Per window end: still-running cases idle for longer than τ = stall_minutes.

    A case waits between two of its consecutive events, and its last event
    completes it — so only the gaps between consecutive events can stall, and a
    completed case is never counted. A stalling gap contributes to every window
    end that falls more than τ after its earlier event and before its later one.
    """
    threshold = pd.Timedelta(minutes=stall_minutes)
    ordered = df.sort_values([CASE, TIMESTAMP])
    current = ordered[TIMESTAMP]
    following = ordered.groupby(CASE)[TIMESTAMP].shift(-1)
    stalls = (following - current) > threshold  # a case's last event has no gap: NaT → False
    onset = np.sort(_epoch_ns(current[stalls] + threshold))
    resumes = np.sort(_epoch_ns(following[stalls]))
    win_end = _epoch_ns(win_idx + pd.Timedelta(minutes=_window_minutes(win_idx)))
    counts = (
        np.searchsorted(onset, win_end, side="left")
        - np.searchsorted(resumes, win_end, side="right")
    )
    return pd.Series(counts.astype(float), index=win_idx)


def attribute_mean(df: pd.DataFrame, win_idx: pd.DatetimeIndex, column: str) -> pd.Series:
    """Per window: mean of a numeric case attribute over the window's events."""
    values = pd.to_numeric(df[column], errors="coerce")
    return values.groupby(df["__win__"]).mean().reindex(win_idx).fillna(0.0).astype(float)


def attribute_std(df: pd.DataFrame, win_idx: pd.DatetimeIndex, column: str) -> pd.Series:
    """Per window: standard deviation of a numeric case attribute."""
    values = pd.to_numeric(df[column], errors="coerce")
    return values.groupby(df["__win__"]).std().reindex(win_idx).fillna(0.0).astype(float)


def attribute_shares(df: pd.DataFrame, win_idx: pd.DatetimeIndex, column: str) -> pd.DataFrame:
    """Per window: the share of events carrying each value of a categorical attribute."""
    values = sorted(df[column].dropna().astype(str).unique().tolist())
    counts = (
        pd.crosstab(df["__win__"], df[column].astype(str))
        .reindex(index=win_idx, fill_value=0)
        .reindex(columns=values, fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    shares.columns = [f"attr_share:{column}={v}" for v in values]
    return shares.astype(float)


def attribute_feature_catalog(
    log: pd.DataFrame,
    numeric_attrs: tuple[str, ...] | list[str],
    categorical_attrs: tuple[str, ...] | list[str],
) -> dict[str, tuple[str, list[str]]]:
    """Selectable case-attribute features: key -> (label, matrix columns).

    A numeric attribute offers its mean and std separately; a categorical one
    offers a single entry covering all its value-share columns, since the
    shares are one distribution and only mean something together.
    """
    catalog: dict[str, tuple[str, list[str]]] = {}
    for col in numeric_attrs:
        catalog[f"attr_mean:{col}"] = (f"Mean {col}", [f"attr_mean:{col}"])
        catalog[f"attr_std:{col}"] = (f"Std {col}", [f"attr_std:{col}"])
    for col in categorical_attrs:
        values = sorted(log[col].dropna().astype(str).unique().tolist())
        catalog[f"attr_share:{col}"] = (
            f"{col} value shares",
            [f"attr_share:{col}={v}" for v in values],
        )
    return catalog


def describe_states(centroids: np.ndarray, columns: list[str]) -> list[str]:
    """Map state centroid vectors to short descriptive names via the dominant feature."""
    if not columns:
        return ["—"] * len(centroids)
    names = []
    for centroid in centroids:
        top = columns[int(np.argmax(centroid))]
        names.append(CELL_NAMES.get(top, "Steady flow"))
    return names


SYSTEM_BUILDERS = (
    "active_cases", "new_arrivals", "completions", "events_per_case",
    "mean_delta_t", "std_delta_t", "stalled_cases",
)


def build(
    log: pd.DataFrame,
    features: tuple[str, ...] | list[str] | None = None,
    window_minutes: int = 60,
    stall_minutes: int = 60,
    numeric_attrs: tuple[str, ...] = (),
    categorical_attrs: tuple[str, ...] = (),
) -> FeatureSet:
    """One row per calendar window with the requested inter-case features.

    `features` may mix system keys (``active_cases`` …) and attribute keys
    (``attr_mean:amount``, ``attr_share:region`` …); None selects everything
    the given attributes allow.
    """
    catalog = attribute_feature_catalog(log, numeric_attrs, categorical_attrs)
    if features is None:
        requested = [*SYSTEM_BUILDERS, *catalog]
    else:
        requested = [f for f in features if f in SYSTEM_BUILDERS or f in catalog]
    if not requested:
        raise ValueError("No known inter-case feature requested")

    df, win_idx = assign_windows(log.sort_values(TIMESTAMP), window_minutes)
    parts: list[pd.Series | pd.DataFrame] = []
    groups: dict[str, list[str]] = {}
    for key in requested:
        if key in SYSTEM_BUILDERS:
            fn = {
                "active_cases": active_cases,
                "new_arrivals": new_arrivals,
                "completions": completions,
                "events_per_case": events_per_case,
                "mean_delta_t": mean_delta_t,
                "std_delta_t": std_delta_t,
            }.get(key)
            part = fn(df, win_idx) if fn else stalled_cases(df, win_idx, stall_minutes)
            part = part.rename(key)
            groups[key] = [key]
        else:
            kind, column = key.split(":", 1)
            if kind == "attr_mean":
                part = attribute_mean(df, win_idx, column).rename(key)
            elif kind == "attr_std":
                part = attribute_std(df, win_idx, column).rename(key)
            else:
                part = attribute_shares(df, win_idx, column)
            groups[key] = [part.name] if isinstance(part, pd.Series) else part.columns.tolist()
        parts.append(part)

    matrix = pd.concat(parts, axis=1)
    matrix.index.name = "window_start"
    index = matrix.index.to_frame(index=False)
    meta = {
        "window_minutes": int(window_minutes),
        "stall_minutes": int(stall_minutes),
        "numeric_attrs": list(numeric_attrs),
        "categorical_attrs": list(categorical_attrs),
        "features": requested,
        "attribute_catalog": {k: label for k, (label, _) in catalog.items()},
    }
    return FeatureSet(
        perspective="inter_case",
        matrix=matrix.reset_index(drop=True),
        index=index,
        groups=groups,
        meta=meta,
    )
