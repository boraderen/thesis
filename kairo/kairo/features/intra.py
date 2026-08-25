"""Intra-case features: one row per event with a prefix view of its case.

Each feature is its own function returning ``(block, column_names)``; `build`
calls only the requested ones, in the registry order, and stacks the blocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import ACTIVITY, CASE, TIMESTAMP
from . import FeatureSet


def activities_of(log: pd.DataFrame) -> list[str]:
    """The distinct activities of the log, in a stable sorted order."""
    return sorted(log[ACTIVITY].unique().tolist())


def transitions_of(log: pd.DataFrame) -> list[tuple[str, str]]:
    """The sorted directly-follows pairs observed inside cases."""
    prev = log.groupby(CASE)[ACTIVITY].shift(1)
    mask = prev.notna()
    pairs = set(zip(prev[mask], log.loc[mask, ACTIVITY]))
    return sorted(pairs)


def _one_hot(labels: pd.Series, activities: list[str]) -> np.ndarray:
    """One binary column per activity; a missing label (NaN) leaves an all-zero row."""
    position = {activity: i for i, activity in enumerate(activities)}
    block = np.zeros((len(labels), len(activities)), dtype=float)
    column = labels.map(position)
    rows = np.flatnonzero(column.notna().to_numpy())
    block[rows, column.dropna().to_numpy(dtype=int)] = 1.0
    return block


def _positions(log: pd.DataFrame) -> np.ndarray:
    """0-based position of each event within its case."""
    return log.groupby(CASE).cumcount().to_numpy()


def _case_cumsum(log: pd.DataFrame, block: np.ndarray) -> np.ndarray:
    """Cumulative sum of a per-event block within each case."""
    frame = pd.DataFrame(block)
    return frame.groupby(log[CASE].to_numpy(), sort=False).cumsum().to_numpy()


def activity_frequency(log: pd.DataFrame, activities: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per event: counts of each activity in the prefix, divided by the prefix length."""
    counts = _case_cumsum(log, _one_hot(log[ACTIVITY], activities))
    prefix_len = (_positions(log) + 1).reshape(-1, 1)
    return counts / prefix_len, [f"activity_freq:{a}" for a in activities]


def directly_follows(
    log: pd.DataFrame, transitions: list[tuple[str, str]]
) -> tuple[np.ndarray, list[str]]:
    """Per event: counts of each observed A→B pair in the prefix, divided by the
    number of transitions so far (0 while the case has a single event)."""
    prev = log.groupby(CASE)[ACTIVITY].shift(1)
    pair_index = {pair: i for i, pair in enumerate(transitions)}
    pairs = pd.Series(list(zip(prev, log[ACTIVITY])), index=log.index)
    pairs[prev.isna()] = np.nan
    block = np.zeros((len(log), len(transitions)), dtype=float)
    column = pairs.map(pair_index)
    rows = np.flatnonzero(column.notna().to_numpy())
    block[rows, column.dropna().to_numpy(dtype=int)] = 1.0
    counts = _case_cumsum(log, block)
    pos = _positions(log).reshape(-1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(pos > 0, counts / np.maximum(pos, 1), 0.0)
    return freq, [f"bigram:{a}→{b}" for a, b in transitions]


def vocabulary(log: pd.DataFrame, activities: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per event: 1 for every activity already seen in the case prefix."""
    seen = _case_cumsum(log, _one_hot(log[ACTIVITY], activities)) > 0
    return seen.astype(float), [f"vocab:{a}" for a in activities]


def progress(log: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Per event: its position within the case as a fraction of the case length."""
    case_len = log.groupby(CASE)[ACTIVITY].transform("size").to_numpy()
    ratio = (_positions(log) + 1) / case_len
    return ratio.reshape(-1, 1).astype(float), ["progress_ratio"]


def current_activity(log: pd.DataFrame, activities: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per event: one-hot of the event's own activity."""
    return _one_hot(log[ACTIVITY], activities), [f"current:{a}" for a in activities]


def past_activities(
    log: pd.DataFrame, activities: list[str], history: int = 3
) -> tuple[np.ndarray, list[str]]:
    """Per event: one-hot of the activities of the last `history` events of the case."""
    by_case = log.groupby(CASE)[ACTIVITY]
    blocks = [_one_hot(by_case.shift(lag), activities) for lag in range(1, history + 1)]
    columns = [f"prev{lag}:{a}" for lag in range(1, history + 1) for a in activities]
    return np.hstack(blocks) if blocks else np.zeros((len(log), 0)), columns


BUILDERS = ("activity_freq", "bigram", "vocab", "progress", "current", "history")


def build(
    log: pd.DataFrame,
    features: tuple[str, ...] | list[str] | None = None,
    history: int = 3,
) -> FeatureSet:
    """One row per event with the requested prefix-based intra-case features."""
    requested = list(BUILDERS) if features is None else [f for f in BUILDERS if f in features]
    if not requested:
        raise ValueError("No known intra-case feature requested")
    df = log.sort_values([CASE, TIMESTAMP]).reset_index(drop=True)
    activities = activities_of(df)
    transitions = transitions_of(df) if "bigram" in requested else []

    blocks: list[np.ndarray] = []
    groups: dict[str, list[str]] = {}
    for key in requested:
        if key == "activity_freq":
            block, cols = activity_frequency(df, activities)
        elif key == "bigram":
            block, cols = directly_follows(df, transitions)
        elif key == "vocab":
            block, cols = vocabulary(df, activities)
        elif key == "progress":
            block, cols = progress(df)
        elif key == "current":
            block, cols = current_activity(df, activities)
        else:
            block, cols = past_activities(df, activities, history=history)
        blocks.append(block)
        groups[key] = cols

    columns = [c for cols in groups.values() for c in cols]
    matrix = pd.DataFrame(np.hstack(blocks), columns=columns)
    index = df[[CASE, ACTIVITY, TIMESTAMP]].copy()
    meta = {
        "activities": activities,
        "transitions": transitions,
        "history": int(history),
        "features": requested,
    }
    return FeatureSet(perspective="intra_case", matrix=matrix, index=index, groups=groups, meta=meta)
