"""Intra-case feature extraction: one row per event with a prefix view of the case."""
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
    transitions: list[tuple[str, str]]


def _directly_follows_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return the sorted directly-follows pairs observed inside cases."""
    pairs: set[tuple[str, str]] = set()
    for _, idxs in df.groupby("case:concept:name", sort=False).indices.items():
        acts = df["concept:name"].iloc[idxs].to_numpy()
        pairs.update(zip(acts[:-1], acts[1:]))
    return sorted(pairs)


def _prefix_blocks(
    df: pd.DataFrame, activities: list[str], transitions: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill activity frequency, bigram frequency, vocabulary, and progress blocks."""
    n = len(df)
    act_to_idx = {activity: i for i, activity in enumerate(activities)}
    pair_to_idx = {pair: i for i, pair in enumerate(transitions)}
    activity_freq = np.zeros((n, len(activities)), dtype=float)
    bigram_freq = np.zeros((n, len(transitions)), dtype=float)
    vocabulary = np.zeros((n, len(activities)), dtype=float)
    progress = np.zeros((n, 1), dtype=float)

    for _, idxs in df.groupby("case:concept:name", sort=False).indices.items():
        acts = df["concept:name"].iloc[idxs].to_numpy()
        act_counts = np.zeros(len(activities), dtype=float)
        pair_counts = np.zeros(len(transitions), dtype=float)
        case_len = len(idxs)

        for pos, row_idx in enumerate(idxs):
            act_counts[act_to_idx[acts[pos]]] += 1.0
            if pos > 0:
                pair_counts[pair_to_idx[(acts[pos - 1], acts[pos])]] += 1.0

            activity_freq[row_idx] = act_counts / float(pos + 1)
            if pos > 0:
                bigram_freq[row_idx] = pair_counts / float(pos)
            vocabulary[row_idx] = (act_counts > 0).astype(float)
            progress[row_idx, 0] = float(pos + 1) / float(case_len)

    return activity_freq, bigram_freq, vocabulary, progress


def _one_hot(labels: pd.Series, activities: list[str]) -> np.ndarray:
    """One binary column per activity; a missing label (NaN) leaves an all-zero row."""
    position = {activity: i for i, activity in enumerate(activities)}
    block = np.zeros((len(labels), len(activities)), dtype=float)
    column = labels.map(position)
    rows = np.flatnonzero(column.notna().to_numpy())
    block[rows, column.dropna().to_numpy(dtype=int)] = 1.0
    return block


@st.cache_data(show_spinner=False)
def build_features(log: pd.DataFrame, history: int = 3) -> tuple[pd.DataFrame, IntraSpec]:
    """Return one row per event with prefix-based intra-case features.

    `history` is the sliding-window size: the activities of the last `history`
    events of the same case are one-hot encoded alongside the current one.
    """
    df = log.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    activities = sorted(df["concept:name"].unique().tolist())
    transitions = _directly_follows_pairs(df)

    activity_freq, bigram_freq, vocabulary, progress = _prefix_blocks(df, activities, transitions)
    current = _one_hot(df["concept:name"], activities)
    by_case = df.groupby("case:concept:name")["concept:name"]
    past = [_one_hot(by_case.shift(lag), activities) for lag in range(1, history + 1)]

    activity_cols = [f"activity_freq:{activity}" for activity in activities]
    bigram_cols = [f"bigram:{src}→{dst}" for src, dst in transitions]
    vocab_cols = [f"vocab:{activity}" for activity in activities]
    progress_cols = ["progress_ratio"]
    current_cols = [f"current:{activity}" for activity in activities]
    history_cols = [
        f"prev{lag}:{activity}" for lag in range(1, history + 1) for activity in activities
    ]
    columns = (
        activity_cols + bigram_cols + vocab_cols + progress_cols + current_cols + history_cols
    )
    groups = {
        "activity_freq": activity_cols,
        "bigram": bigram_cols,
        "vocab": vocab_cols,
        "progress": progress_cols,
        "current": current_cols,
        "history": history_cols,
    }

    matrix = np.hstack([activity_freq, bigram_freq, vocabulary, progress, current, *past])
    feat = pd.DataFrame(matrix, columns=columns)
    feat.insert(0, "case:concept:name", df["case:concept:name"].values)
    feat.insert(1, "concept:name", df["concept:name"].values)
    feat.insert(2, "time:timestamp", df["time:timestamp"].values)
    spec = IntraSpec(columns=columns, groups=groups, activities=activities, transitions=transitions)
    return feat, spec
