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
    for _, idxs in df.groupby("case_id", sort=False).indices.items():
        acts = df["activity"].iloc[idxs].to_numpy()
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

    for _, idxs in df.groupby("case_id", sort=False).indices.items():
        acts = df["activity"].iloc[idxs].to_numpy()
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


@st.cache_data(show_spinner=False)
def build_features(log: pd.DataFrame) -> tuple[pd.DataFrame, IntraSpec]:
    """Return one row per event with prefix-based intra-case features."""
    df = log.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    activities = sorted(df["activity"].unique().tolist())
    transitions = _directly_follows_pairs(df)

    activity_freq, bigram_freq, vocabulary, progress = _prefix_blocks(df, activities, transitions)
    activity_cols = [f"activity_freq:{activity}" for activity in activities]
    bigram_cols = [f"bigram:{src}→{dst}" for src, dst in transitions]
    vocab_cols = [f"vocab:{activity}" for activity in activities]
    progress_cols = ["progress_ratio"]
    columns = activity_cols + bigram_cols + vocab_cols + progress_cols
    groups = {
        "activity_freq": activity_cols,
        "bigram": bigram_cols,
        "vocab": vocab_cols,
        "progress": progress_cols,
    }

    matrix = np.hstack([activity_freq, bigram_freq, vocabulary, progress])
    feat = pd.DataFrame(matrix, columns=columns)
    feat.insert(0, "case_id", df["case_id"].values)
    feat.insert(1, "activity", df["activity"].values)
    feat.insert(2, "timestamp", df["timestamp"].values)
    spec = IntraSpec(columns=columns, groups=groups, activities=activities, transitions=transitions)
    return feat, spec
