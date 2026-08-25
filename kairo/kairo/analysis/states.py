"""From clustered states to interpretable objects: assignments, trajectories,
transitions, per-window distributions, and state profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cluster import StateModel
from ..features import FeatureSet
from ..data.schema import CASE, TIMESTAMP
from ..data.windows import floor_to_window


def trajectories(fs: FeatureSet, model: StateModel) -> pd.DataFrame:
    """State over time: the FeatureSet's index columns plus each row's state.

    One row per event for intra-case (carrying case, activity and timestamp),
    one row per window for the windowed perspectives.
    """
    out = fs.index.copy().reset_index(drop=True)
    out["state_id"] = model.state_ids
    out["state"] = [model.labels[i] for i in model.state_ids]
    return out


def _top_changes(before: pd.Series, after: pd.Series, top_n: int) -> str:
    """A 'name: a→b (Δ)' string for the top_n features that moved most."""
    delta = (after - before).abs()
    top = delta.sort_values(ascending=False).head(top_n).index
    parts = []
    for col in top:
        a, b = float(before[col]), float(after[col])
        if a == b:
            continue
        parts.append(f"{col}: {a:.2f} → {b:.2f} ({b - a:+.2f})")
    return "; ".join(parts) if parts else "—"


def find_transitions(
    timestamps: pd.Series,
    state_ids: np.ndarray,
    labels: list[str],
    features: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """A row per state change with timestamp, labels, and dominant feature deltas."""
    columns = ["timestamp", "boundary", "from", "to", "from_idx", "to_idx", "top_changes"]
    if len(state_ids) < 2:
        return pd.DataFrame(columns=columns)
    changes = np.where(np.diff(state_ids) != 0)[0]
    feat = features.reset_index(drop=True)
    times = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    rows = []
    for idx in changes:
        src, dst = int(state_ids[idx]), int(state_ids[idx + 1])
        t_prev, t_next = times.iloc[idx], times.iloc[idx + 1]
        rows.append({
            "timestamp": t_next,
            "boundary": t_prev + (t_next - t_prev) / 2,
            "from": labels[src] if src < len(labels) else f"S{src}",
            "to": labels[dst] if dst < len(labels) else f"S{dst}",
            "from_idx": src,
            "to_idx": dst,
            "top_changes": _top_changes(feat.iloc[idx], feat.iloc[idx + 1], top_n),
        })
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def case_transitions(
    fs: FeatureSet, model: StateModel, columns: list[str] | None = None, top_n: int = 3
) -> pd.DataFrame:
    """All within-case transitions of an intra-case run, with a leading case column."""
    frame = fs.frame
    frame = frame.assign(state_id=model.state_ids)
    cols = columns or fs.columns
    parts = []
    for case, sub in frame.groupby(CASE, sort=False):
        sub = sub.reset_index(drop=True)
        found = find_transitions(sub[TIMESTAMP], sub["state_id"].to_numpy(), model.labels, sub[cols], top_n)
        if not found.empty:
            found.insert(0, "case", case)
            parts.append(found)
    if not parts:
        return pd.DataFrame(columns=["case", "timestamp", "boundary", "from", "to",
                                     "from_idx", "to_idx", "top_changes"])
    return pd.concat(parts, ignore_index=True)


def transition_matrix(
    state_ids: np.ndarray,
    n_states: int,
    labels: list[str] | None = None,
    case_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Counts of consecutive state pairs; with `case_ids`, only within-case pairs."""
    counts = np.zeros((n_states, n_states), dtype=int)
    src, dst = state_ids[:-1], state_ids[1:]
    keep = np.ones(len(src), dtype=bool)
    if case_ids is not None:
        keep = np.asarray(case_ids[:-1]) == np.asarray(case_ids[1:])
    for a, b in zip(src[keep], dst[keep]):
        counts[int(a), int(b)] += 1
    names = labels if labels and len(labels) == n_states else [f"S{i}" for i in range(n_states)]
    return pd.DataFrame(counts, index=names, columns=names)


def state_distribution(
    timestamps: pd.Series,
    state_ids: np.ndarray,
    n_states: int,
    window_minutes: int,
) -> pd.DataFrame:
    """Per calendar window: the fraction of samples that landed in each state.

    Columns are ``S0..S{n-1}`` plus ``window_start``.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), utc=True)
    df = pd.DataFrame({"ts": ts, "state_id": np.asarray(state_ids)})
    origin = df["ts"].min()
    df["__win__"] = floor_to_window(df["ts"], origin, window_minutes)
    counts = (
        df.groupby(["__win__", "state_id"]).size().unstack(fill_value=0)
        .reindex(columns=range(n_states), fill_value=0)
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    fractions = counts.div(totals, axis=0).fillna(0.0)
    fractions.columns = [f"S{s}" for s in fractions.columns]
    fractions.index.name = "window_start"
    return fractions.reset_index()


def state_profiles(fs: FeatureSet, model: StateModel, top_n: int = 5) -> pd.DataFrame:
    """What characterises each state: its most deviating features vs the overall mean.

    One row per (state, feature) for the top_n features whose state mean sits
    furthest from the overall mean, measured in overall standard deviations.
    """
    values = fs.matrix.to_numpy(dtype=float)
    overall_mean = values.mean(axis=0)
    overall_std = values.std(axis=0)
    safe_std = np.where(overall_std > 0, overall_std, 1.0)
    rows = []
    for state in range(model.n_states):
        mask = model.state_ids == state
        if not mask.any():
            continue
        deviation = (values[mask].mean(axis=0) - overall_mean) / safe_std
        order = np.argsort(-np.abs(deviation))[:top_n]
        for j in order:
            rows.append({
                "state_id": state,
                "state": model.labels[state],
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "feature": fs.columns[j],
                "state_mean": float(values[mask].mean(axis=0)[j]),
                "overall_mean": float(overall_mean[j]),
                "deviation": float(deviation[j]),
            })
    return pd.DataFrame(rows)
