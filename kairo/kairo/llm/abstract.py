"""Textual abstractions: turn computed objects into strings for LLM prompting.

Every function follows the same contract as pm4py's abstractions: it returns a
plain string, makes no network call, and truncates itself to `max_len`
characters (with a marker when it does).
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from ..analysis.cluster import StateModel, state_distances
from ..features import FeatureSet
from ..data.log import LogStatistics
from ..analysis.reduce import PCAResult
from ..data.schema import CASE, FEATURE_LABELS

MAX_LEN = 10000


_TRUNCATED = "\n... [truncated]"


def _clip(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - len(_TRUNCATED))].rstrip() + _TRUNCATED


def _header(text: str, include_header: bool) -> list[str]:
    return [text] if include_header else []


def abstract_log_statistics(
    stats: LogStatistics, max_len: int = MAX_LEN, include_header: bool = True
) -> str:
    """The basic statistics of the log as text."""
    lines = _header("Event log statistics:", include_header)
    lines += [
        f"cases: {stats.cases}, events: {stats.events}, activities: {stats.activities}, "
        f"resources: {stats.resources}",
        f"time span: {stats.start} → {stats.end} ({stats.span_minutes:.0f} minutes)",
        f"throughput time (days): min {stats.tpt_days_min:.2f}, mean {stats.tpt_days_mean:.2f}, "
        f"median {stats.tpt_days_median:.2f}, max {stats.tpt_days_max:.2f}",
        f"case length (events): min {stats.length_min}, mean {stats.length_mean:.1f}, "
        f"median {stats.length_median:.1f}, max {stats.length_max}",
    ]
    top = stats.activity_counts.head(15)
    lines.append("activity frequencies: " + ", ".join(f"{a} ({n})" for a, n in top.items()))
    if stats.resource_counts is not None:
        top_r = stats.resource_counts.head(15)
        lines.append("resource frequencies: " + ", ".join(f"{r} ({n})" for r, n in top_r.items()))
    if stats.case_attributes:
        lines.append("case attributes: " + ", ".join(
            f"{col} ({kind})" for col, kind in stats.case_attributes.items()))
    return _clip("\n".join(lines), max_len)


def abstract_log_attributes(
    log: pd.DataFrame, max_len: int = MAX_LEN, include_header: bool = True
) -> str:
    """The columns of the log with dtypes, quantiles, and top categorical values.

    Numeric and date columns come first with their quantiles; then categorical
    value-frequency pairs fill the remaining budget by descending count.
    """
    lines = _header("Log columns (dtype, then quantiles or top values):", include_header)
    body: dict[str, str] = {}
    frame = log[[c for c in log.columns if c != CASE]]
    for col in frame.columns:
        dtype = str(frame[col].dtype)
        empty = int(frame[col].isna().sum())
        if any(t in dtype for t in ("float", "int", "date")):
            try:
                quantiles = frame[col].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
                body[col] = f"{dtype}, empty: {empty}, quantiles: {quantiles}"
            except Exception:
                body[col] = f"{dtype}, empty: {empty}"
    values = []
    for col in frame.columns:
        if col in body:
            continue
        for value, count in frame[col].value_counts().items():
            values.append((col, value, int(count)))
    values.sort(key=lambda v: v[2], reverse=True)
    used = sum(len(k) + len(v) for k, v in body.items())
    for col, value, count in values:
        if used >= max_len:
            break
        entry = f" ({value}; freq. {count})"
        if col not in body:
            body[col] = f"{frame[col].dtype}, empty: {int(frame[col].isna().sum())}, values:"
            used += len(col) + len(body[col])
        body[col] += entry
        used += len(entry)
    lines += [f"{col}: {desc}" for col, desc in body.items()]
    return _clip("\n".join(lines), max_len)


def abstract_features(fs: FeatureSet, max_len: int = MAX_LEN, include_header: bool = True) -> str:
    """The feature matrix: shape, groups, and per-column value ranges."""
    labels = FEATURE_LABELS.get(fs.perspective, {})
    lines = _header(
        f"{fs.perspective} feature matrix: {len(fs.matrix)} rows × {len(fs.columns)} columns.",
        include_header,
    )
    for key, cols in fs.groups.items():
        label = labels.get(key.split(":", 1)[0], key)
        lines.append(f"- {key} ({label}): {len(cols)} columns")
    lines.append("Column ranges (min → max, mean):")
    described = fs.matrix.describe().T
    for col, row in described.iterrows():
        lines.append(f"  {col}: {row['min']:.3f} → {row['max']:.3f}, mean {row['mean']:.3f}")
        if sum(len(x) for x in lines) > max_len:
            break
    return _clip("\n".join(lines), max_len)


def abstract_pca(
    pca: PCAResult | None,
    columns: list[str] | None = None,
    max_len: int = MAX_LEN,
    include_header: bool = True,
) -> str:
    """The PCA reduction: kept components, explained variance, top loadings."""
    if pca is None:
        return "PCA was skipped — clustering ran on the raw feature columns."
    kept = pca.explained_variance_ratio[: pca.n_components]
    lines = _header(
        f"PCA reduced {pca.raw_dim} feature columns to {pca.n_components} components "
        f"({float(np.sum(kept)):.1%} of variance).",
        include_header,
    )
    lines.append("Explained variance per component: "
                 + ", ".join(f"PC{i + 1}={r:.3f}" for i, r in enumerate(kept)))
    if columns:
        lines.append("Strongest loadings per component:")
        loadings = pca.top_loadings(columns, k=4)
        for pc, group in loadings.groupby("component"):
            entries = ", ".join(f"{r.column} ({r.loading:+.2f})" for r in group.itertuples())
            lines.append(f"  PC{pc}: {entries}")
    return _clip("\n".join(lines), max_len)


def abstract_states(
    model: StateModel,
    profiles: pd.DataFrame | None = None,
    max_len: int = MAX_LEN,
    include_header: bool = True,
) -> str:
    """The clustered states: method, geometry, per-state size and character."""
    geometry = f"{model.grid_h}×{model.grid_w} grid" if model.method == "som" else f"{model.n_states} states"
    total = max(1, int(model.counts.sum()))
    lines = _header(f"States from {model.method} ({geometry}), parameters {model.params}:", include_header)
    for state in range(model.n_states):
        share = model.counts[state] / total
        dom = model.dominant[state] if state < len(model.dominant) else ""
        entry = f"- {model.labels[state]}: {int(model.counts[state])} samples ({share:.1%})"
        if dom:
            entry += f", dominant: {dom}"
        lines.append(entry)
    if profiles is not None and not profiles.empty:
        lines.append("Distinguishing features per state (deviation from the overall mean, in stds):")
        for (state, _), group in profiles.groupby(["state", "state_id"], sort=True):
            entries = ", ".join(f"{r.feature} ({r.deviation:+.1f}σ)" for r in group.itertuples())
            lines.append(f"  {state}: {entries}")
    return _clip("\n".join(lines), max_len)


def abstract_state_grid(model: StateModel, max_len: int = MAX_LEN, include_header: bool = True) -> str:
    """The SOM grid as a text matrix with counts, plus neighbour distances."""
    lines = _header(f"State grid ({model.grid_h} rows × {model.grid_w} columns):", include_header)
    for r in range(model.grid_h):
        row = []
        for c in range(model.grid_w):
            state = r * model.grid_w + c
            row.append(f"{model.labels[state]}(n={int(model.counts[state])})")
        lines.append("  " + " | ".join(row))
    distances = state_distances(model)
    n = model.n_states
    if n > 1:
        off = distances[~np.eye(n, dtype=bool)]
        lines.append(
            f"State-vector distances: closest pair {off.min():.3f}, "
            f"median {np.median(off):.3f}, farthest {off.max():.3f}."
        )
    return _clip("\n".join(lines), max_len)


def abstract_state_profiles(
    profiles: pd.DataFrame, max_len: int = MAX_LEN, include_header: bool = True
) -> str:
    """Per state: size and the features that distinguish it from the rest."""
    if profiles.empty:
        return "No state profiles available."
    lines = _header("State profiles (top deviating features vs the overall mean):", include_header)
    for (state, _), group in profiles.groupby(["state", "state_id"], sort=True):
        first = group.iloc[0]
        lines.append(f"- {state} ({int(first['n'])} samples, {first['share']:.1%}):")
        for row in group.itertuples():
            lines.append(
                f"    {row.feature}: state mean {row.state_mean:.3f} vs overall "
                f"{row.overall_mean:.3f} ({row.deviation:+.1f}σ)"
            )
    return _clip("\n".join(lines), max_len)


def abstract_trajectory(
    result, case_id: str | None = None, max_len: int = MAX_LEN, include_header: bool = True
) -> str:
    """One trajectory as a state sequence with timestamps and dwell lengths.

    For intra-case results pass a `case_id` (default: the first case); windowed
    perspectives have a single global trajectory.
    """
    traj = result.trajectories
    if result.perspective == "intra_case":
        if case_id is None:
            case_id = traj[CASE].iloc[0]
        traj = traj[traj[CASE] == str(case_id)].reset_index(drop=True)
        subject = f"case {case_id}"
        time_col = traj.columns[2]
    else:
        subject = f"the log in {result.config.window_minutes}-minute windows"
        time_col = "window_start"
    lines = _header(f"State trajectory of {subject}:", include_header)
    ids = traj["state_id"].to_numpy()
    times = traj[time_col].tolist()
    if len(ids) == 0:
        return "Empty trajectory."
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            label = result.states.labels[int(ids[start])]
            lines.append(f"- {times[start]} → {label} ({i - start} sample(s))")
            start = i
    return _clip("\n".join(lines), max_len)


def abstract_transitions(
    transitions: pd.DataFrame, max_len: int = MAX_LEN, include_header: bool = True
) -> str:
    """The observed state transitions with their dominant feature changes."""
    if transitions.empty:
        return "No state transitions were observed."
    lines = _header(f"{len(transitions)} state transitions (from → to, top feature changes):",
                    include_header)
    # "from" is a Python keyword, which itertuples would mangle — rename first.
    renamed = transitions.rename(columns={"from": "src", "to": "dst"})
    for row in renamed.itertuples():
        prefix = f"[case {row.case}] " if hasattr(row, "case") else ""
        lines.append(f"- {prefix}{row.timestamp}: {row.src} → {row.dst}; {row.top_changes}")
        if sum(len(x) for x in lines) > max_len:
            break
    return _clip("\n".join(lines), max_len)


def abstract_transition_matrix(
    matrix: pd.DataFrame, max_len: int = MAX_LEN, include_header: bool = True
) -> str:
    """The state-to-state transition counts, one line per non-zero pair."""
    lines = _header("State transition counts:", include_header)
    for src in matrix.index:
        for dst in matrix.columns:
            count = int(matrix.loc[src, dst])
            if count > 0:
                lines.append(f"- {src} → {dst}: {count}")
    if len(lines) == int(bool(include_header)):
        lines.append("(no transitions)")
    return _clip("\n".join(lines), max_len)


def abstract_state_distribution(
    distribution: pd.DataFrame, max_len: int = MAX_LEN, include_header: bool = True
) -> str:
    """The per-window state frequency table as text rows."""
    cols = [c for c in distribution.columns if c.startswith("S")]
    lines = _header(
        f"Per-window state distribution ({len(distribution)} windows × {len(cols)} states):",
        include_header,
    )
    for row in distribution.itertuples(index=False):
        shares = ", ".join(f"{c}={getattr(row, c):.2f}" for c in cols if getattr(row, c) > 0)
        lines.append(f"- {row.window_start}: {shares or 'empty'}")
        if sum(len(x) for x in lines) > max_len:
            break
    return _clip("\n".join(lines), max_len)


def abstract_drift_signal(
    signal: pd.DataFrame, top_k: int = 5, max_len: int = MAX_LEN, include_header: bool = True
) -> str:
    """The drift score series and its strongest windows."""
    if signal.empty:
        return "No drift signal was computed."
    lines = _header(f"Drift signal over {len(signal)} windows:", include_header)
    scores = signal["score"].to_numpy(dtype=float)
    lines.append(
        f"score mean {np.nanmean(scores):.3f}, max {np.nanmax(scores):.3f} "
        f"at window {int(np.nanargmax(scores))}"
    )
    order = np.argsort(-scores)[:top_k]
    lines.append("strongest windows:")
    for i in sorted(order):
        lines.append(f"- window {i} starting {signal['window_start'].iloc[i]}: score {scores[i]:.3f}")
    series = ", ".join(f"{s:.3f}" for s in scores)
    lines.append(f"full series: [{series}]")
    return _clip("\n".join(lines), max_len)


def abstract_config(config, max_len: int = MAX_LEN, include_header: bool = True) -> str:
    """Every parameter of a pipeline config, one per line."""
    lines = _header(f"{type(config).__name__} parameters:", include_header)
    for key, value in asdict(config).items():
        lines.append(f"- {key}: {value}")
    return _clip("\n".join(lines), max_len)


def abstract_result(result, max_len: int = MAX_LEN, include_header: bool = True) -> str:
    """One pipeline run, fully described: config, features, PCA, states,
    transitions, distribution, and drift signal — under a shared budget."""
    from ..analysis.states import state_profiles  # local import to avoid a cycle

    parts = _header(f"=== {result.perspective} pipeline run ===", include_header)
    budget = max_len // 7
    parts.append(abstract_config(result.config, max_len=budget))
    parts.append(abstract_features(result.features, max_len=budget))
    parts.append(abstract_pca(result.pca, columns=result.features.columns, max_len=budget))
    profiles = state_profiles(result.features, result.states)
    parts.append(abstract_states(result.states, profiles=profiles, max_len=budget * 2))
    parts.append(abstract_transitions(result.transitions, max_len=budget))
    parts.append(abstract_drift_signal(result.signal, max_len=budget))
    return _clip("\n\n".join(parts), max_len)
