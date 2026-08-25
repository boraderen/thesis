"""All figures: palette, state grid, trajectories, distributions, drift signals.

Every function returns a plotly Figure and renders nothing — the caller decides
where it goes (Streamlit, a notebook, or `save_figure` for LLM vision input).
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from ..analysis.cluster import StateModel, state_distances
from .palette import ACCENT, DISTINCT_COLORS, _is_dark, blend, state_bg, state_fg
from ..data.log import LogStatistics
from ..analysis.reduce import PCAResult

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Log overview
# ---------------------------------------------------------------------------

def plot_activity_frequency(stats: LogStatistics) -> go.Figure:
    """Horizontal bar chart of activity counts, most frequent on top."""
    counts = stats.activity_counts.sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index.tolist(), orientation="h",
        marker_color=ACCENT,
        hovertemplate="%{y}<br>%{x:,} events<extra></extra>",
    ))
    fig.update_layout(
        height=max(220, 28 * len(counts)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Events", yaxis_title=None,
    )
    return fig


# ---------------------------------------------------------------------------
# Reduction and clustering diagnostics
# ---------------------------------------------------------------------------

def plot_pca_variance(pca: PCAResult) -> go.Figure:
    """Bar plot of per-component explained variance with the elbow line."""
    ratios = pca.explained_variance_ratio
    components = np.arange(1, len(ratios) + 1)
    fig = go.Figure(go.Bar(
        x=components, y=ratios, marker_color="#3C3489",
        hovertemplate="PC%{x}<br>ratio=%{y:.3f}<extra></extra>",
    ))
    cumulative = float(np.sum(ratios[:pca.n_components]))
    fig.add_vline(x=pca.n_components + 0.5, line=dict(color="#712B13", width=2, dash="dash"))
    fig.update_layout(
        title=f"PCA reduces {pca.raw_dim}D → {pca.n_components}D ({cumulative:.1%} variance explained)",
        xaxis_title="Component", yaxis_title="Explained variance",
        margin=dict(l=10, r=10, t=40, b=10), height=260,
    )
    return fig


def plot_k_distance(distances: np.ndarray, k: int) -> go.Figure:
    """Sorted k-th-nearest-neighbour distances — the knee is the usual eps candidate."""
    fig = go.Figure(go.Scatter(
        x=np.arange(1, len(distances) + 1), y=distances, mode="lines",
        line=dict(color="#3C3489", width=2),
        hovertemplate="point %{x}<br>distance=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Distance to the {k}-th nearest neighbour (median {np.median(distances):.3f})",
        xaxis_title="Points, sorted by distance", yaxis_title=f"{k}-th NN distance",
        margin=dict(l=10, r=10, t=40, b=10), height=260,
    )
    return fig


# ---------------------------------------------------------------------------
# State grid
# ---------------------------------------------------------------------------

def _cell_annotations(
    model: StateModel, weights: np.ndarray, monochrome: bool
) -> tuple[list[dict], np.ndarray]:
    """The plotly annotations and hover text of the grid cells."""
    grid_h, grid_w = model.grid_h, model.grid_w
    hover_text = np.full((grid_h, grid_w), "", dtype=object)
    annotations: list[dict] = []
    for state in range(model.n_states):
        r, c = divmod(state, grid_w)
        dom = (model.dominant[state] if state < len(model.dominant) else "") or "—"
        hover_text[r, c] = f"{model.labels[state]}<br>dominant: {dom}<br>n={int(model.counts[state])}"
        if monochrome:
            font_color = "#FFFFFF" if weights[state] > 0.5 else "#333333"
        else:
            font_color = "#FFFFFF" if _is_dark(DISTINCT_COLORS[state % len(DISTINCT_COLORS)]) else "#333333"
        annotations.append(dict(
            x=c, y=r, text=f"{model.labels[state]}<br>n={int(model.counts[state])}",
            showarrow=False, font=dict(color=font_color, size=13), xref="x", yref="y",
        ))
    return annotations, hover_text


def plot_state_grid(model: StateModel, monochrome: bool = False, title: str = "") -> go.Figure:
    """The state grid as an annotated heatmap.

    Colored view: one distinct constant color per state. With `monochrome`,
    brightness encodes state frequency instead (near white = empty, black = the
    fullest state).
    """
    n = model.n_states
    weights = np.asarray(model.counts, dtype=float) / max(1.0, float(np.max(model.counts)))
    annotations, hover_text = _cell_annotations(model, weights, monochrome)
    if model.grid_h > 5 or model.grid_w > 5:
        annotations = []  # too many cells for readable on-cell labels — hover has them

    def cell_color(i: int) -> str:
        if monochrome:
            return blend("#F2F2F2", "#000000", weights[i])
        return DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

    fig = go.Figure(go.Heatmap(
        z=np.arange(n).reshape(model.grid_h, model.grid_w),
        text=hover_text, hoverinfo="text", showscale=False,
        colorscale=[[i / max(1, n - 1), cell_color(i)] for i in range(n)],
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        title=title, annotations=annotations,
        xaxis=dict(visible=False, scaleanchor="y", constrain="domain"),
        yaxis=dict(visible=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10), height=320,
    )
    return fig


def plot_state_distances(model: StateModel, metric: str = "euclidean") -> go.Figure:
    """Pairwise distances between state vectors, as an annotated matrix."""
    distances = state_distances(model, metric)
    labels = model.labels
    n = len(labels)
    off_diagonal = distances[~np.eye(n, dtype=bool)] if n > 1 else np.zeros(1)
    fig = go.Figure(go.Heatmap(
        z=distances, x=labels, y=labels, colorscale="Blues", colorbar=dict(title="distance"),
        hovertemplate="%{y} ↔ %{x}<br>distance=%{z:.3f}<extra></extra>",
        xgap=1, ygap=1,
    ))
    if n <= 8:
        fig.update_traces(text=np.round(distances, 2), texttemplate="%{text}")
    fig.update_layout(
        title=(
            f"State-to-state distance — closest pair {off_diagonal.min():.3f}, "
            f"median {np.median(off_diagonal):.3f}, farthest {off_diagonal.max():.3f}"
        ),
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10), height=320,
    )
    return fig


def plot_transition_matrix(matrix: pd.DataFrame) -> go.Figure:
    """State-to-state transition counts as an annotated heatmap."""
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=matrix.columns.tolist(), y=matrix.index.tolist(),
        colorscale="Purples", colorbar=dict(title="count"),
        hovertemplate="%{y} → %{x}<br>%{z} transitions<extra></extra>",
        xgap=1, ygap=1,
    ))
    if len(matrix) <= 10:
        fig.update_traces(text=matrix.values, texttemplate="%{text}")
    fig.update_layout(
        xaxis=dict(title="to", scaleanchor="y", constrain="domain"),
        yaxis=dict(title="from", autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10), height=320,
    )
    return fig


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def _runs(state_ids: np.ndarray) -> list[tuple[int, int, int]]:
    """Collapse consecutive equal state ids into (start_idx, end_idx_inclusive, state)."""
    if len(state_ids) == 0:
        return []
    changes = np.where(np.diff(state_ids) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes - 1, [len(state_ids) - 1]))
    return [(int(s), int(e), int(state_ids[s])) for s, e in zip(starts, ends)]


def _hover_label(sid: int, labels: list[str], dominants: list[str] | None) -> str:
    label = labels[sid] if sid < len(labels) else f"S{sid}"
    dom = (dominants[sid] if dominants and sid < len(dominants) else "") or "—"
    return f"{label}<br>dominant: {dom}"


def _draw_segments(
    fig: go.Figure, x: pd.Series, runs: list[tuple[int, int, int]], end_x,
    labels: list[str], dominants: list[str] | None,
) -> None:
    """One rectangle shape per state run + an invisible hover scatter."""
    hover_x, hover_text = [], []
    for start, end, sid in runs:
        x0 = x.iloc[start]
        x1 = x.iloc[end + 1] if end + 1 < len(x) else end_x
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=1,
            line=dict(width=0), fillcolor=DISTINCT_COLORS[sid % len(DISTINCT_COLORS)], layer="below",
        )
        hover_x.append(x0 + (x1 - x0) / 2)
        hover_text.append(_hover_label(sid, labels, dominants))
    fig.add_trace(go.Scatter(
        x=hover_x, y=[0.5] * len(hover_x), mode="markers",
        marker=dict(opacity=0, size=20), text=hover_text, hoverinfo="x+text",
        showlegend=False,
    ))


def _draw_window_ticks(fig: go.Figure, x: pd.Series, end_x) -> None:
    """Thin solid vertical lines at each window boundary."""
    xs: list = []
    ys: list = []
    for ts in [*x, end_x]:
        xs.extend([ts, ts, None])
        ys.extend([0, 1, None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="rgba(0,0,0,0.10)", width=1),
        hoverinfo="skip", showlegend=False,
    ))


def plot_trajectory(
    timestamps: pd.Series,
    state_ids: np.ndarray,
    model: StateModel,
    title: str = "",
    height: int = 180,
    window_ticks: bool = False,
) -> go.Figure:
    """Left-aligned state segments over time: color changes at the new sample's timestamp.

    Consecutive equal states are merged into one rectangle. With `window_ticks`,
    thin gray lines at every boundary keep individual windows visible inside
    same-state runs.
    """
    fig = go.Figure()
    if len(state_ids) == 0:
        return fig
    x = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    runs = _runs(np.asarray(state_ids))
    last_x = x.iloc[-1]
    span = (last_x - x.iloc[0]) if len(x) > 1 else pd.Timedelta(minutes=1)
    tail = span / max(1, len(x) - 1) if len(x) > 1 else pd.Timedelta(minutes=1)
    end_x = last_x + tail
    _draw_segments(fig, x, runs, end_x, model.labels, model.dominant)
    if window_ticks and len(x) > 1:
        _draw_window_ticks(fig, x, end_x)
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis=dict(showgrid=False, type="date", range=[x.iloc[0], end_x]),
        yaxis=dict(visible=False, range=[0, 1]),
    )
    return fig


def add_transition_markers(fig: go.Figure, timestamps: pd.Series) -> go.Figure:
    """Dashed vertical markers at each transition timestamp."""
    for ts in timestamps:
        fig.add_vline(x=ts, line=dict(color="#3C3489", width=1, dash="dash"), opacity=0.55)
    return fig


# ---------------------------------------------------------------------------
# Distributions and drift signals
# ---------------------------------------------------------------------------

def add_window_boundaries(
    fig: go.Figure, window_starts: pd.Series, y_min: float = 0, y_max: float = 1
) -> go.Figure:
    """Subtle vertical lines at each calendar window start (one trace, fast)."""
    if len(window_starts) == 0:
        return fig
    xs: list = []
    ys: list = []
    for ts in window_starts:
        xs.extend([ts, ts, None])
        ys.extend([y_min, y_max, None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="rgba(0,0,0,0.10)", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    return fig


def plot_state_distribution(distribution: pd.DataFrame, model: StateModel) -> go.Figure:
    """Stacked area chart of per-window state fractions."""
    fig = go.Figure()
    columns = [c for c in distribution.columns if c.startswith("S")]
    for i, col in enumerate(columns):
        label = model.labels[i] if i < len(model.labels) else col
        fig.add_trace(go.Scatter(
            x=distribution["window_start"], y=distribution[col],
            mode="lines", stackgroup="one",
            line=dict(width=0.5, color=state_fg(i)),
            fillcolor=state_bg(i), name=label,
            hovertemplate=f"{label}<br>%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Fraction", range=[0, 1]),
        xaxis=dict(title="Window"),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_drift_signal(signal: pd.DataFrame, title: str = "", height: int = 220) -> go.Figure:
    """Line plot of a per-window drift score, ticked by window index and start."""
    fig = go.Figure()
    axis = dict(title="Window")
    if "score" in signal.columns and not signal.empty:
        index = list(range(len(signal)))
        fig.add_trace(go.Scatter(
            x=signal["window_start"], y=signal["score"], customdata=index,
            mode="lines", line=dict(color="#3C3489", width=2),
            hovertemplate="window %{customdata}<br>%{x}<br>score=%{y:.3f}<extra></extra>",
            showlegend=False,
        ))
        step = max(1, len(signal) // 12)  # one tick per window gets unreadable fast
        axis.update(
            tickmode="array",
            tickvals=list(signal["window_start"].iloc[::step]),
            ticktext=[str(i) for i in index[::step]],
        )
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis=axis, yaxis=dict(title="Score", rangemode="tozero"),
    )
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def save_figure(fig: go.Figure, path: str | Path | None = None, scale: int = 2) -> Path:
    """Write a figure to a PNG, for reports or as LLM vision input."""
    if path is None:
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        path = handle.name
        handle.close()
    # Time axes carry pandas Timestamps, which the image writer cannot serialize.
    # Plotly's own encoder does know them, so round-trip through it first; it drops
    # sub-microsecond precision, which no rendered axis can show anyway.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Discarding nonzero nanoseconds")
        pio.from_json(pio.to_json(fig)).write_image(str(path), scale=scale)
    return Path(path)
