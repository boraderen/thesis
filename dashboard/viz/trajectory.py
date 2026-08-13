"""Horizontal timeline plot of state labels over an ordered axis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viz.palette import DISTINCT_COLORS


def _runs(state_ids: np.ndarray) -> list[tuple[int, int, int]]:
    """Collapse consecutive equal state ids into (start_idx, end_idx_inclusive, state)."""
    if len(state_ids) == 0:
        return []
    changes = np.where(np.diff(state_ids) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes - 1, [len(state_ids) - 1]))
    return [(int(s), int(e), int(state_ids[s])) for s, e in zip(starts, ends)]


def _hover_label(sid: int, labels: list[str], dominants: list[str] | None) -> str:
    """Single hover string for a state id, with dominant info if available."""
    label = labels[sid] if sid < len(labels) else f"S{sid}"
    dom = (dominants[sid] if dominants and sid < len(dominants) else "") or "—"
    return f"{label}<br>dominant: {dom}"


def _draw_segments(
    fig: go.Figure, x: pd.Series, runs: list[tuple[int, int, int]], end_x,
    cell_labels: list[str], cell_dominant: list[str] | None,
) -> None:
    """Add one rectangle shape per state run + an invisible hover scatter."""
    hover_x, hover_text = [], []
    for start, end, sid in runs:
        x0 = x.iloc[start]
        x1 = x.iloc[end + 1] if end + 1 < len(x) else end_x
        fill = DISTINCT_COLORS[sid % len(DISTINCT_COLORS)]
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=1,
            line=dict(width=0), fillcolor=fill, layer="below",
        )
        hover_x.append(x0 + (x1 - x0) / 2)
        hover_text.append(_hover_label(sid, cell_labels, cell_dominant))
    fig.add_trace(go.Scatter(
        x=hover_x, y=[0.5] * len(hover_x), mode="markers",
        marker=dict(opacity=0, size=20), text=hover_text, hoverinfo="x+text",
        showlegend=False,
    ))


def _draw_window_ticks(fig: go.Figure, x: pd.Series, end_x) -> None:
    """Overlay thin solid vertical lines at each window boundary."""
    xs: list = []
    ys: list = []
    for ts in [*x, end_x]:
        xs.extend([ts, ts, None])
        ys.extend([0, 1, None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="rgba(0,0,0,0.10)", width=1),
        hoverinfo="skip", showlegend=False,
    ))


def state_timeline(
    x: pd.Series,
    state_ids: np.ndarray,
    cell_labels: list[str],
    title: str = "",
    height: int = 180,
    cell_dominant: list[str] | None = None,
    window_ticks: bool = False,
) -> go.Figure:
    """Left-aligned state segments: color changes at the new event's timestamp.

    Consecutive equal states are merged into one rectangle. If `window_ticks`
    is true, thin solid gray lines at every window boundary are overlaid so
    individual windows remain visible inside same-state runs.
    """
    fig = go.Figure()
    if len(state_ids) == 0:
        return fig
    x = pd.to_datetime(pd.Series(x).reset_index(drop=True))
    runs = _runs(np.asarray(state_ids))
    last_x = x.iloc[-1]
    span = (last_x - x.iloc[0]) if len(x) > 1 else pd.Timedelta(minutes=1)
    tail = span / max(1, len(x) - 1) if len(x) > 1 else pd.Timedelta(minutes=1)
    end_x = last_x + tail
    _draw_segments(fig, x, runs, end_x, cell_labels, cell_dominant)
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
    """Overlay dashed vertical markers at each transition timestamp."""
    for ts in timestamps:
        fig.add_vline(x=ts, line=dict(color="#3C3489", width=1, dash="dash"), opacity=0.55)
    return fig


def pca_variance_plot(ratios: np.ndarray, chosen_k: int, raw_dim: int) -> go.Figure:
    """Bar plot of per-component explained variance with the elbow line."""
    components = np.arange(1, len(ratios) + 1)
    fig = go.Figure(
        data=[
            go.Bar(
                x=components, y=ratios, marker_color="#3C3489",
                hovertemplate="PC%{x}<br>ratio=%{y:.3f}<extra></extra>",
            )
        ]
    )
    cumulative = float(np.sum(ratios[:chosen_k]))
    fig.add_vline(x=chosen_k + 0.5, line=dict(color="#712B13", width=2, dash="dash"))
    fig.update_layout(
        title=f"PCA reduces {raw_dim}D → {chosen_k}D ({cumulative:.1%} variance explained)",
        xaxis_title="Component",
        yaxis_title="Explained variance",
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
    )
    return fig



def k_distance_plot(distances: np.ndarray, k: int) -> go.Figure:
    """Sorted k-th-nearest-neighbour distances — the knee is the usual eps candidate."""
    fig = go.Figure(data=[go.Scatter(
        x=np.arange(1, len(distances) + 1), y=distances, mode="lines",
        line=dict(color="#3C3489", width=2),
        hovertemplate="point %{x}<br>distance=%{y:.3f}<extra></extra>",
    )])
    fig.update_layout(
        title=f"Distance to the {k}-th nearest neighbour (median {np.median(distances):.3f})",
        xaxis_title="Points, sorted by distance",
        yaxis_title=f"{k}-th NN distance",
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
    )
    return fig
