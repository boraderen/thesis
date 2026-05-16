"""Horizontal timeline plot of state labels over an ordered axis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viz.palette import state_bg, state_fg


def _runs(state_ids: np.ndarray) -> list[tuple[int, int, int]]:
    """Collapse consecutive equal state ids into (start_idx, end_idx_inclusive, state)."""
    if len(state_ids) == 0:
        return []
    changes = np.where(np.diff(state_ids) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes - 1, [len(state_ids) - 1]))
    return [(int(s), int(e), int(state_ids[s])) for s, e in zip(starts, ends)]


def state_timeline(
    x: pd.Series,
    state_ids: np.ndarray,
    cell_labels: list[str],
    title: str = "",
    height: int = 180,
) -> go.Figure:
    """Render a state trajectory as left-aligned segments.

    Each segment runs from its starting event to the next state change (or to
    the final timestamp for the last segment). Consecutive events in the same
    state are merged into one rectangle, so the plot stays cheap even on
    hundred-thousand-window logs.
    """
    fig = go.Figure()
    if len(state_ids) == 0:
        return fig
    x = pd.to_datetime(pd.Series(x).reset_index(drop=True))
    runs = _runs(np.asarray(state_ids))
    last_x = x.iloc[-1]
    span = (last_x - x.iloc[0]) if len(x) > 1 else pd.Timedelta(minutes=1)
    tail = span / max(1, len(x) - 1) if len(x) > 1 else pd.Timedelta(minutes=1)
    for start, end, sid in runs:
        x0 = x.iloc[start]
        x1 = x.iloc[end + 1] if end + 1 < len(x) else last_x + tail
        label = cell_labels[sid] if sid < len(cell_labels) else f"S{sid}"
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=1,
            line=dict(width=0), fillcolor=state_bg(sid), layer="below",
        )
        fig.add_annotation(
            x=x0, y=0.5, text=f"<b>{label}</b>",
            xanchor="left", yanchor="middle",
            showarrow=False, font=dict(color=state_fg(sid), size=11),
        )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis=dict(showgrid=False, range=[x.iloc[0], last_x + tail]),
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
    fig.add_vline(x=chosen_k + 0.5, line=dict(color="#712B13", width=2, dash="dash"))
    fig.update_layout(
        title=f"PCA reduces {raw_dim}D → {chosen_k}D (elbow @ k={chosen_k})",
        xaxis_title="Component",
        yaxis_title="Explained variance",
        margin=dict(l=10, r=10, t=40, b=10),
        height=260,
    )
    return fig
