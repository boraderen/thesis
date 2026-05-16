"""Horizontal timeline plot of state labels over an ordered axis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viz.palette import state_bg, state_fg


def state_timeline(
    x: pd.Series,
    state_ids: np.ndarray,
    cell_labels: list[str],
    title: str = "",
    height: int = 180,
    cell_dominant: list[str] | None = None,
) -> go.Figure:
    """Render a state trajectory as a single coloured-strip Heatmap (one trace).

    Plotly's heatmap centers each cell on its event/window timestamp, so a
    cell's visible boundary sits at the midpoint between successive units.
    `xgap=2` adds a thin gap between cells so each observation is visibly
    delimited.
    """
    fig = go.Figure()
    if len(state_ids) == 0:
        return fig
    n_states = max(len(cell_labels), int(state_ids.max()) + 1) if len(state_ids) else 1
    colorscale = [[i / max(1, n_states - 1), state_bg(i)] for i in range(n_states)]

    def _hover(sid: int) -> str:
        label = cell_labels[sid] if sid < len(cell_labels) else f"S{sid}"
        dom = (cell_dominant[sid] if cell_dominant and sid < len(cell_dominant) else "") or "—"
        return f"{label}<br>dominant: {dom}"

    hover = np.array([_hover(int(s)) for s in state_ids])
    fig.add_trace(go.Heatmap(
        x=x, y=["state"], z=[state_ids],
        text=[hover], hoverinfo="x+text",
        colorscale=colorscale, showscale=False,
        zmin=0, zmax=max(1, n_states - 1),
        xgap=2,
    ))
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(visible=False),
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
