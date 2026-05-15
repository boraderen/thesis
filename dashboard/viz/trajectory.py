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
) -> go.Figure:
    """Render a sequence of state labels as a coloured strip along x."""
    fig = go.Figure()
    if len(state_ids) == 0:
        return fig
    for i, sid in enumerate(state_ids):
        bg = state_bg(int(sid))
        fg = state_fg(int(sid))
        x0 = x.iloc[i] if i < len(x) else i
        x1 = x.iloc[i + 1] if i + 1 < len(x) else x0
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=1,
            line=dict(width=0), fillcolor=bg, layer="below",
        )
        fig.add_annotation(
            x=x0, y=0.5, text=str(cell_labels[int(sid)]),
            showarrow=False, font=dict(color=fg, size=10), xanchor="left",
        )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(visible=False, range=[0, 1]),
    )
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
