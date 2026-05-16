"""Stacked-area and line charts for the joint drift signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viz.palette import state_bg, state_fg


def add_window_boundaries(
    fig: go.Figure, window_starts: pd.Series, y_min: float = 0, y_max: float = 1
) -> go.Figure:
    """Overlay subtle vertical lines at each calendar window start (one trace, fast)."""
    if len(window_starts) == 0:
        return fig
    xs: list = []
    ys: list = []
    for ts in window_starts:
        xs.extend([ts, ts, None])
        ys.extend([y_min, y_max, None])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="rgba(0,0,0,0.10)", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    return fig


def stacked_area_intra(df: pd.DataFrame, columns: list[str], labels: list[str]) -> go.Figure:
    """Stacked area chart of per-window intra-case state fractions."""
    fig = go.Figure()
    for i, col in enumerate(columns):
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["window_start"], y=df[col],
                mode="lines", stackgroup="one",
                line=dict(width=0.5, color=state_fg(i)),
                fillcolor=state_bg(i),
                name=labels[i] if i < len(labels) else col,
                hovertemplate=f"{labels[i] if i < len(labels) else col}<br>%{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Fraction", range=[0, 1]),
        xaxis=dict(title="Window"),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def state_index_line(df: pd.DataFrame, col: str, labels: list[str], title: str) -> go.Figure:
    """Line plot of a single state index over windows, coloured per state."""
    fig = go.Figure()
    if col in df.columns:
        values = df[col].astype(float)
        fig.add_trace(
            go.Scatter(
                x=df["window_start"], y=values,
                mode="lines+markers",
                line=dict(color="#3C3489", width=1.2),
                marker=dict(
                    size=8,
                    color=[state_bg(int(v)) if not np.isnan(v) else "#FFFFFF" for v in values],
                    line=dict(width=1, color="#333333"),
                ),
                hovertext=[labels[int(v)] if not np.isnan(v) and int(v) < len(labels) else "" for v in values],
                hovertemplate="%{x}<br>%{hovertext}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(title="State idx"),
        xaxis=dict(title="Window"),
    )
    return fig
