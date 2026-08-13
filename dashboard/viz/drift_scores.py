"""Line plots for the four per-perspective drift attribution scores."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PERSP_COLOR = {
    "cf_score": "#3C3489",
    "resource_score": "#633806",
    "inter_score": "#085041",
    "data_score": "#712B13",
}


def score_line(df: pd.DataFrame, column: str, title: str = "", height: int = 180) -> go.Figure:
    """Line plot of a single per-window drift score, ticked by window index and start."""
    fig = go.Figure()
    axis = dict(title="Window")
    if column in df.columns and not df.empty:
        index = list(range(len(df)))
        fig.add_trace(go.Scatter(
            x=df["window_start"], y=df[column], customdata=index,
            mode="lines", line=dict(color=PERSP_COLOR.get(column, "#333"), width=2),
            hovertemplate="window %{customdata}<br>%{x}<br>score=%{y:.3f}<extra></extra>",
            showlegend=False,
        ))
        # One tick per window gets unreadable past a dozen or so windows.
        step = max(1, len(df) // 12)
        axis.update(
            tickmode="array",
            tickvals=list(df["window_start"].iloc[::step]),
            ticktext=[str(i) for i in index[::step]],
        )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        xaxis=axis,
        yaxis=dict(title="Score", rangemode="tozero"),
    )
    return fig
