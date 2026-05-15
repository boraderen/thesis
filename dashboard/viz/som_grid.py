"""SOM grid rendered as an annotated heatmap of cell counts + labels."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from viz.palette import state_bg, state_fg


def som_heatmap(
    grid_h: int,
    grid_w: int,
    counts: np.ndarray,
    labels: list[str],
    title: str = "",
) -> go.Figure:
    """Return a plotly heatmap with each cell shaded by its state color."""
    z_colors = np.full((grid_h, grid_w), "", dtype=object)
    text = np.full((grid_h, grid_w), "", dtype=object)
    annotations: list[dict] = []
    for cell_id in range(grid_h * grid_w):
        r, c = divmod(cell_id, grid_w)
        bg = state_bg(cell_id)
        fg = state_fg(cell_id)
        z_colors[r, c] = bg
        text[r, c] = f"{labels[cell_id]}<br>n={int(counts[cell_id])}"
        annotations.append(
            dict(
                x=c, y=r, text=text[r, c], showarrow=False,
                font=dict(color=fg, size=13), xref="x", yref="y",
            )
        )
    fig = go.Figure(
        data=go.Heatmap(
            z=np.arange(grid_h * grid_w).reshape(grid_h, grid_w),
            text=text,
            hoverinfo="text",
            showscale=False,
            colorscale=[[i / max(1, grid_h * grid_w - 1), state_bg(i)] for i in range(grid_h * grid_w)],
        )
    )
    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(visible=False, scaleanchor="y", constrain="domain"),
        yaxis=dict(visible=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=320,
    )
    return fig
