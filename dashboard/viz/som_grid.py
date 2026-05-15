"""SOM grid rendered as an annotated heatmap of cell counts + labels."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from viz.palette import state_bg, state_fg


def _cell_annotations(grid_h: int, grid_w: int, counts: np.ndarray, labels: list[str]) -> tuple[list[dict], np.ndarray]:
    """Build the plotly annotation list and the text overlay matrix for the heatmap."""
    text = np.full((grid_h, grid_w), "", dtype=object)
    annotations: list[dict] = []
    for cell_id in range(grid_h * grid_w):
        r, c = divmod(cell_id, grid_w)
        text[r, c] = f"{labels[cell_id]}<br>n={int(counts[cell_id])}"
        annotations.append(dict(
            x=c, y=r, text=text[r, c], showarrow=False,
            font=dict(color=state_fg(cell_id), size=13),
            xref="x", yref="y",
        ))
    return annotations, text


def som_heatmap(
    grid_h: int, grid_w: int,
    counts: np.ndarray, labels: list[str], title: str = "",
) -> go.Figure:
    """Return a plotly heatmap with each cell shaded by its state color."""
    annotations, text = _cell_annotations(grid_h, grid_w, counts, labels)
    n_cells = grid_h * grid_w
    fig = go.Figure(data=go.Heatmap(
        z=np.arange(n_cells).reshape(grid_h, grid_w),
        text=text, hoverinfo="text", showscale=False,
        colorscale=[[i / max(1, n_cells - 1), state_bg(i)] for i in range(n_cells)],
    ))
    fig.update_layout(
        title=title, annotations=annotations,
        xaxis=dict(visible=False, scaleanchor="y", constrain="domain"),
        yaxis=dict(visible=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=320,
    )
    return fig
