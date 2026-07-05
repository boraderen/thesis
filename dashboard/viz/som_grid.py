"""SOM grid rendered as an annotated heatmap: state hue per cell, brightness by density."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from viz.palette import state_bg, state_fg


def _blend(light: str, dark: str, w: float) -> str:
    """Linear blend between two hex colors: w=0 → light, w=1 → dark."""
    a = [int(light[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(dark[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * w):02X}" for x, y in zip(a, b))


def _cell_annotations(
    grid_h: int,
    grid_w: int,
    counts: np.ndarray,
    labels: list[str],
    dominants: list[str] | None,
    weights: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Build the plotly annotations, on-cell label text, and hover text."""
    cell_text = np.full((grid_h, grid_w), "", dtype=object)
    hover_text = np.full((grid_h, grid_w), "", dtype=object)
    annotations: list[dict] = []
    for cell_id in range(grid_h * grid_w):
        r, c = divmod(cell_id, grid_w)
        cell_text[r, c] = f"{labels[cell_id]}<br>n={int(counts[cell_id])}"
        dom = (dominants[cell_id] if dominants and cell_id < len(dominants) else "") or "—"
        hover_text[r, c] = f"{labels[cell_id]}<br>dominant: {dom}<br>n={int(counts[cell_id])}"
        annotations.append(dict(
            x=c, y=r, text=cell_text[r, c], showarrow=False,
            font=dict(color="#FFFFFF" if weights[cell_id] > 0.5 else state_fg(cell_id), size=13),
            xref="x", yref="y",
        ))
    return annotations, cell_text, hover_text


def som_heatmap(
    grid_h: int, grid_w: int,
    counts: np.ndarray, labels: list[str], title: str = "",
    dominants: list[str] | None = None,
) -> go.Figure:
    """Return a plotly heatmap of the SOM grid.

    Each cell keeps its state hue, and its brightness encodes density: the
    color runs from the light state background (empty cell) to the dark state
    foreground (the fullest cell of this grid).
    """
    n_cells = grid_h * grid_w
    weights = np.asarray(counts, dtype=float) / max(1.0, float(np.max(counts)))
    annotations, _, hover_text = _cell_annotations(grid_h, grid_w, counts, labels, dominants, weights)
    fig = go.Figure(data=go.Heatmap(
        z=np.arange(n_cells).reshape(grid_h, grid_w),
        text=hover_text, hoverinfo="text", showscale=False,
        colorscale=[
            [i / max(1, n_cells - 1), _blend(state_bg(i), state_fg(i), weights[i])]
            for i in range(n_cells)
        ],
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        title=title, annotations=annotations,
        xaxis=dict(visible=False, scaleanchor="y", constrain="domain"),
        yaxis=dict(visible=False, autorange="reversed"),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        height=320,
    )
    return fig
