"""SOM grid rendered as an annotated heatmap: one distinct color per cell, or a
monochrome view where brightness encodes cell frequency."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from viz.palette import DISTINCT_COLORS, blend


def _is_dark(color: str) -> bool:
    """Perceived-luminance check to pick a readable label color."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return (0.299 * r + 0.587 * g + 0.114 * b) < 140


def _cell_annotations(
    grid_h: int,
    grid_w: int,
    counts: np.ndarray,
    labels: list[str],
    dominants: list[str] | None,
    weights: np.ndarray,
    monochrome: bool = False,
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
        if monochrome:
            font_color = "#FFFFFF" if weights[cell_id] > 0.5 else "#333333"
        else:
            cell = DISTINCT_COLORS[cell_id % len(DISTINCT_COLORS)]
            font_color = "#FFFFFF" if _is_dark(cell) else "#333333"
        annotations.append(dict(
            x=c, y=r, text=cell_text[r, c], showarrow=False,
            font=dict(color=font_color, size=13),
            xref="x", yref="y",
        ))
    return annotations, cell_text, hover_text


def som_heatmap(
    grid_h: int, grid_w: int,
    counts: np.ndarray, labels: list[str], title: str = "",
    dominants: list[str] | None = None,
    monochrome: bool = False,
) -> go.Figure:
    """Return a plotly heatmap of the SOM grid.

    Colored view: one distinct constant color per cell from `DISTINCT_COLORS`.
    With `monochrome` every cell uses the same black hue and brightness encodes
    the cell frequency instead (near white = empty, black = the fullest cell of
    this grid).
    """
    n_cells = grid_h * grid_w
    weights = np.asarray(counts, dtype=float) / max(1.0, float(np.max(counts)))
    annotations, _, hover_text = _cell_annotations(
        grid_h, grid_w, counts, labels, dominants, weights, monochrome
    )
    if grid_h > 5 or grid_w > 5:
        # Too many cells for readable on-cell labels — hover still has them.
        annotations = []

    def cell_color(i: int) -> str:
        if monochrome:
            return blend("#F2F2F2", "#000000", weights[i])
        return DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

    fig = go.Figure(data=go.Heatmap(
        z=np.arange(n_cells).reshape(grid_h, grid_w),
        text=hover_text, hoverinfo="text", showscale=False,
        colorscale=[[i / max(1, n_cells - 1), cell_color(i)] for i in range(n_cells)],
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


def cell_distance_heatmap(distances: np.ndarray, labels: list[str]) -> go.Figure:
    """Pairwise Euclidean distances between SOM cell vectors, as an annotated matrix."""
    n = len(labels)
    off_diagonal = distances[~np.eye(n, dtype=bool)] if n > 1 else np.zeros(1)
    fig = go.Figure(data=go.Heatmap(
        z=distances, x=labels, y=labels, colorscale="Blues", colorbar=dict(title="distance"),
        hovertemplate="%{y} ↔ %{x}<br>distance=%{z:.3f}<extra></extra>",
        xgap=1, ygap=1,
    ))
    if n <= 8:
        fig.update_traces(text=np.round(distances, 2), texttemplate="%{text}")
    fig.update_layout(
        title=(
            f"Cell-to-cell distance — closest pair {off_diagonal.min():.3f}, "
            f"median {np.median(off_diagonal):.3f}, farthest {off_diagonal.max():.3f}"
        ),
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    return fig
