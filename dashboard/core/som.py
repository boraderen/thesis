"""Self-organising map: train, assign BMU, label cells by dominant feature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import streamlit as st
from minisom import MiniSom


@dataclass(frozen=True)
class SOMResult:
    """Outcome of training a SOM on a feature matrix."""

    grid_h: int
    grid_w: int
    bmus: np.ndarray  # shape (n_samples, 2) row, col
    state_ids: np.ndarray  # 1D index = row * grid_w + col
    cell_labels: list[str]  # length grid_h * grid_w — log-agnostic enumeration "S{id}"
    cell_counts: np.ndarray  # 1D length grid_h * grid_w
    cell_dominant: list[str]  # length grid_h * grid_w — descriptive overlay for tooltips


def _label_cells(
    grid_h: int,
    grid_w: int,
    bmus: np.ndarray,
    annotations: Iterable[str] | None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Return labels, counts, and the most-common annotation per cell (separately)."""
    counts = np.zeros(grid_h * grid_w, dtype=int)
    labels = [f"S{i}" for i in range(grid_h * grid_w)]
    dominants = [""] * (grid_h * grid_w)
    annotations = list(annotations) if annotations is not None else None
    for cell_id in range(grid_h * grid_w):
        mask = (bmus[:, 0] * grid_w + bmus[:, 1]) == cell_id
        counts[cell_id] = int(mask.sum())
        if annotations and counts[cell_id] > 0:
            picked = [annotations[i] for i in np.where(mask)[0]]
            vals, c = np.unique(picked, return_counts=True)
            dominants[cell_id] = str(vals[c.argmax()])
    return labels, counts, dominants


@st.cache_data(show_spinner=False)
def train_som(
    matrix: np.ndarray,
    grid_h: int = 3,
    grid_w: int = 3,
    iterations: int = 500,
    seed: int = 7,
    annotations: tuple[str, ...] | None = None,
) -> SOMResult:
    """Train a SOM and return BMU assignments + cells labelled as 'S{id} · {dominant}'."""
    if matrix.size == 0:
        raise ValueError("Empty matrix")
    dim = matrix.shape[1]
    sigma = max(1.0, min(grid_h, grid_w) / 2.0)
    som = MiniSom(grid_h, grid_w, dim, sigma=sigma, learning_rate=0.5, random_seed=seed)
    som.random_weights_init(matrix)
    som.train(matrix, iterations, verbose=False)
    bmus = np.array([som.winner(x) for x in matrix], dtype=int)
    labels, counts, dominants = _label_cells(grid_h, grid_w, bmus, annotations)
    state_ids = bmus[:, 0] * grid_w + bmus[:, 1]
    return SOMResult(
        grid_h=grid_h,
        grid_w=grid_w,
        bmus=bmus,
        state_ids=state_ids,
        cell_labels=labels,
        cell_counts=counts,
        cell_dominant=dominants,
    )
