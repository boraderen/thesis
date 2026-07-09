"""DBSCAN clustering as a SOM alternative: same result shape, one row of cells."""
from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.cluster import DBSCAN

from core.som import SOMResult, _label_cells


@st.cache_data(show_spinner=False)
def cluster_dbscan(
    matrix: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    annotations: tuple[str, ...] | None = None,
) -> SOMResult:
    """Run DBSCAN on the reduced matrix and pack the result as a 1×k SOMResult.

    Clusters become states S0..S{k-1}; noise points (if any) become one extra
    trailing state labelled "Noise", so every downstream consumer of SOMResult
    (grid plot, timelines, transitions, drift signals) works unchanged.
    """
    if matrix.size == 0:
        raise ValueError("Empty matrix")
    raw = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(matrix)
    n_clusters = int(raw.max()) + 1 if (raw >= 0).any() else 0
    has_noise = bool((raw == -1).any())
    n_states = max(1, n_clusters + (1 if has_noise else 0))
    state_ids = np.where(raw >= 0, raw, n_clusters).astype(int)
    bmus = np.stack([np.zeros_like(state_ids), state_ids], axis=1)
    labels, counts, dominants = _label_cells(1, n_states, bmus, annotations)
    if has_noise:
        labels[n_clusters] = "Noise"
    return SOMResult(
        grid_h=1,
        grid_w=n_states,
        bmus=bmus,
        state_ids=state_ids,
        cell_labels=labels,
        cell_counts=counts,
        cell_dominant=dominants,
    )
