"""k-means clustering as a SOM alternative: same result shape, one row of cells."""
from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.cluster import KMeans

from core.som import SOMResult, _label_cells


@st.cache_data(show_spinner=False)
def cluster_kmeans(
    matrix: np.ndarray,
    n_clusters: int = 6,
    seed: int = 7,
    annotations: tuple[str, ...] | None = None,
) -> SOMResult:
    """Run k-means and pack the result as a 1×k SOMResult (states S0..S{k-1}).

    k is clamped to the number of samples; the fixed seed keeps assignments
    reproducible across runs.
    """
    if matrix.size == 0:
        raise ValueError("Empty matrix")
    k = max(1, min(int(n_clusters), matrix.shape[0]))
    ids = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(matrix).astype(int)
    bmus = np.stack([np.zeros_like(ids), ids], axis=1)
    labels, counts, dominants = _label_cells(1, k, bmus, annotations)
    return SOMResult(
        grid_h=1,
        grid_w=k,
        bmus=bmus,
        state_ids=ids,
        cell_labels=labels,
        cell_counts=counts,
        cell_dominant=dominants,
    )
