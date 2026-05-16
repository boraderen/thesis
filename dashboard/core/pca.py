"""PCA with elbow-based component selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PCAResult:
    """A fitted PCA together with its transformed matrix and elbow info."""

    components: np.ndarray
    explained_variance_ratio: np.ndarray
    transformed: np.ndarray
    chosen_k: int
    raw_dim: int


def _elbow(ratios: np.ndarray, threshold: float = 0.05, k_min: int = 2, k_max: int = 20) -> int:
    """Pick the smallest k where the next component adds < threshold variance."""
    k_max = max(k_min, min(k_max, len(ratios)))
    for k in range(k_min, k_max):
        if ratios[k] < threshold:
            return k
    return k_max


@st.cache_data(show_spinner=False)
def fit_pca(matrix: np.ndarray, threshold: float = 0.05) -> PCAResult:
    """Fit a PCA on `matrix` and select k via the elbow rule."""
    if matrix.size == 0:
        raise ValueError("Empty feature matrix")
    if matrix.shape[0] < 2:
        k = matrix.shape[1]
        return PCAResult(
            components=np.eye(k),
            explained_variance_ratio=np.ones(k) / k if k else np.zeros(0),
            transformed=matrix.copy(),
            chosen_k=k,
            raw_dim=k,
        )
    max_k = min(matrix.shape[0], matrix.shape[1], 20)
    pca = PCA(n_components=max_k)
    transformed = pca.fit_transform(matrix)
    k = _elbow(pca.explained_variance_ratio_, threshold=threshold, k_min=2, k_max=max_k)
    return PCAResult(
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        transformed=transformed[:, :k],
        chosen_k=k,
        raw_dim=matrix.shape[1],
    )
