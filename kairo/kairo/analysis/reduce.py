"""Dimensionality reduction: standardization and PCA with elbow-based selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# What to feed the clustering: the columns as they are, or z-scored first.
SCALING = {
    "none": "No standardization",
    "standardize": "Standardize",
}


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Z-score every column, so no column dominates on its units alone.

    A constant column has no spread to divide by and comes back as zeros.
    """
    return StandardScaler().fit_transform(np.asarray(matrix, dtype=float))


@dataclass(frozen=True)
class PCAResult:
    """A fitted PCA together with its transformed matrix and elbow info."""

    components: np.ndarray
    explained_variance_ratio: np.ndarray
    transformed: np.ndarray
    n_components: int
    raw_dim: int

    def top_loadings(self, columns: list[str], k: int = 5) -> pd.DataFrame:
        """The k strongest-loading input columns of each kept component."""
        rows = []
        for i in range(min(self.n_components, len(self.components))):
            loadings = self.components[i]
            order = np.argsort(-np.abs(loadings))[:k]
            for j in order:
                rows.append({
                    "component": i + 1,
                    "column": columns[j] if j < len(columns) else f"col{j}",
                    "loading": float(loadings[j]),
                })
        return pd.DataFrame(rows)


def _elbow(ratios: np.ndarray, threshold: float = 0.05, k_min: int = 2, k_max: int = 20) -> int:
    """The smallest k where the next component adds < threshold variance."""
    k_max = max(k_min, min(k_max, len(ratios)))
    for k in range(k_min, k_max):
        if ratios[k] < threshold:
            return k
    return k_max


def fit_pca(
    matrix: np.ndarray,
    n_components: int | None = None,
    elbow_threshold: float = 0.05,
    max_components: int = 20,
) -> PCAResult:
    """Fit a PCA; pick k via the elbow rule unless `n_components` overrides it."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        raise ValueError("Empty feature matrix")
    if matrix.shape[0] < 2:
        k = matrix.shape[1]
        return PCAResult(
            components=np.eye(k),
            explained_variance_ratio=np.ones(k) / k if k else np.zeros(0),
            transformed=matrix.copy(),
            n_components=k,
            raw_dim=k,
        )
    max_k = min(matrix.shape[0], matrix.shape[1], max_components)
    pca = PCA(n_components=max_k)
    transformed = pca.fit_transform(matrix)
    if n_components is not None and n_components > 0:
        k = max(1, min(int(n_components), max_k))
    else:
        k = _elbow(pca.explained_variance_ratio_, threshold=elbow_threshold, k_min=2, k_max=max_k)
    return PCAResult(
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        transformed=transformed[:, :k],
        n_components=k,
        raw_dim=matrix.shape[1],
    )


def reduce(
    matrix: np.ndarray,
    skip_pca: bool = False,
    n_components: int | None = None,
    scaling: str = "none",
) -> tuple[np.ndarray, PCAResult | None]:
    """The page-order reduction: optional PCA, then optional standardization.

    Standardization, when asked for, applies to whatever goes into the
    clustering — the raw columns, or the PCA output.
    """
    pca = None if skip_pca else fit_pca(matrix, n_components=n_components)
    reduced = np.asarray(matrix, dtype=float) if pca is None else pca.transformed
    if scaling == "standardize":
        reduced = standardize(reduced)
    return reduced, pca
