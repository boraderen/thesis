"""State clustering: SOM, k-means, and DBSCAN behind one result shape."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from minisom import MiniSom
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

DISTANCES = {
    "euclidean": "Euclidean",
    "manhattan": "Manhattan",
    "chebyshev": "Chebyshev",
    "cosine": "Cosine",
}

# k-means minimises squared Euclidean error around its means, so it cannot take
# an arbitrary metric. Cosine is served by scaling every row to unit length
# first — on unit vectors, Euclidean order matches cosine order (spherical
# k-means) — and Manhattan/Chebyshev have no such equivalent there.
SUPPORTED_DISTANCES = {
    "som": tuple(DISTANCES),
    "dbscan": tuple(DISTANCES),
    "kmeans": ("euclidean", "cosine"),
}

METHODS = {"som": "SOM", "kmeans": "k-means", "dbscan": "DBSCAN"}
SOM_INIT = {"random": "Random samples", "pca": "PCA plane"}


@dataclass(frozen=True)
class StateModel:
    """The states a clustering produced over a feature matrix.

    SOM lays states on a grid_h × grid_w grid; k-means and DBSCAN come back as
    one row of states so every consumer works the same. `centroids` is the
    per-state mean of the clustered vectors (an empty SOM cell falls back to
    its codebook vector); `codebook` is SOM-only.
    """

    method: str
    grid_h: int
    grid_w: int
    state_ids: np.ndarray
    labels: list[str]
    counts: np.ndarray
    dominant: list[str]
    centroids: np.ndarray
    codebook: np.ndarray | None
    params: dict

    @property
    def n_states(self) -> int:
        return self.grid_h * self.grid_w


def _label_states(
    n_states: int, state_ids: np.ndarray, annotations: Iterable[str] | None
) -> tuple[list[str], np.ndarray, list[str]]:
    """Labels S0..S{n-1}, per-state counts, and the most common annotation per state."""
    counts = np.zeros(n_states, dtype=int)
    labels = [f"S{i}" for i in range(n_states)]
    dominants = [""] * n_states
    annotations = list(annotations) if annotations is not None else None
    for state in range(n_states):
        mask = state_ids == state
        counts[state] = int(mask.sum())
        if annotations and counts[state] > 0:
            picked = [annotations[i] for i in np.where(mask)[0]]
            vals, c = np.unique(picked, return_counts=True)
            dominants[state] = str(vals[c.argmax()])
    return labels, counts, dominants


def state_means(matrix: np.ndarray, state_ids: np.ndarray, n_states: int) -> np.ndarray:
    """Per-state mean of the given vectors; an empty state stays all zero."""
    matrix = np.asarray(matrix, dtype=float)
    centroids = np.zeros((n_states, matrix.shape[1] if matrix.ndim > 1 else 1))
    for state in range(n_states):
        mask = state_ids == state
        if mask.any():
            centroids[state] = matrix[mask].mean(axis=0)
    return centroids


def _iteration_budget(n_samples: int, epochs: int = 5, cap: int = 50_000, floor: int = 500) -> int:
    """Single-sample weight updates: ~epochs full passes, capped for big logs."""
    return max(floor, min(cap, epochs * n_samples))


def cluster_som(
    matrix: np.ndarray,
    grid: tuple[int, int] = (3, 3),
    metric: str = "euclidean",
    init: str = "random",
    epochs: int = 5,
    seed: int = 7,
    annotations: tuple[str, ...] | None = None,
) -> StateModel:
    """Train a SOM for ~`epochs` random-order passes and assign each row its BMU.

    `init` selects the weight initialization: "random" sets each cell to a
    randomly drawn data sample; "pca" spans the grid across the first two
    principal components (falls back to random for one-dimensional inputs).
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        raise ValueError("Empty matrix")
    grid_h, grid_w = int(grid[0]), int(grid[1])
    dim = matrix.shape[1]
    sigma = max(1.0, min(grid_h, grid_w) / 2.0)
    som = MiniSom(grid_h, grid_w, dim, sigma=sigma, learning_rate=0.5, random_seed=seed,
                  activation_distance=metric)
    if init == "pca" and dim >= 2:
        som.pca_weights_init(matrix)
    else:
        som.random_weights_init(matrix)
    som.train(matrix, _iteration_budget(matrix.shape[0], epochs=epochs), random_order=True, verbose=False)
    bmus = np.array([som.winner(x) for x in matrix], dtype=int)
    state_ids = bmus[:, 0] * grid_w + bmus[:, 1]
    labels, counts, dominants = _label_states(grid_h * grid_w, state_ids, annotations)
    codebook = som.get_weights().reshape(grid_h * grid_w, dim)
    centroids = state_means(matrix, state_ids, grid_h * grid_w)
    centroids[counts == 0] = codebook[counts == 0]
    return StateModel(
        method="som", grid_h=grid_h, grid_w=grid_w, state_ids=state_ids,
        labels=labels, counts=counts, dominant=dominants,
        centroids=centroids, codebook=codebook,
        params={"grid": (grid_h, grid_w), "metric": metric, "init": init,
                "epochs": epochs, "seed": seed},
    )


def cluster_kmeans(
    matrix: np.ndarray,
    n_clusters: int = 6,
    metric: str = "euclidean",
    seed: int = 7,
    annotations: tuple[str, ...] | None = None,
) -> StateModel:
    """Run k-means and pack the result as one row of states S0..S{k-1}.

    k is clamped to the number of samples; the fixed seed keeps assignments
    reproducible. `metric="cosine"` is served by scaling every row to unit
    length first (spherical k-means).
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        raise ValueError("Empty matrix")
    k = max(1, min(int(n_clusters), matrix.shape[0]))
    points = normalize(matrix) if metric == "cosine" else matrix
    state_ids = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(points).astype(int)
    labels, counts, dominants = _label_states(k, state_ids, annotations)
    return StateModel(
        method="kmeans", grid_h=1, grid_w=k, state_ids=state_ids,
        labels=labels, counts=counts, dominant=dominants,
        centroids=state_means(matrix, state_ids, k), codebook=None,
        params={"n_clusters": int(n_clusters), "metric": metric, "seed": seed},
    )


def cluster_dbscan(
    matrix: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    annotations: tuple[str, ...] | None = None,
) -> StateModel:
    """Run DBSCAN and pack the result as one row of states.

    Clusters become S0..S{k-1}; noise points (if any) become one extra trailing
    state labelled "Noise", so every downstream consumer works unchanged.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        raise ValueError("Empty matrix")
    raw = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric=metric).fit_predict(matrix)
    n_clusters = int(raw.max()) + 1 if (raw >= 0).any() else 0
    has_noise = bool((raw == -1).any())
    n_states = max(1, n_clusters + (1 if has_noise else 0))
    state_ids = np.where(raw >= 0, raw, n_clusters).astype(int)
    labels, counts, dominants = _label_states(n_states, state_ids, annotations)
    if has_noise:
        labels[n_clusters] = "Noise"
    return StateModel(
        method="dbscan", grid_h=1, grid_w=n_states, state_ids=state_ids,
        labels=labels, counts=counts, dominant=dominants,
        centroids=state_means(matrix, state_ids, n_states), codebook=None,
        params={"eps": float(eps), "min_samples": int(min_samples), "metric": metric},
    )


def cluster(
    matrix: np.ndarray,
    method: str = "som",
    annotations: tuple[str, ...] | None = None,
    **params,
) -> StateModel:
    """Dispatch to the chosen clustering method with its own parameters."""
    if method == "som":
        return cluster_som(matrix, annotations=annotations, **params)
    if method == "kmeans":
        return cluster_kmeans(matrix, annotations=annotations, **params)
    if method == "dbscan":
        return cluster_dbscan(matrix, annotations=annotations, **params)
    raise ValueError(f"Unknown clustering method: {method!r} (use one of {tuple(METHODS)})")


def k_distances(matrix: np.ndarray, k: int = 5, metric: str = "euclidean") -> np.ndarray:
    """Each point's distance to its k-th nearest neighbour, sorted ascending.

    The knee of that curve is the usual read for DBSCAN's `eps`: below it a
    point's k-th neighbour is close enough to make it a core point.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        raise ValueError("Empty matrix")
    k = max(1, min(int(k), len(matrix) - 1))
    # +1 neighbour because the query point is its own nearest neighbour.
    finder = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(matrix)
    distances, _ = finder.kneighbors(matrix)
    return np.sort(distances[:, k])


def state_distances(model: StateModel, metric: str = "euclidean") -> np.ndarray:
    """Distance between every pair of state vectors (codebook if the model has one)."""
    vectors = model.codebook if model.codebook is not None else model.centroids
    return pairwise_distances(np.asarray(vectors, dtype=float), metric=metric)


def silhouette(matrix: np.ndarray, model: StateModel, metric: str = "euclidean") -> float:
    """Silhouette score of the state assignment; NaN when it is undefined."""
    matrix = np.asarray(matrix, dtype=float)
    n_labels = len(np.unique(model.state_ids))
    if n_labels < 2 or n_labels >= len(matrix):
        return float("nan")
    return float(silhouette_score(matrix, model.state_ids, metric=metric))
