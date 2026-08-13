"""Distance measures offered for clustering, and which method can honour them."""
from __future__ import annotations

DISTANCE_LABELS = {
    "euclidean": "Euclidean",
    "manhattan": "Manhattan",
    "chebyshev": "Chebyshev",
    "cosine": "Cosine",
}

# k-means minimises squared Euclidean error around its means, so it cannot take
# an arbitrary metric. Cosine is served by scaling every row to unit length
# first — on unit vectors, Euclidean order matches cosine order (spherical
# k-means) — and Manhattan/Chebyshev have no such equivalent, so they are not
# offered there.
SUPPORTED_DISTANCES = {
    "som": tuple(DISTANCE_LABELS),
    "dbscan": tuple(DISTANCE_LABELS),
    "kmeans": ("euclidean", "cosine"),
}
