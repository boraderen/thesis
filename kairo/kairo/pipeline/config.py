"""Pipeline configurations: one frozen dataclass per perspective.

Each config carries every knob of its pipeline at the same granularity as the
dashboard sidebar, so a run is fully described — and reproducible — by one
serializable object.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..data.schema import INTRA_FEATURES


@dataclass(frozen=True)
class IntraConfig:
    """Every knob of the intra-case pipeline, in dashboard-sidebar granularity."""

    features: tuple[str, ...] = tuple(INTRA_FEATURES)
    history: int = 3
    skip_pca: bool = False
    pca_components: int | None = None      # None = elbow rule
    scaling: str = "none"                  # none | standardize
    clustering: str = "som"                # som | kmeans | dbscan
    metric: str = "euclidean"
    grid: tuple[int, int] = (3, 3)         # som
    som_init: str = "random"               # random | pca
    n_clusters: int = 6                    # kmeans
    eps: float = 0.5                       # dbscan
    min_samples: int = 5                   # dbscan
    window_minutes: int = 60               # state-distribution window W
    divergence: str = "js"                 # kl | js | tv | hellinger
    reference: str = "previous"            # previous | recent | baseline
    lookback: int = 5
    seed: int = 7


@dataclass(frozen=True)
class ResourceConfig:
    """Every knob of the resource pipeline; empty resources/activities = all."""

    features: tuple[str, ...] = ("events", "active", "duration", "wait", "activity_events", "ho")
    resources: tuple[str, ...] = ()
    activities: tuple[str, ...] = ()
    window_minutes: int = 60
    skip_pca: bool = False
    pca_components: int | None = None
    scaling: str = "none"
    clustering: str = "som"
    metric: str = "euclidean"
    grid: tuple[int, int] = (2, 3)
    som_init: str = "random"
    n_clusters: int = 6
    eps: float = 0.5
    min_samples: int = 5
    signal_metric: str = "euclidean"       # window-to-window vector distance
    signal_reference: str = "previous"     # previous | recent | baseline
    signal_lookback: int = 5               # l, for the "recent" reference
    seed: int = 7


@dataclass(frozen=True)
class InterConfig:
    """Every knob of the inter-case pipeline; features may include attr_* keys."""

    features: tuple[str, ...] = ()         # empty = all system + attribute features
    window_minutes: int = 60
    stall_minutes: int = 60
    numeric_attrs: tuple[str, ...] = ()
    categorical_attrs: tuple[str, ...] = ()
    skip_pca: bool = False
    pca_components: int | None = None
    scaling: str = "none"
    clustering: str = "som"
    metric: str = "euclidean"
    grid: tuple[int, int] = (2, 2)
    som_init: str = "random"
    n_clusters: int = 6
    eps: float = 0.5
    min_samples: int = 5
    signal_metric: str = "euclidean"
    signal_reference: str = "previous"
    signal_lookback: int = 5
    seed: int = 7


CONFIGS = {"intra_case": IntraConfig, "resource": ResourceConfig, "inter_case": InterConfig}
