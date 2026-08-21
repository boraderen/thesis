"""End-to-end pipelines: a config dataclass per perspective, one result shape.

Each runner is a thin composition of the step functions, so the dashboard (or
an agent) can equally run the whole pipeline in one call or drive the steps one
by one with the same parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from .cluster import StateModel, cluster, state_means
from .drift import drift_signal, window_vector_shift
from .features import FeatureSet, build_features
from .features.inter import describe_states
from .reduce import PCAResult, reduce
from .schema import ACTIVITY, CASE, INTRA_FEATURES, TIMESTAMP
from .states import case_transitions, find_transitions, state_distribution, trajectories


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
    seed: int = 7


CONFIGS = {"intra_case": IntraConfig, "resource": ResourceConfig, "inter_case": InterConfig}


@dataclass(frozen=True)
class StateResult:
    """Everything one pipeline run produced, ready to plot, abstract, or compare."""

    perspective: str
    config: IntraConfig | ResourceConfig | InterConfig
    features: FeatureSet
    pca: PCAResult | None
    reduced: np.ndarray
    states: StateModel
    trajectories: pd.DataFrame
    transitions: pd.DataFrame
    distribution: pd.DataFrame
    signal: pd.DataFrame

    def to_dict(self) -> dict:
        """A light JSON-ready summary of the run (no matrices)."""
        return {
            "perspective": self.perspective,
            "config": asdict(self.config),
            "rows": int(len(self.features.matrix)),
            "feature_columns": len(self.features.columns),
            "pca_components": None if self.pca is None else self.pca.n_components,
            "n_states": self.states.n_states,
            "method": self.states.method,
            "transitions": int(len(self.transitions)),
            "windows": int(len(self.signal)),
        }


def _cluster_params(config, method: str) -> dict:
    if method == "som":
        return {"grid": tuple(config.grid), "metric": config.metric,
                "init": config.som_init, "seed": config.seed}
    if method == "kmeans":
        return {"n_clusters": config.n_clusters, "metric": config.metric, "seed": config.seed}
    return {"eps": config.eps, "min_samples": config.min_samples, "metric": config.metric}


def run_intra_case(log: pd.DataFrame, config: IntraConfig = IntraConfig()) -> StateResult:
    """features → (PCA) → (scaling) → clustering → trajectories, transitions,
    per-window state distribution, and the divergence drift signal."""
    fs = build_features(log, "intra_case", features=config.features, history=config.history)
    matrix = fs.values()
    reduced, pca = reduce(matrix, config.skip_pca, config.pca_components, config.scaling)
    annotations = tuple(fs.index[ACTIVITY].astype(str))
    model = cluster(reduced, method=config.clustering, annotations=annotations,
                    **_cluster_params(config, config.clustering))
    traj = trajectories(fs, model)
    transitions = case_transitions(fs, model)
    dist = state_distribution(fs.index[TIMESTAMP], model.state_ids, model.n_states,
                              config.window_minutes)
    signal = drift_signal(dist, config.divergence, config.reference, config.lookback)
    return StateResult(
        perspective="intra_case", config=config, features=fs, pca=pca, reduced=reduced,
        states=model, trajectories=traj, transitions=transitions,
        distribution=dist, signal=signal,
    )


def _run_windowed(log: pd.DataFrame, perspective: str, config, build_kwargs: dict) -> StateResult:
    """Shared runner for the two windowed perspectives (resource, inter-case)."""
    fs = build_features(log, perspective, **build_kwargs)
    if len(fs.matrix) < 2:
        raise ValueError(
            f"Window W is wider than the log span — only {len(fs.matrix)} window emerged. "
            "Pick a smaller window_minutes."
        )
    matrix = fs.values()
    reduced, pca = reduce(matrix, config.skip_pca, config.pca_components, config.scaling)
    model = cluster(reduced, method=config.clustering,
                    **_cluster_params(config, config.clustering))
    if perspective == "inter_case":
        raw_centroids = state_means(matrix, model.state_ids, model.n_states)
        model = replace(model, dominant=describe_states(raw_centroids, fs.columns))
    traj = trajectories(fs, model)
    window_starts = fs.index["window_start"]
    transitions = find_transitions(window_starts, model.state_ids, model.labels, fs.matrix)
    dist = state_distribution(window_starts, model.state_ids, model.n_states,
                              config.window_minutes)
    signal = window_vector_shift(window_starts, reduced, config.signal_metric)
    return StateResult(
        perspective=perspective, config=config, features=fs, pca=pca, reduced=reduced,
        states=model, trajectories=traj, transitions=transitions,
        distribution=dist, signal=signal,
    )


def run_resource(log: pd.DataFrame, config: ResourceConfig = ResourceConfig()) -> StateResult:
    """Windowed resource features → (PCA) → clustering → trajectory, transitions,
    and the window-to-window vector-distance drift signal."""
    return _run_windowed(log, "resource", config, {
        "features": config.features,
        "window_minutes": config.window_minutes,
        "resources": config.resources or None,
        "activities": config.activities or None,
    })


def run_inter_case(log: pd.DataFrame, config: InterConfig = InterConfig()) -> StateResult:
    """Windowed inter-case features → (PCA) → clustering → trajectory, transitions,
    and the window-to-window vector-distance drift signal."""
    return _run_windowed(log, "inter_case", config, {
        "features": config.features or None,
        "window_minutes": config.window_minutes,
        "stall_minutes": config.stall_minutes,
        "numeric_attrs": config.numeric_attrs,
        "categorical_attrs": config.categorical_attrs,
    })


def run(log: pd.DataFrame, perspective: str, config=None) -> StateResult:
    """Run one perspective's pipeline with its (default) config."""
    runners = {"intra_case": run_intra_case, "resource": run_resource, "inter_case": run_inter_case}
    if perspective not in runners:
        raise ValueError(f"Unknown perspective: {perspective!r} (use one of {tuple(runners)})")
    if config is None:
        config = CONFIGS[perspective]()
    return runners[perspective](log, config)
