"""The flat public surface: every advertised name resolves and every module imports."""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import kairo

SUBPACKAGES = ("data", "features", "analysis", "pipeline", "viz", "llm")

FLAT_API = (
    # the event log
    "read_log", "map_columns", "log_statistics", "LogStatistics",
    # pipelines
    "run", "run_intra_case", "run_resource", "run_inter_case",
    "IntraConfig", "ResourceConfig", "InterConfig", "StateResult",
    # plots
    "plot_activity_frequency", "plot_pca_variance", "plot_k_distance", "plot_state_grid",
    "plot_state_distances", "plot_transition_matrix", "plot_trajectory",
    "plot_state_distribution", "plot_drift_signal", "add_transition_markers",
    "add_window_boundaries", "save_figure",
    # abstractions
    "abstract_log_statistics", "abstract_log_attributes", "abstract_features",
    "abstract_pca", "abstract_states", "abstract_state_grid", "abstract_state_profiles",
    "abstract_trajectory", "abstract_transitions", "abstract_transition_matrix",
    "abstract_state_distribution", "abstract_drift_signal", "abstract_config",
    "abstract_result",
    # language models
    "query", "anthropic_query", "openai_query", "google_query", "local_query",
    "build_prompt", "ask", "explain_plot", "nlp_to_config",
    # the values a config accepts
    "DISTANCES", "SUPPORTED_DISTANCES", "METHODS", "SOM_INIT", "SCALING",
    "DIVERGENCES", "REFERENCES", "PROVIDERS", "FEATURE_LABELS",
)

# The individual steps a runner chains: usable, but not on the flat surface.
NOT_FLAT = (
    "build_features", "select_columns", "FeatureSet",
    "standardize", "fit_pca", "reduce", "PCAResult",
    "cluster", "cluster_som", "cluster_kmeans", "cluster_dbscan",
    "k_distances", "state_distances", "silhouette", "StateModel",
    "trajectories", "find_transitions", "case_transitions",
    "transition_matrix", "state_distribution", "state_profiles",
    "divergence", "drift_signal", "window_vector_shift",
    # helpers
    "classify_attributes", "span_label", "feature_availability", "requirements_table",
    "window_minute_label", "default_window_minutes", "attribute_feature_catalog",
    "describe_states", "state_means", "assign_states", "top_drift_windows",
    "CONFIGS", "available_models", "inject", "log_knowledge",
    "state_color", "state_bg", "state_fg", "DISTINCT_COLORS",
    "KNOWLEDGE", "DEFAULT_PARTS", "FEATURE_DESCRIPTIONS",
)


@pytest.mark.parametrize("name", FLAT_API)
def test_flat_api_name_resolves(name):
    assert hasattr(kairo, name), f"kairo.{name} is not exported"


def test_every_module_imports():
    """Catches a helper left behind by a move — the module would fail at import."""
    for info in pkgutil.walk_packages(kairo.__path__, prefix="kairo."):
        importlib.import_module(info.name)


@pytest.mark.parametrize("pkg", SUBPACKAGES)
def test_subpackage_reachable(pkg):
    assert hasattr(kairo, pkg)


def test_flat_and_namespaced_are_the_same_object():
    assert kairo.ask is kairo.llm.ask
    assert kairo.read_log is kairo.data.read_log
    assert kairo.plot_state_grid is kairo.viz.plot_state_grid
    assert kairo.run_intra_case is kairo.pipeline.run_intra_case


@pytest.mark.parametrize("name", NOT_FLAT)
def test_helpers_stay_off_the_top_level(name):
    """The flat surface is the analysis steps, not every helper."""
    assert not hasattr(kairo, name), f"kairo.{name} should not be exported"


def test_steps_remain_usable_in_their_subpackage():
    """The flat surface is small, but every step is still callable."""
    import kairo.features.inter
    import kairo.viz.palette

    # pipeline steps
    assert kairo.features.build_features and kairo.features.select_columns
    assert kairo.analysis.reduce and kairo.analysis.fit_pca and kairo.analysis.standardize
    assert kairo.analysis.cluster and kairo.analysis.cluster_som
    assert kairo.analysis.cluster_kmeans and kairo.analysis.cluster_dbscan
    assert kairo.analysis.trajectories and kairo.analysis.find_transitions
    assert kairo.analysis.state_distribution and kairo.analysis.state_profiles
    assert kairo.analysis.drift_signal and kairo.analysis.window_vector_shift
    assert kairo.analysis.k_distances and kairo.analysis.silhouette
    # data structures
    assert kairo.features.FeatureSet and kairo.analysis.StateModel and kairo.analysis.PCAResult
    # helpers
    assert kairo.data.classify_attributes and kairo.data.span_label
    assert kairo.data.requirements_table and kairo.data.feature_availability
    assert kairo.data.window_minute_label and kairo.data.default_window_minutes
    assert kairo.analysis.state_means and kairo.analysis.top_drift_windows
    assert kairo.features.inter.attribute_feature_catalog
    assert kairo.features.inter.describe_states
    assert kairo.llm.available_models and kairo.llm.inject
    assert kairo.viz.palette.state_bg
    assert kairo.pipeline.CONFIGS


def test_a_full_manual_run_without_the_flat_api(log):
    """Driving the steps by hand must reproduce what the runner does."""
    cfg = kairo.IntraConfig(clustering="kmeans", n_clusters=4, pca_components=2,
                            window_minutes=1440)
    fs = kairo.features.build_features(log, "intra_case", features=cfg.features,
                                       history=cfg.history)
    reduced, _ = kairo.analysis.reduce(fs.values(), n_components=2)
    model = kairo.analysis.cluster_kmeans(reduced, n_clusters=4, seed=cfg.seed)
    dist = kairo.analysis.state_distribution(fs.index["time:timestamp"], model.state_ids,
                                             model.n_states, cfg.window_minutes)
    signal = kairo.analysis.drift_signal(dist, cfg.divergence, cfg.reference, cfg.lookback)

    ran = kairo.run_intra_case(log, cfg)
    assert (model.state_ids == ran.states.state_ids).all()
    assert signal["score"].equals(ran.signal["score"])
