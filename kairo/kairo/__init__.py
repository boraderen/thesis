"""kairo — state-based process monitoring and concept-drift detection.

Flat public API, pm4py-style: read a log, build features, reduce, cluster,
derive trajectories / transitions / distributions / drift signals — or run a
whole pipeline with one config. `kairo.abstract_*` turns every object into
text; `kairo.llm` talks to language models about the results.
"""
from . import abstract, llm, schema, windows
from .abstract import (
    abstract_config,
    abstract_drift_signal,
    abstract_features,
    abstract_log_attributes,
    abstract_log_statistics,
    abstract_pca,
    abstract_result,
    abstract_state_distribution,
    abstract_state_grid,
    abstract_state_profiles,
    abstract_states,
    abstract_trajectory,
    abstract_transition_matrix,
    abstract_transitions,
)
from .cluster import (
    DISTANCES,
    METHODS,
    SOM_INIT,
    SUPPORTED_DISTANCES,
    StateModel,
    cluster,
    cluster_dbscan,
    cluster_kmeans,
    cluster_som,
    k_distances,
    silhouette,
    state_distances,
)
from .drift import (
    DIVERGENCES,
    REFERENCES,
    divergence,
    drift_signal,
    top_drift_windows,
    window_vector_shift,
)
from .features import FeatureSet, build_features, select_columns
from .log import (
    LogStatistics,
    classify_attributes,
    log_statistics,
    map_columns,
    read_log,
    span_label,
)
from .pipeline import (
    CONFIGS,
    InterConfig,
    IntraConfig,
    ResourceConfig,
    StateResult,
    run,
    run_inter_case,
    run_intra_case,
    run_resource,
)
from .plots import (
    add_transition_markers,
    add_window_boundaries,
    plot_activity_frequency,
    plot_drift_signal,
    plot_k_distance,
    plot_pca_variance,
    plot_state_distances,
    plot_state_distribution,
    plot_state_grid,
    plot_trajectory,
    plot_transition_matrix,
    save_figure,
    state_bg,
    state_color,
    state_fg,
)
from .reduce import SCALING, PCAResult, fit_pca, reduce, standardize
from .states import (
    assign_states,
    case_transitions,
    find_transitions,
    state_distribution,
    state_profiles,
    trajectories,
    transition_matrix,
)

__version__ = "0.1.0"
