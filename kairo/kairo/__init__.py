"""kairo — state-based process monitoring and concept-drift detection.

The top level is deliberately small: read a log, run one of the three
pipelines, then plot, describe, or ask a language model about the result.

    log    = kairo.read_log("log.xes")
    log    = kairo.map_columns(log, {...})
    result = kairo.run_intra_case(log, kairo.IntraConfig(...))

    kairo.plot_state_grid(result.states).show()
    print(kairo.abstract_result(result))
    kairo.ask("Where does it drift?", result=result, log=log)

The individual steps a runner chains — feature extraction, PCA, clustering,
trajectories, drift measures — live in the subpackages and are equally usable
on their own: ``kairo.features``, ``kairo.analysis``, ``kairo.data``,
``kairo.viz``, ``kairo.llm``, ``kairo.pipeline``.
"""
from . import analysis, data, features, llm, pipeline, viz

# --- the event log -------------------------------------------------------
from .data.log import LogStatistics, log_statistics, map_columns, read_log

# --- pipelines -----------------------------------------------------------
from .pipeline.config import InterConfig, IntraConfig, ResourceConfig
from .pipeline.run import StateResult, run, run_inter_case, run_intra_case, run_resource

# --- figures -------------------------------------------------------------
from .viz.plots import (
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
)

# --- text abstractions ---------------------------------------------------
from .llm.abstract import (
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

# --- language models -----------------------------------------------------
from .llm.connectors import (
    PROVIDERS,
    anthropic_query,
    google_query,
    local_query,
    openai_query,
    query,
)
from .llm.prompt import ask, build_prompt, explain_plot, nlp_to_config

# --- the values a config accepts -----------------------------------------
from .analysis.cluster import DISTANCES, METHODS, SOM_INIT, SUPPORTED_DISTANCES
from .analysis.drift import DIVERGENCES, REFERENCES
from .analysis.reduce import SCALING
from .data.schema import FEATURE_LABELS

__version__ = "0.1.0"
