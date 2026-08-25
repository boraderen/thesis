"""LLM access: connectors, knowledge injection, and prompting helpers."""
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
from .connectors import (
    DEFAULTS,
    PROVIDERS,
    anthropic_query,
    available_models,
    google_query,
    local_query,
    openai_query,
    query,
)
from .knowledge import DEFAULT_PARTS, KNOWLEDGE, inject, log_knowledge
from .prompt import ask, build_prompt, explain_plot, nlp_to_config

__all__ = [
    "DEFAULTS", "PROVIDERS", "KNOWLEDGE", "DEFAULT_PARTS",
    "abstract_config", "abstract_drift_signal", "abstract_features",
    "abstract_log_attributes", "abstract_log_statistics", "abstract_pca",
    "abstract_result", "abstract_state_distribution", "abstract_state_grid",
    "abstract_state_profiles", "abstract_states", "abstract_trajectory",
    "abstract_transition_matrix", "abstract_transitions",
    "anthropic_query", "openai_query", "google_query", "local_query", "query",
    "available_models", "inject", "log_knowledge",
    "build_prompt", "ask", "explain_plot", "nlp_to_config",
]
