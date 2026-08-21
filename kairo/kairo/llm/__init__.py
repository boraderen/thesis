"""LLM access: connectors, knowledge injection, and prompting helpers."""
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
    "anthropic_query", "openai_query", "google_query", "local_query", "query",
    "available_models", "inject", "log_knowledge",
    "build_prompt", "ask", "explain_plot", "nlp_to_config",
]
