"""Prompt assembly and the high-level LLM entry points."""
from __future__ import annotations

import json
import re
from dataclasses import fields

import pandas as pd

from ..abstract import abstract_result
from ..cluster import DISTANCES, METHODS, SOM_INIT
from ..drift import DIVERGENCES, REFERENCES
from ..pipeline import CONFIGS
from ..plots import save_figure
from ..reduce import SCALING
from ..schema import FEATURE_LABELS
from .connectors import anthropic_query
from .knowledge import DEFAULT_PARTS, inject


def build_prompt(
    question: str,
    result=None,
    log: pd.DataFrame | None = None,
    parts: tuple[str, ...] = DEFAULT_PARTS,
    max_len: int = 12000,
) -> str:
    """Assemble the full prompt without any network call.

    Blocks, each under its own share of `max_len`: the knowledge injection,
    the current log, the analysis result, and the question.
    """
    blocks: list[str] = []
    knowledge = inject(log=log, parts=parts, max_len=max_len // 3)
    if knowledge:
        blocks.append(f"<knowledge>\n{knowledge}\n</knowledge>")
    if result is not None:
        blocks.append(f"<analysis>\n{abstract_result(result, max_len=max_len // 2)}\n</analysis>")
    blocks.append(f"<question>\n{question}\n</question>")
    return "\n\n".join(blocks)


def ask(
    question: str,
    result=None,
    log: pd.DataFrame | None = None,
    executor=anthropic_query,
    parts: tuple[str, ...] = DEFAULT_PARTS,
    max_len: int = 12000,
    **kwargs,
) -> str:
    """Ask a question about an analysis result and/or a log.

    `executor` is any callable ``(prompt, **kwargs) -> str`` — one of the kairo
    connectors, or your own function.
    """
    return executor(build_prompt(question, result=result, log=log, parts=parts, max_len=max_len),
                    **kwargs)


def explain_plot(
    fig,
    question: str | None = None,
    result=None,
    log: pd.DataFrame | None = None,
    executor=anthropic_query,
    parts: tuple[str, ...] = ("drift_detection", "method", "statistics"),
    max_len: int = 8000,
    **kwargs,
) -> str:
    """Send a kairo figure to a vision-capable model and return its reading.

    The figure is rendered to a PNG (needs kaleido: ``pip install kairo[vision]``)
    and attached to the prompt. The executor must accept an `image_path` keyword
    — all kairo connectors do.
    """
    path = save_figure(fig)
    question = question or "Explain what this plot from the state-based drift analysis shows."
    prompt = build_prompt(question, result=result, log=log, parts=parts, max_len=max_len)
    try:
        return executor(prompt, image_path=str(path), **kwargs)
    finally:
        path.unlink(missing_ok=True)


def _config_schema(perspective: str) -> str:
    """A textual schema of a perspective's config for the model to fill in."""
    config_cls = CONFIGS[perspective]
    lines = [f"Fields of {config_cls.__name__} (JSON object, all fields optional):"]
    for field in fields(config_cls):
        lines.append(f"- {field.name}: {field.type}, default {field.default!r}")
    lines += [
        f"Allowed values — clustering: {list(METHODS)}, metric: {list(DISTANCES)}, "
        f"scaling: {list(SCALING)}, som_init: {list(SOM_INIT)}",
        f"features must come from: {list(FEATURE_LABELS[perspective])}",
    ]
    if perspective == "intra_case":
        lines.append(f"divergence: {list(DIVERGENCES)}, reference: {list(REFERENCES)}")
    return "\n".join(lines)


def nlp_to_config(
    request: str,
    perspective: str,
    log: pd.DataFrame | None = None,
    executor=anthropic_query,
    obtain_config: bool = True,
    **kwargs,
):
    """Turn a natural-language request into a pipeline config dataclass.

    With ``obtain_config=False`` the assembled prompt is returned instead of
    being executed — inspect it, or run it elsewhere.
    """
    if perspective not in CONFIGS:
        raise ValueError(f"Unknown perspective: {perspective!r} (use one of {tuple(CONFIGS)})")
    prompt = build_prompt(
        "Translate the following request into pipeline parameters.\n\n"
        f"Request: {request}\n\n{_config_schema(perspective)}\n\n"
        "Answer with ONLY a JSON object between ```json and ``` fences, containing just "
        "the fields that should differ from their defaults.",
        log=log,
        parts=("method", "log") if log is not None else ("method",),
    )
    if not obtain_config:
        return prompt
    response = executor(prompt, **kwargs)
    match = re.search(r"```json\s*(.*?)```", response, flags=re.DOTALL)
    raw = match.group(1) if match else response
    values = json.loads(raw)
    config_cls = CONFIGS[perspective]
    known = {f.name: f for f in fields(config_cls)}
    kept = {}
    for key, value in values.items():
        if key not in known:
            continue
        if isinstance(value, list):
            value = tuple(value)
        kept[key] = value
    return config_cls(**kept)
