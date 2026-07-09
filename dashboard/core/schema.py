"""Canonical log schema: expected column names and per-feature column requirements."""
from __future__ import annotations

from typing import Iterable

import streamlit as st

MUST_HAVE_COLUMNS = ("case:concept:name", "concept:name", "time:timestamp")

# (what the column holds, the name it must have)
OPTIONAL_COLUMNS = (
    ("the executing resource of an event", "org:resource"),
    ("an event's duration in minutes", "event:duration_min"),
)

INTRA_FEATURE_LABELS = {
    "activity_freq": "Activity frequencies",
    "bigram": "Directly-follows counts",
    "vocab": "Distinct activity set",
    "progress": "Case progress",
}
RESOURCE_FEATURE_LABELS = {
    "events": "Events per resource",
    "active": "Active cases per resource",
    "duration": "Mean event duration per resource",
    "wait": "Mean wait into resource",
    "activity_events": "Activity-resource event shares",
    "ho": "Handover shares",
}
INTER_FEATURE_LABELS = {
    "active_cases": "Active cases",
    "new_arrivals": "New arrivals",
    "completions": "Completions",
    "total_events": "Total events",
    "mean_delta_t": "Mean Δt (ln min)",
    "std_delta_t": "Std Δt (ln min)",
    "stalled_cases": "Stalled cases",
}

# feature key -> columns the log must contain; an inner tuple means any one of them.
INTRA_FEATURE_COLUMNS = {key: MUST_HAVE_COLUMNS for key in INTRA_FEATURE_LABELS}
RESOURCE_FEATURE_COLUMNS = {
    "events": ("time:timestamp", "org:resource"),
    "active": ("case:concept:name", "time:timestamp", "org:resource"),
    "duration": ("time:timestamp", "org:resource", "event:duration_min"),
    "wait": ("case:concept:name", "time:timestamp", "org:resource"),
    "activity_events": ("concept:name", "time:timestamp", "org:resource"),
    "ho": ("case:concept:name", "time:timestamp", "org:resource"),
}
INTER_FEATURE_COLUMNS = {
    "active_cases": ("case:concept:name", "time:timestamp"),
    "new_arrivals": ("case:concept:name", "time:timestamp"),
    "completions": ("case:concept:name", "time:timestamp"),
    "total_events": ("time:timestamp",),
    "mean_delta_t": ("time:timestamp",),
    "std_delta_t": ("time:timestamp",),
    "stalled_cases": ("case:concept:name", "time:timestamp"),
}

PERSPECTIVES = (
    ("Intra-case", INTRA_FEATURE_LABELS, INTRA_FEATURE_COLUMNS),
    ("Resource", RESOURCE_FEATURE_LABELS, RESOURCE_FEATURE_COLUMNS),
    ("Inter-case", INTER_FEATURE_LABELS, INTER_FEATURE_COLUMNS),
)


def _requirement_text(req: str | tuple[str, ...]) -> str:
    """One requirement as text; alternatives joined with 'or'."""
    return " or ".join(req) if isinstance(req, tuple) else req


def missing_columns(present: Iterable[str], required: tuple) -> list[str]:
    """Unmet requirements; an inner tuple is met when any of its names is present."""
    cols = set(present)
    return [
        _requirement_text(req)
        for req in required
        if (cols.isdisjoint(req) if isinstance(req, tuple) else req not in cols)
    ]


def feature_availability(
    present: Iterable[str], feature_columns: dict[str, tuple]
) -> tuple[list[str], dict[str, str]]:
    """Split feature keys into computable ones and {key: missing-columns text}."""
    available: list[str] = []
    disabled: dict[str, str] = {}
    for key, required in feature_columns.items():
        miss = missing_columns(present, required)
        if miss:
            disabled[key] = ", ".join(miss)
        else:
            available.append(key)
    return available, disabled


def render_schema_help() -> None:
    """Expected-columns section shared by the landing and upload pages."""
    st.markdown(
        "**Expected columns** `case:concept:name`, "
        "`concept:name`, `time:timestamp`"
    )
    st.markdown(
        "\n".join(
            f"- If your log has a column for {concept}, its name must be `{name}`."
            for concept, name in OPTIONAL_COLUMNS
        )
    )
    rows = ["| Perspective | Feature | Required columns |", "| --- | --- | --- |"]
    for perspective, labels, columns in PERSPECTIVES:
        for key, label in labels.items():
            needed = ", ".join(f"`{_requirement_text(r)}`" for r in columns[key])
            rows.append(f"| {perspective} | {label} | {needed} |")
    with st.expander("Required columns per feature"):
        st.markdown("\n".join(rows))
