"""Canonical log schema: the mappable roles and the columns each feature needs."""
from __future__ import annotations

from typing import Iterable

import streamlit as st

MUST_HAVE_COLUMNS = ("case:concept:name", "concept:name", "time:timestamp")

# Role the user assigns a column to on the upload page -> canonical column name.
ROLES = {
    "case ID": "case:concept:name",
    "activity": "concept:name",
    "timestamp": "time:timestamp",
    "resource": "org:resource",
    "event duration": "event:duration_min",
}
OPTIONAL_ROLES = ("resource", "event duration")  # may be left unmapped
ROLE_OF = {column: role for role, column in ROLES.items()}
MAX_CASE_ATTRS = 10

INTRA_FEATURE_LABELS = {
    "activity_freq": "Activity frequencies",
    "bigram": "Directly-follows counts",
    "vocab": "Distinct activity set",
    "progress": "Case progress",
    "current": "Current activity",
    "history": "Past activities",
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
    "events_per_case": "Events per active case",
    "mean_delta_t": "Mean Δt",
    "std_delta_t": "Std Δt",
    "stalled_cases": "Stalled cases",
}
# Inter-case features one *mapped case attribute* contributes. They are not part
# of INTER_FEATURE_LABELS: the real feature keys carry the attribute's own name
# and only exist once a log is mapped, while these describe the kinds up front.
CASE_ATTRIBUTE = "case attribute"
ATTRIBUTE_FEATURE_LABELS = {
    "attr_mean": "Mean of a numeric case attribute",
    "attr_std": "Std of a numeric case attribute",
    "attr_share": "Value shares of a categorical case attribute",
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
    "events_per_case": ("case:concept:name", "time:timestamp"),
    "mean_delta_t": ("case:concept:name", "time:timestamp"),
    "std_delta_t": ("case:concept:name", "time:timestamp"),
    "stalled_cases": ("case:concept:name", "time:timestamp"),
}
ATTRIBUTE_FEATURE_COLUMNS = {key: ("time:timestamp", CASE_ATTRIBUTE) for key in ATTRIBUTE_FEATURE_LABELS}

PERSPECTIVES = (
    ("Intra-case", INTRA_FEATURE_LABELS, INTRA_FEATURE_COLUMNS),
    ("Resource", RESOURCE_FEATURE_LABELS, RESOURCE_FEATURE_COLUMNS),
    (
        "Inter-case",
        {**INTER_FEATURE_LABELS, **ATTRIBUTE_FEATURE_LABELS},
        {**INTER_FEATURE_COLUMNS, **ATTRIBUTE_FEATURE_COLUMNS},
    ),
)


def _requirement_text(req: str | tuple[str, ...]) -> str:
    """One requirement as the role it is mapped from; alternatives joined with 'or'."""
    if isinstance(req, tuple):
        return " or ".join(ROLE_OF.get(r, r) for r in req)
    return ROLE_OF.get(req, req)


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


def render_feature_requirements() -> None:
    """Static overview of the features and the mapped roles each one needs."""
    rows = ["| Perspective | Feature | Required columns |", "| --- | --- | --- |"]
    for perspective, labels, columns in PERSPECTIVES:
        for key, label in labels.items():
            needed = ", ".join(f"`{_requirement_text(r)}`" for r in columns[key])
            rows.append(f"| {perspective} | {label} | {needed} |")
    with st.expander("Required columns per feature"):
        st.markdown("\n".join(rows))
