"""Canonical log schema: column names, mappable roles, and the feature catalog."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

# Canonical column names every kairo function relies on.
CASE = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
RESOURCE = "org:resource"
DURATION = "event:duration_min"

MUST_HAVE = (CASE, ACTIVITY, TIMESTAMP)

# Role a column can be mapped to -> canonical column name.
ROLES = {
    "case ID": CASE,
    "activity": ACTIVITY,
    "timestamp": TIMESTAMP,
    "resource": RESOURCE,
    "event duration": DURATION,
}
OPTIONAL_ROLES = ("resource", "event duration")
ROLE_OF = {column: role for role, column in ROLES.items()}
MAX_CASE_ATTRS = 10

PERSPECTIVES = ("intra_case", "resource", "inter_case")

# ---------------------------------------------------------------------------
# Feature catalog: key -> label, key -> description, key -> required columns.
# ---------------------------------------------------------------------------

INTRA_FEATURES = {
    "activity_freq": "Activity frequencies",
    "bigram": "Directly-follows counts",
    "vocab": "Distinct activity set",
    "progress": "Case progress",
    "current": "Current activity",
    "history": "Past activities",
}
RESOURCE_FEATURES = {
    "events": "Events per resource",
    "active": "Active cases per resource",
    "duration": "Mean event duration per resource",
    "wait": "Mean wait into resource",
    "activity_events": "Activity-resource event shares",
    "ho": "Handover shares",
}
INTER_FEATURES = {
    "active_cases": "Active cases",
    "new_arrivals": "New arrivals",
    "completions": "Completions",
    "events_per_case": "Events per active case",
    "mean_delta_t": "Mean Δt",
    "std_delta_t": "Std Δt",
    "stalled_cases": "Stalled cases",
}
# Feature kinds one mapped case attribute contributes to the inter-case matrix.
CASE_ATTRIBUTE = "case attribute"
ATTRIBUTE_FEATURES = {
    "attr_mean": "Mean of a numeric case attribute",
    "attr_std": "Std of a numeric case attribute",
    "attr_share": "Value shares of a categorical case attribute",
}

FEATURE_LABELS: dict[str, dict[str, str]] = {
    "intra_case": INTRA_FEATURES,
    "resource": RESOURCE_FEATURES,
    "inter_case": {**INTER_FEATURES, **ATTRIBUTE_FEATURES},
}

FEATURE_DESCRIPTIONS = {
    # intra-case
    "activity_freq": "One column per activity — how often the activity occurred in the case so far, divided by the number of events so far (the prefix).",
    "bigram": "One column per observed directly-follows pair A→B — how often that transition occurred in the prefix, divided by the number of transitions so far.",
    "vocab": "One binary column per activity — 1 if the activity has already occurred in the case prefix, 0 otherwise.",
    "progress": "Position of the event within its case as a fraction of the total case length (last event = 1).",
    "current": "One binary column per activity — 1 for the activity of this event.",
    "history": "For each of the last n events of the same case, one binary column per activity — 1 for that event's activity. Before the case has n predecessors, the missing slots stay 0.",
    # resource
    "events": "Number of events assigned to each resource in the calendar window.",
    "active": "Number of distinct cases touched by each resource in the window.",
    "duration": "Mean event duration in minutes for each resource in the window (requires a mapped event duration column).",
    "wait": "Computed for each resource r: mean difference between this event's and the previous case event's timestamp, in minutes, over r's events in the window — counted only for events whose previous event was executed by a different resource.",
    "activity_events": "Computed for each activity a and each resource r: the share of a's events in the window executed by r — a-events by r divided by all a-events in the window.",
    "ho": "Computed for each ordered resource pair r1→r2: the share of r1's within-case handovers in the window that go to r2 — handovers r1→r2 divided by all handovers from r1 in the window.",
    # inter-case
    "active_cases": "Number of distinct cases with at least one event in the calendar window.",
    "new_arrivals": "Number of cases whose first event falls inside the window.",
    "completions": "Number of cases whose last event falls inside the window.",
    "events_per_case": "Events in the window divided by the number of cases active in it.",
    "mean_delta_t": "Mean gap between an event and the previous event of the same case, in minutes.",
    "std_delta_t": "Standard deviation of those within-case gaps, in minutes.",
    "stalled_cases": "Number of cases still running at the window end whose most recent event is older than the stall threshold τ. Completed cases are not counted.",
    # case attributes
    "attr_mean": "Mean of a numeric case attribute over the events in the window.",
    "attr_std": "Standard deviation of a numeric case attribute over the events in the window.",
    "attr_share": "One column per value of a categorical case attribute — the share of the window's events carrying that value.",
}

# feature key -> columns the log must contain; an inner tuple means any one of them.
FEATURE_COLUMNS: dict[str, dict[str, tuple]] = {
    "intra_case": {key: MUST_HAVE for key in INTRA_FEATURES},
    "resource": {
        "events": (TIMESTAMP, RESOURCE),
        "active": (CASE, TIMESTAMP, RESOURCE),
        "duration": (TIMESTAMP, RESOURCE, DURATION),
        "wait": (CASE, TIMESTAMP, RESOURCE),
        "activity_events": (ACTIVITY, TIMESTAMP, RESOURCE),
        "ho": (CASE, TIMESTAMP, RESOURCE),
    },
    "inter_case": {
        **{key: (CASE, TIMESTAMP) for key in INTER_FEATURES},
        **{key: (TIMESTAMP, CASE_ATTRIBUTE) for key in ATTRIBUTE_FEATURES},
    },
}


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
    present: Iterable[str], perspective: str
) -> tuple[list[str], dict[str, str]]:
    """Split a perspective's feature keys into computable ones and {key: missing text}."""
    available: list[str] = []
    disabled: dict[str, str] = {}
    for key, required in FEATURE_COLUMNS[perspective].items():
        miss = missing_columns(present, required)
        if miss:
            disabled[key] = ", ".join(miss)
        else:
            available.append(key)
    return available, disabled


def requirements_table() -> pd.DataFrame:
    """Static overview of the features and the mapped roles each one needs."""
    rows = []
    for perspective in PERSPECTIVES:
        for key, label in FEATURE_LABELS[perspective].items():
            needed = ", ".join(_requirement_text(r) for r in FEATURE_COLUMNS[perspective][key])
            rows.append({"perspective": perspective, "feature": label, "required columns": needed})
    return pd.DataFrame(rows)
