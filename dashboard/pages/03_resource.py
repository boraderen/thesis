"""Resource SOM page: per-window resource workload → PCA → SOM cells → trajectory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.features.resource import build_features
from core.loader import span_label
from core.pca import fit_pca
from core.som import train_som
from core.transitions import find_transitions
from core.windows import (
    SOM_GRID_OPTIONS,
    default_window_minutes,
    log_span_minutes,
    som_grid_label,
    window_minute_choices,
    window_minute_label,
)
from viz.som_grid import som_heatmap
from viz.tables import styled_feature_table
from viz.trajectory import add_transition_markers, pca_variance_plot, state_timeline

st.set_page_config(page_title="Resource SOM", layout="wide")
st.title("3 — Resource SOM")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))
if "resource" not in log.columns:
    st.error("This log has no `resource` column — the resource SOM is disabled.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    default_W = int(st.session_state.get("resource_W", default_window_minutes(log_span_minutes(log))))
    options = window_minute_choices(default_W)
    window_minutes = st.selectbox(
        "Window W", options, index=options.index(default_W), format_func=window_minute_label
    )
    st.session_state["resource_W"] = window_minutes
    grid_default = st.session_state.get("resource_grid", (2, 3))
    if grid_default not in SOM_GRID_OPTIONS:
        grid_default = (2, 3)
    grid_h, grid_w = st.selectbox(
        "SOM grid", SOM_GRID_OPTIONS,
        index=SOM_GRID_OPTIONS.index(grid_default),
        format_func=som_grid_label,
    )
    st.session_state["resource_grid"] = (grid_h, grid_w)
    pca_k = st.number_input(
        "PCA components (0 = auto/elbow)",
        min_value=0, max_value=20, value=int(st.session_state.get("resource_pca_k", 0)), step=1,
    )
    st.session_state["resource_pca_k"] = pca_k

matrix_df, spec = build_features(log, window_minutes=window_minutes)
if len(matrix_df) < 2:
    st.warning(
        f"Window W is wider than the log span — only {len(matrix_df)} window emerged. "
        "Pick a smaller W to get a trajectory."
    )
    st.stop()

KIND_LABELS = {
    "events": "Events per resource",
    "active": "Active cases per resource",
    "duration": "Mean event duration per resource",
    "wait": "Mean wait into resource",
    "activity_events": "Activity-resource event counts",
    "ho": "Handover counts",
}
KIND_DESCRIPTIONS = {
    "events": "Number of events assigned to each resource in the calendar window.",
    "active": "Number of distinct cases touched by each resource in the window.",
    "duration": "Mean event duration in minutes for each resource in the window.",
    "wait": "Mean log-minutes between the previous event and this resource's event, counted only when the resource changes.",
    "activity_events": "Number of events for each activity-resource pair in the window, such as activity a performed by res_001.",
    "ho": "Number of within-case handovers from one resource to another inside the window.",
}
kind_groups = {k: [c for c in spec.columns if c.startswith(f"{k}:")] for k in KIND_LABELS}

with st.sidebar:
    st.subheader("Features for state clustering")
    picked_kinds = st.multiselect(
        "Feature kinds",
        options=list(KIND_LABELS),
        default=list(KIND_LABELS),
        format_func=lambda k: KIND_LABELS[k],
    )
    picked_resources = st.multiselect(
        "Resources",
        options=spec.resources,
        default=spec.resources,
    )
    picked_activities = st.multiselect(
        "Activities",
        options=spec.activities,
        default=spec.activities,
        help="Used by activity-resource event count features.",
    )
    with st.expander("Feature glossary"):
        for kind, label in KIND_LABELS.items():
            st.markdown(f"**{label}.** {KIND_DESCRIPTIONS[kind]}")

def _matches_resource(column: str) -> bool:
    return any(column.endswith(f":{resource}") for resource in picked_resources)


def _matches_activity_resource(column: str) -> bool:
    return any(
        column.startswith(f"activity_events:{activity}:") and column.endswith(f":{resource}")
        for activity in picked_activities
        for resource in picked_resources
    )


selected_cols = []
for kind in picked_kinds:
    if kind == "ho":
        selected_cols.extend(
            c for c in kind_groups[kind]
            if any(f"ho:{a}→{b}" == c for a in picked_resources for b in picked_resources if a != b)
        )
    elif kind == "activity_events":
        selected_cols.extend(c for c in kind_groups[kind] if _matches_activity_resource(c))
    else:
        selected_cols.extend(c for c in kind_groups[kind] if _matches_resource(c))
if not selected_cols:
    st.warning("Select at least one feature kind, one resource, and one activity when using activity-resource features.")
    st.stop()

st.subheader("Feature matrix")
st.caption(
    f"{len(matrix_df):,} windows × {len(selected_cols)} columns (W={spec.window_minutes} min, "
    f"{len(spec.resources)} resources)"
)
preview_cols = ["window_start", *selected_cols]
preview_groups = {g: [c for c in cols if c in selected_cols] for g, cols in spec.groups.items()}
styled = styled_feature_table(matrix_df[preview_cols], preview_groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

mat = matrix_df[selected_cols].to_numpy()
pca = fit_pca(mat, force_k=int(pca_k) if pca_k else None)
st.subheader("PCA")
st.plotly_chart(pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim), width="stretch")

som = train_som(pca.transformed, grid_h=grid_h, grid_w=grid_w, annotations=None)
matrix_df = matrix_df.assign(state_id=som.state_ids)
st.session_state["resource_matrix"] = matrix_df
st.session_state["resource_spec"] = spec
st.session_state["resource_som"] = som

col_l, col_r = st.columns([1, 1])
with col_l:
    st.subheader("SOM grid")
    st.plotly_chart(
        som_heatmap(
            som.grid_h, som.grid_w, som.cell_counts, som.cell_labels,
            title="Window states", dominants=som.cell_dominant,
        ),
        width="stretch",
    )
with col_r:
    st.subheader("Resource state trajectory")
    transitions = find_transitions(
        matrix_df["window_start"], som.state_ids, som.cell_labels, matrix_df[selected_cols]
    )
    fig = state_timeline(
        matrix_df["window_start"], som.state_ids, som.cell_labels,
        title=f"State per {spec.window_minutes}-min window",
        cell_dominant=som.cell_dominant,
    )
    if not transitions.empty:
        add_transition_markers(fig, transitions["timestamp"])
    st.plotly_chart(fig, width="stretch")

st.subheader("Transitions")
if transitions.empty:
    st.caption("No state changes in this trajectory.")
else:
    st.caption(f"{len(transitions)} transitions detected.")
    st.dataframe(
        transitions[["timestamp", "from", "to", "top_changes"]].head(500).rename(columns={"timestamp": "at"}),
        width="stretch", height=min(420, 60 + 36 * len(transitions)),
    )
