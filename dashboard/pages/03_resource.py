"""Resource SOM page: per-window resource workload → PCA → SOM cells → trajectory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.controls import log_signature, seed_choice, seed_multi, seed_widget
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

RESOURCE_SCHEMA_VERSION = "resource_window_v1"
RESOURCE_RESULT_KEYS = (
    "resource_matrix",
    "resource_spec",
    "resource_pca",
    "resource_som",
    "resource_selected_cols",
    "resource_run_config",
    "resource_log_signature",
)

current_log_signature = log_signature(log, RESOURCE_SCHEMA_VERSION)
if (
    st.session_state.get("resource_log_signature") is not None
    and st.session_state.get("resource_log_signature") != current_log_signature
):
    for key in (*RESOURCE_RESULT_KEYS, "resource_W_sel"):
        st.session_state.pop(key, None)

# Mirror how build_features derives its vocabularies so the multiselect options
# match spec.resources / spec.activities without building the window matrix.
all_resources = sorted(log["resource"].astype(str).dropna().unique().tolist())
all_activities = sorted(log["activity"].astype(str).dropna().unique().tolist())

run_cfg = st.session_state.get("resource_run_config") or {}
seed_multi("resource_kinds_sel", run_cfg.get("kinds", list(KIND_LABELS)), list(KIND_LABELS))
seed_multi("resource_res_sel", run_cfg.get("resources", all_resources), all_resources)
seed_multi("resource_act_sel", run_cfg.get("activities", all_activities), all_activities)
seed_widget("resource_pca_k_sel", int(run_cfg.get("pca_k", 0)))
grid_default = tuple(run_cfg.get("grid", (2, 3)))
seed_choice(
    "resource_grid_sel",
    grid_default if grid_default in SOM_GRID_OPTIONS else (2, 3),
    SOM_GRID_OPTIONS,
)
seed_widget(
    "resource_W_sel",
    int(run_cfg.get("window_minutes", default_window_minutes(log_span_minutes(log)))),
)

with st.sidebar:
    st.header("Controls")
    with st.form("resource_pipeline_controls"):
        st.subheader("Features for state clustering")
        picked_kinds = st.multiselect(
            "Feature kinds",
            options=list(KIND_LABELS),
            key="resource_kinds_sel",
            format_func=lambda k: KIND_LABELS[k],
        )
        picked_resources = st.multiselect(
            "Resources",
            options=all_resources,
            key="resource_res_sel",
        )
        picked_activities = st.multiselect(
            "Activities",
            options=all_activities,
            key="resource_act_sel",
            help="Used by activity-resource event count features.",
        )
        with st.expander("Feature glossary"):
            for kind, label in KIND_LABELS.items():
                st.markdown(f"**{label}.** {KIND_DESCRIPTIONS[kind]}")
        st.subheader("PCA")
        pca_k = st.number_input(
            "PCA components (0 = auto/elbow)",
            min_value=0,
            max_value=20,
            step=1,
            key="resource_pca_k_sel",
        )
        st.subheader("SOM")
        grid_h, grid_w = st.selectbox(
            "SOM grid",
            SOM_GRID_OPTIONS,
            key="resource_grid_sel",
            format_func=som_grid_label,
        )
        st.subheader("Windows")
        window_minutes = st.selectbox(
            "Window W",
            window_minute_choices(st.session_state["resource_W_sel"]),
            key="resource_W_sel",
            format_func=window_minute_label,
        )
        run_pipeline = st.form_submit_button(
            "Run resource pipeline",
            width="stretch",
        )


def _selected_columns(
    columns: list[str],
    picked_kinds: list[str],
    picked_resources: list[str],
    picked_activities: list[str],
) -> list[str]:
    """Filter spec columns down to the picked kinds / resources / activities."""

    def matches_resource(column: str) -> bool:
        return any(column.endswith(f":{resource}") for resource in picked_resources)

    def matches_activity_resource(column: str) -> bool:
        return any(
            column.startswith(f"activity_events:{activity}:") and column.endswith(f":{resource}")
            for activity in picked_activities
            for resource in picked_resources
        )

    selected: list[str] = []
    for kind in picked_kinds:
        kind_cols = [c for c in columns if c.startswith(f"{kind}:")]
        if kind == "ho":
            selected.extend(
                c for c in kind_cols
                if any(f"ho:{a}→{b}" == c for a in picked_resources for b in picked_resources if a != b)
            )
        elif kind == "activity_events":
            selected.extend(c for c in kind_cols if matches_activity_resource(c))
        else:
            selected.extend(c for c in kind_cols if matches_resource(c))
    return selected


if run_pipeline:
    with st.spinner("Building features, fitting PCA, training SOM…"):
        matrix_df, spec = build_features(log, window_minutes=int(window_minutes))
        if len(matrix_df) < 2:
            st.warning(
                f"Window W is wider than the log span — only {len(matrix_df)} window emerged. "
                "Pick a smaller W to get a trajectory."
            )
            st.stop()
        selected_cols = _selected_columns(spec.columns, picked_kinds, picked_resources, picked_activities)
        if not selected_cols:
            st.warning(
                "Select at least one feature kind, one resource, and one activity "
                "when using activity-resource features."
            )
            st.stop()
        pca = fit_pca(matrix_df[selected_cols].to_numpy(), force_k=int(pca_k) if pca_k else None)
        som = train_som(pca.transformed, grid_h=grid_h, grid_w=grid_w, annotations=None)
        matrix_df = matrix_df.assign(state_id=som.state_ids)
    st.session_state["resource_matrix"] = matrix_df
    st.session_state["resource_spec"] = spec
    st.session_state["resource_pca"] = pca
    st.session_state["resource_som"] = som
    st.session_state["resource_selected_cols"] = selected_cols
    st.session_state["resource_log_signature"] = current_log_signature
    st.session_state["resource_run_config"] = {
        "grid": (grid_h, grid_w),
        "pca_k": int(pca_k),
        "window_minutes": int(window_minutes),
        "kinds": list(picked_kinds),
        "resources": list(picked_resources),
        "activities": list(picked_activities),
    }

if (
    any(key not in st.session_state for key in RESOURCE_RESULT_KEYS)
    or st.session_state.get("resource_log_signature") != current_log_signature
):
    st.info("Set the sidebar controls and run the resource pipeline.")
    st.stop()

matrix_df = st.session_state["resource_matrix"]
spec = st.session_state["resource_spec"]
pca = st.session_state["resource_pca"]
som = st.session_state["resource_som"]
selected_cols = st.session_state["resource_selected_cols"]

st.subheader("Feature matrix")
st.caption(
    f"{len(matrix_df):,} windows × {len(selected_cols)} columns (W={spec.window_minutes} min, "
    f"{len(spec.resources)} resources)"
)
preview_cols = ["window_start", *selected_cols]
preview_groups = {g: [c for c in cols if c in selected_cols] for g, cols in spec.groups.items()}
styled = styled_feature_table(matrix_df[preview_cols], preview_groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

st.subheader("PCA")
st.plotly_chart(pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim), width="stretch")

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
