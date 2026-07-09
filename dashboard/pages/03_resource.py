"""Resource state page: per-window resource workload → PCA → SOM or DBSCAN states → trajectory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.controls import log_signature, seed_choice, seed_multi, seed_widget
from core.dbscan import cluster_dbscan
from core.features.resource import build_features
from core.kmeans import cluster_kmeans
from core.loader import span_label
from core.pca import fit_pca
from core.schema import (
    RESOURCE_FEATURE_COLUMNS,
    RESOURCE_FEATURE_LABELS as KIND_LABELS,
    feature_availability,
)
from core.som import train_som
from core.state_attribution import state_id_shift
from core.transitions import find_transitions
from core.windows import (
    default_window_minutes,
    log_span_minutes,
    window_minute_choices,
    window_minute_label,
)
from viz.drift_scores import score_line
from viz.drift_signal import add_window_boundaries
from viz.som_grid import som_heatmap
from viz.tables import styled_feature_table
from viz.trajectory import add_transition_markers, pca_variance_plot, state_timeline

st.set_page_config(page_title="Resource states", layout="wide")
st.title("3 — Resource states")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))
if "org:resource" not in log.columns:
    st.error("This log has no `org:resource` column — the resource page is disabled.")
    st.stop()

KIND_DESCRIPTIONS = {
    "events": "Number of events assigned to each resource in the calendar window.",
    "active": "Number of distinct cases touched by each resource in the window.",
    "duration": "Mean event duration in minutes for each resource in the window (requires `event:duration_min`).",
    "wait": "Computed for each resource r: mean difference between this event's and the previous case event's `time:timestamp`, in minutes, over r's events in the window — counted only for events whose previous event was executed by a different resource.",
    "activity_events": "Computed for each activity a and each resource r: the share of a's events in the window executed by r — a-events by r divided by all a-events in the window.",
    "ho": "Computed for each ordered resource pair r1→r2: the share of r1's within-case handovers in the window that go to r2 — handovers r1→r2 divided by all handovers from r1 in the window.",
}

CLUSTERING_LABELS = {
    "som": "SOM",
    "dbscan": "DBSCAN",
    "kmeans": "k-means",
}

GRID_TITLES = {
    "som": "SOM grid",
    "dbscan": "DBSCAN clusters",
    "kmeans": "k-means clusters",
}

INIT_LABELS = {
    "random": "Random samples",
    "pca": "PCA plane",
}

RESOURCE_SCHEMA_VERSION = "resource_window_v6"
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
all_resources = sorted(log["org:resource"].astype(str).dropna().unique().tolist())
all_activities = sorted(log["concept:name"].astype(str).dropna().unique().tolist())

available_kinds, disabled_kinds = feature_availability(log.columns, RESOURCE_FEATURE_COLUMNS)

run_cfg = st.session_state.get("resource_run_config") or {}
default_kinds = run_cfg.get("kinds", available_kinds)
for kind in KIND_LABELS:
    seed_widget(f"resource_kind_{kind}", kind in default_kinds)
    if kind in disabled_kinds:
        st.session_state[f"resource_kind_{kind}"] = False
seed_multi("resource_res_sel", run_cfg.get("resources", all_resources), all_resources)
seed_multi("resource_act_sel", run_cfg.get("activities", all_activities), all_activities)
seed_widget("resource_pca_k_sel", int(run_cfg.get("pca_k", 0)))
seed_choice("resource_cluster_sel", run_cfg.get("clustering", "som"), tuple(CLUSTERING_LABELS))
seed_choice("resource_init_sel", run_cfg.get("som_init", "random"), tuple(INIT_LABELS))
seed_widget("resource_eps_sel", float(run_cfg.get("eps", 0.5)))
seed_widget("resource_minpts_sel", int(run_cfg.get("min_samples", 5)))
seed_widget("resource_kmeans_k_sel", int(run_cfg.get("kmeans_k", 6)))
grid_default = tuple(run_cfg.get("grid", (2, 3)))
seed_widget("resource_grid_h_sel", int(grid_default[0]))
seed_widget("resource_grid_w_sel", int(grid_default[1]))
seed_widget(
    "resource_W_sel",
    int(run_cfg.get("window_minutes", default_window_minutes(log_span_minutes(log)))),
)

with st.sidebar:
    st.header("Controls")
    with st.form("resource_pipeline_controls"):
        st.subheader("Features for state clustering")
        st.markdown("Feature kinds")
        picked_kinds = [
            kind
            for kind, label in KIND_LABELS.items()
            if st.checkbox(label, key=f"resource_kind_{kind}", disabled=kind in disabled_kinds)
        ]
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
                st.markdown(f"**{label}:** {KIND_DESCRIPTIONS[kind]}")
        st.subheader("PCA")
        pca_k = st.number_input(
            "PCA components (0 = auto/elbow)",
            min_value=0,
            max_value=20,
            step=1,
            key="resource_pca_k_sel",
        )
        st.subheader("Clustering")
        clustering = st.radio(
            "Method",
            options=tuple(CLUSTERING_LABELS),
            key="resource_cluster_sel",
            format_func=lambda m: CLUSTERING_LABELS[m],
        )
        col_grid_h, col_grid_w = st.columns(2)
        grid_h = col_grid_h.number_input(
            "SOM grid height", min_value=1, max_value=50, step=1, key="resource_grid_h_sel"
        )
        grid_w = col_grid_w.number_input(
            "SOM grid width", min_value=1, max_value=50, step=1, key="resource_grid_w_sel"
        )
        som_init = st.radio(
            "SOM init",
            options=tuple(INIT_LABELS),
            key="resource_init_sel",
            format_func=lambda i: INIT_LABELS[i],
        )
        eps = st.number_input(
            "DBSCAN eps",
            min_value=0.05,
            max_value=100.0,
            step=0.05,
            key="resource_eps_sel",
        )
        min_samples = st.number_input(
            "DBSCAN min samples",
            min_value=1,
            max_value=1000,
            step=1,
            key="resource_minpts_sel",
        )
        kmeans_k = st.number_input(
            "k-means clusters",
            min_value=2,
            max_value=25,
            step=1,
            key="resource_kmeans_k_sel",
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
    with st.spinner("Building features, fitting PCA, clustering states…"):
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
        if clustering == "dbscan":
            som = cluster_dbscan(pca.transformed, eps=float(eps), min_samples=int(min_samples))
        elif clustering == "kmeans":
            som = cluster_kmeans(pca.transformed, n_clusters=int(kmeans_k))
        else:
            som = train_som(
                pca.transformed, grid_h=grid_h, grid_w=grid_w, annotations=None, init=som_init
            )
        matrix_df = matrix_df.assign(state_id=som.state_ids)
    st.session_state["resource_matrix"] = matrix_df
    st.session_state["resource_spec"] = spec
    st.session_state["resource_pca"] = pca
    st.session_state["resource_som"] = som
    st.session_state["resource_selected_cols"] = selected_cols
    st.session_state["resource_log_signature"] = current_log_signature
    st.session_state["resource_run_config"] = {
        "grid": (grid_h, grid_w),
        "clustering": clustering,
        "som_init": som_init,
        "eps": float(eps),
        "min_samples": int(min_samples),
        "kmeans_k": int(kmeans_k),
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
    ran_clustering = st.session_state["resource_run_config"].get("clustering", "som")
    st.subheader(GRID_TITLES.get(ran_clustering, "States"))
    grid_view = st.radio(
        "Grid view", ("State colors", "Frequency"),
        horizontal=True, label_visibility="collapsed", key="resource_grid_style",
    )
    st.plotly_chart(
        som_heatmap(
            som.grid_h, som.grid_w, som.cell_counts, som.cell_labels,
            title="Window states", dominants=som.cell_dominant,
            monochrome=grid_view == "Frequency",
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
        cell_dominant=som.cell_dominant, window_ticks=True,
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

st.subheader("Drift signal")
st.caption(
    "Rolling KL divergence of the recent state mix against the full-log "
    "baseline — a sustained shift in the trajectory pushes the score up."
)
shift = state_id_shift(
    matrix_df[["window_start", "state_id"]], n_states=som.grid_h * som.grid_w
)
shift_fig = score_line(shift, "score", title="Rolling KL(state mix || baseline)")
add_window_boundaries(
    shift_fig, shift["window_start"], y_max=max(float(shift["score"].max()), 1e-9)
)
st.plotly_chart(shift_fig, width="stretch")
