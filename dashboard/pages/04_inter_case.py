"""Inter-case SOM page: system-level features → optional PCA → SOM cells → trajectory."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from core.controls import log_signature, seed_choice, seed_multi, seed_widget
from core.features.inter_case import build_features, describe_cells
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

st.set_page_config(page_title="Inter-case SOM", layout="wide")
st.title("4 — Inter-case SOM")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))

FEATURE_LABELS = {
    "active_cases": "Active cases",
    "new_arrivals": "New arrivals",
    "completions": "Completions",
    "total_events": "Total events",
    "mean_delta_t": "Mean Δt (ln min)",
    "std_delta_t": "Std Δt (ln min)",
    "stalled_cases": "Stalled cases",
}
FEATURE_DESCRIPTIONS = {
    "active_cases": "Number of distinct cases with at least one event in the calendar window.",
    "new_arrivals": "Number of cases whose first event falls inside the window.",
    "completions": "Number of cases whose last event falls inside the window.",
    "total_events": "Total number of events in the window.",
    "mean_delta_t": "Mean gap between consecutive events inside the window, in ln-minutes.",
    "std_delta_t": "Standard deviation of the gaps between consecutive events inside the window, in ln-minutes.",
    "stalled_cases": "Number of cases whose last event is older than the stall threshold τ at the window end.",
}
INTER_FEATURES = list(FEATURE_LABELS)

INTER_SCHEMA_VERSION = "inter_window_v1"
INTER_RESULT_KEYS = (
    "inter_matrix",
    "inter_spec",
    "inter_pca",
    "inter_som",
    "inter_selected_cols",
    "inter_run_config",
    "inter_log_signature",
)

current_log_signature = log_signature(log, INTER_SCHEMA_VERSION)
if (
    st.session_state.get("inter_log_signature") is not None
    and st.session_state.get("inter_log_signature") != current_log_signature
):
    for key in (*INTER_RESULT_KEYS, "inter_W_sel"):
        st.session_state.pop(key, None)

run_cfg = st.session_state.get("inter_run_config") or {}
seed_multi("inter_feats_sel", run_cfg.get("features", INTER_FEATURES), INTER_FEATURES)
seed_widget("inter_stall_sel", int(run_cfg.get("stall_minutes", 60)))
seed_widget("inter_pca_k_sel", int(run_cfg.get("pca_k", 0)))
grid_default = tuple(run_cfg.get("grid", (2, 2)))
seed_choice(
    "inter_grid_sel",
    grid_default if grid_default in SOM_GRID_OPTIONS else (2, 2),
    SOM_GRID_OPTIONS,
)
seed_widget(
    "inter_W_sel",
    int(run_cfg.get("window_minutes", default_window_minutes(log_span_minutes(log)))),
)

with st.sidebar:
    st.header("Controls")
    with st.form("inter_pipeline_controls"):
        st.subheader("Features for state clustering")
        picked = st.multiselect(
            "Include features",
            options=INTER_FEATURES,
            key="inter_feats_sel",
            format_func=lambda c: FEATURE_LABELS.get(c, c),
        )
        stall = st.slider(
            "Stall threshold τ (minutes)",
            min_value=15,
            max_value=480,
            step=15,
            key="inter_stall_sel",
            help="A case counts as stalled when its last event is older than τ at the window end.",
        )
        with st.expander("Feature glossary"):
            for feature, label in FEATURE_LABELS.items():
                st.markdown(f"**{label}.** {FEATURE_DESCRIPTIONS[feature]}")
        st.subheader("PCA")
        pca_k = st.number_input(
            "PCA components (0 = auto/elbow)",
            min_value=0,
            max_value=20,
            step=1,
            key="inter_pca_k_sel",
        )
        st.subheader("SOM")
        grid_h, grid_w = st.selectbox(
            "SOM grid",
            SOM_GRID_OPTIONS,
            key="inter_grid_sel",
            format_func=som_grid_label,
        )
        st.subheader("Windows")
        window_minutes = st.selectbox(
            "Window W",
            window_minute_choices(st.session_state["inter_W_sel"]),
            key="inter_W_sel",
            format_func=window_minute_label,
        )
        run_pipeline = st.form_submit_button(
            "Run inter-case pipeline",
            width="stretch",
        )

if run_pipeline:
    if not picked:
        st.warning("Select at least one feature.")
        st.stop()
    with st.spinner("Building features, fitting PCA, training SOM…"):
        matrix_df, spec = build_features(
            log, window_minutes=int(window_minutes), stall_minutes=int(stall)
        )
        if len(matrix_df) < 2:
            st.warning(
                f"Window W is wider than the log span — only {len(matrix_df)} window emerged. "
                "Pick a smaller W to get a trajectory."
            )
            st.stop()
        selected_cols = list(picked)
        mat = matrix_df[selected_cols].to_numpy()
        pca = fit_pca(mat, force_k=int(pca_k) if pca_k else None)
        som = train_som(pca.transformed, grid_h=grid_h, grid_w=grid_w, annotations=None)
        centroids = np.zeros((som.grid_h * som.grid_w, len(selected_cols)))
        for cell_id in range(som.grid_h * som.grid_w):
            mask = som.state_ids == cell_id
            if mask.any():
                centroids[cell_id] = mat[mask].mean(axis=0)
        som = replace(som, cell_dominant=describe_cells(centroids, selected_cols))
        matrix_df = matrix_df.assign(state_id=som.state_ids)
    st.session_state["inter_matrix"] = matrix_df
    st.session_state["inter_spec"] = spec
    st.session_state["inter_pca"] = pca
    st.session_state["inter_som"] = som
    st.session_state["inter_selected_cols"] = selected_cols
    st.session_state["inter_log_signature"] = current_log_signature
    st.session_state["inter_run_config"] = {
        "grid": (grid_h, grid_w),
        "pca_k": int(pca_k),
        "window_minutes": int(window_minutes),
        "stall_minutes": int(stall),
        "features": list(picked),
    }

if (
    any(key not in st.session_state for key in INTER_RESULT_KEYS)
    or st.session_state.get("inter_log_signature") != current_log_signature
):
    st.info("Set the sidebar controls and run the inter-case pipeline.")
    st.stop()

matrix_df = st.session_state["inter_matrix"]
spec = st.session_state["inter_spec"]
pca = st.session_state["inter_pca"]
som = st.session_state["inter_som"]
selected_cols = st.session_state["inter_selected_cols"]

st.subheader("Feature matrix")
st.caption(
    f"{len(matrix_df):,} windows × {len(selected_cols)} features "
    f"(W={spec.window_minutes} min, τ={spec.stall_minutes} min)"
)
preview_cols = ["window_start", *selected_cols]
preview_groups = {g: [c for c in cols if c in selected_cols] for g, cols in spec.groups.items()}
styled = styled_feature_table(matrix_df[preview_cols], preview_groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

st.subheader("PCA")
st.plotly_chart(
    pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim),
    width="stretch",
)

col_l, col_r = st.columns([1, 1])
with col_l:
    st.subheader("SOM grid")
    st.plotly_chart(
        som_heatmap(
            som.grid_h, som.grid_w, som.cell_counts, som.cell_labels,
            title="System-level states", dominants=som.cell_dominant,
        ),
        width="stretch",
    )
with col_r:
    st.subheader("Inter-case state trajectory")
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
