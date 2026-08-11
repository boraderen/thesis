"""Inter-case state page: system-level features → PCA → SOM or DBSCAN states → trajectory."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from core.controls import log_signature, seed_choice, seed_widget
from core.dbscan import cluster_dbscan
from core.features.inter_case import attribute_features, build_features, describe_cells
from core.kmeans import cluster_kmeans
from core.loader import span_label
from core.pca import fit_pca
from core.schema import INTER_FEATURE_LABELS as FEATURE_LABELS
from core.som import train_som
from core.state_attribution import window_vector_shift
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

st.set_page_config(page_title="Inter-case states", layout="wide")
st.title("4 — Inter-case states")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))

FEATURE_DESCRIPTIONS = {
    "active_cases": "Number of distinct cases with at least one event in the calendar window.",
    "new_arrivals": "Number of cases whose first event falls inside the window.",
    "completions": "Number of cases whose last event falls inside the window.",
    "total_events": "Total number of events in the window.",
    "mean_delta_t": "Mean gap between consecutive events inside the window, in ln-minutes.",
    "std_delta_t": "Standard deviation of the gaps between consecutive events inside the window, in ln-minutes.",
    "stalled_cases": "Number of cases whose last event is older than the stall threshold τ at the window end.",
}


def attr_description(key: str, columns: list[str]) -> str:
    """Glossary line for one case-attribute feature."""
    kind, target = key.split(":", 1)
    if kind == "attr_mean":
        return f"Mean of {target} over the events in the window."
    if kind == "attr_std":
        return f"Standard deviation of {target} over the events in the window."
    return (
        f"One column per value of {target} ({len(columns)} in this log) — the share of the "
        "window's events carrying that value. The values come as a set, not one by one."
    )


# Features contributed by the case attributes mapped on the upload page.
numeric_attrs = tuple(st.session_state.get("case_numeric_attrs", []))
categorical_attrs = tuple(st.session_state.get("case_categorical_attrs", []))
ATTR_FEATURES = attribute_features(log, numeric_attrs, categorical_attrs)
FEATURE_LABELS = {**FEATURE_LABELS, **{k: label for k, (label, _) in ATTR_FEATURES.items()}}
FEATURE_DESCRIPTIONS = {
    **FEATURE_DESCRIPTIONS,
    **{k: attr_description(k, cols) for k, (_, cols) in ATTR_FEATURES.items()},
}
# A feature key maps to the matrix columns it selects — one each for the
# system-level features, several for a categorical attribute's value shares.
FEATURE_COLUMNS = {feature: [feature] for feature in FEATURE_LABELS}
FEATURE_COLUMNS.update({k: cols for k, (_, cols) in ATTR_FEATURES.items()})
INTER_FEATURES = list(FEATURE_LABELS)

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

attr_columns = sorted(c for _, cols in ATTR_FEATURES.values() for c in cols)
current_log_signature = log_signature(log, f"{INTER_SCHEMA_VERSION}:{attr_columns}")
if (
    st.session_state.get("inter_log_signature") is not None
    and st.session_state.get("inter_log_signature") != current_log_signature
):
    for key in (*INTER_RESULT_KEYS, "inter_W_sel"):
        st.session_state.pop(key, None)

run_cfg = st.session_state.get("inter_run_config") or {}
default_features = run_cfg.get("features", INTER_FEATURES)
for feature in INTER_FEATURES:
    seed_widget(f"inter_feat_{feature}", feature in default_features)
seed_widget("inter_stall_sel", int(run_cfg.get("stall_minutes", 60)))
seed_widget("inter_pca_k_sel", int(run_cfg.get("pca_k", 0)))
seed_choice("inter_cluster_sel", run_cfg.get("clustering", "som"), tuple(CLUSTERING_LABELS))
seed_choice("inter_init_sel", run_cfg.get("som_init", "random"), tuple(INIT_LABELS))
seed_widget("inter_eps_sel", float(run_cfg.get("eps", 0.5)))
seed_widget("inter_minpts_sel", int(run_cfg.get("min_samples", 5)))
seed_widget("inter_kmeans_k_sel", int(run_cfg.get("kmeans_k", 6)))
grid_default = tuple(run_cfg.get("grid", (2, 2)))
seed_widget("inter_grid_h_sel", int(grid_default[0]))
seed_widget("inter_grid_w_sel", int(grid_default[1]))
seed_widget(
    "inter_W_sel",
    int(run_cfg.get("window_minutes", default_window_minutes(log_span_minutes(log)))),
)

with st.sidebar:
    st.header("Controls")
    st.subheader("Select features")
    picked = [
        feature
        for feature in INTER_FEATURES
        if st.checkbox(FEATURE_LABELS.get(feature, feature), key=f"inter_feat_{feature}")
    ]
    with st.expander("Feature glossary"):
        for feature, label in FEATURE_LABELS.items():
            st.markdown(f"**{label}.** {FEATURE_DESCRIPTIONS[feature]}")
    stall = st.slider("Stall threshold τ (minutes)", min_value=15, max_value=480, step=15,
                      key="inter_stall_sel")

    st.subheader("Dimensionality reduction")
    st.number_input("PCA components (0 = auto)", min_value=0, max_value=20, step=1,
                    key="inter_pca_k_sel")

    st.subheader("Clustering")
    clustering = st.radio(
        "Method",
        options=tuple(CLUSTERING_LABELS),
        key="inter_cluster_sel",
        format_func=lambda m: CLUSTERING_LABELS[m],
    )
    if clustering == "som":
        col_grid_h, col_grid_w = st.columns(2)
        col_grid_h.number_input("SOM grid height", min_value=1, max_value=50, step=1,
                                key="inter_grid_h_sel")
        col_grid_w.number_input("SOM grid width", min_value=1, max_value=50, step=1,
                                key="inter_grid_w_sel")
        st.radio("SOM init", options=tuple(INIT_LABELS), key="inter_init_sel",
                 format_func=lambda i: INIT_LABELS[i])
    elif clustering == "dbscan":
        st.number_input("DBSCAN eps", min_value=0.05, max_value=100.0, step=0.05,
                        key="inter_eps_sel")
        st.number_input("DBSCAN min samples", min_value=1, max_value=1000, step=1,
                        key="inter_minpts_sel")
    else:
        st.number_input("k-means clusters", min_value=2, max_value=25, step=1,
                        key="inter_kmeans_k_sel")

    st.subheader("Windows")
    window_minutes = st.selectbox(
        "Window W",
        window_minute_choices(st.session_state["inter_W_sel"]),
        key="inter_W_sel",
        format_func=window_minute_label,
    )
    run_pipeline = st.button("Run inter-case pipeline", width="stretch", type="primary")

# The parameters of the unselected clustering methods are not rendered, so they
# are read from their seeded slots rather than from a widget's return value.
pca_k = int(st.session_state["inter_pca_k_sel"])
grid_h = int(st.session_state["inter_grid_h_sel"])
grid_w = int(st.session_state["inter_grid_w_sel"])
som_init = st.session_state["inter_init_sel"]
eps = float(st.session_state["inter_eps_sel"])
min_samples = int(st.session_state["inter_minpts_sel"])
kmeans_k = int(st.session_state["inter_kmeans_k_sel"])

if run_pipeline:
    if not picked:
        st.warning("Select at least one feature.")
        st.stop()
    with st.spinner("Building features, fitting PCA, clustering states…"):
        matrix_df, spec = build_features(
            log,
            window_minutes=int(window_minutes),
            stall_minutes=int(stall),
            numeric_attrs=numeric_attrs,
            categorical_attrs=categorical_attrs,
        )
        if len(matrix_df) < 2:
            st.warning(
                f"Window W is wider than the log span — only {len(matrix_df)} window emerged. "
                "Pick a smaller W to get a trajectory."
            )
            st.stop()
        selected_cols = [col for feature in picked for col in FEATURE_COLUMNS[feature]]
        mat = matrix_df[selected_cols].to_numpy()
        pca = fit_pca(mat, force_k=int(pca_k) if pca_k else None)
        if clustering == "dbscan":
            som = cluster_dbscan(pca.transformed, eps=float(eps), min_samples=int(min_samples))
        elif clustering == "kmeans":
            som = cluster_kmeans(pca.transformed, n_clusters=int(kmeans_k))
        else:
            som = train_som(
                pca.transformed, grid_h=grid_h, grid_w=grid_w, annotations=None, init=som_init
            )
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
        "clustering": clustering,
        "som_init": som_init,
        "eps": float(eps),
        "min_samples": int(min_samples),
        "kmeans_k": int(kmeans_k),
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
    ran_clustering = st.session_state["inter_run_config"].get("clustering", "som")
    st.subheader(GRID_TITLES.get(ran_clustering, "States"))
    grid_view = st.radio(
        "Grid view", ("State colors", "Frequency"),
        horizontal=True, label_visibility="collapsed", key="inter_grid_style",
    )
    st.plotly_chart(
        som_heatmap(
            som.grid_h, som.grid_w, som.cell_counts, som.cell_labels,
            title="System-level states", dominants=som.cell_dominant,
            monochrome=grid_view == "Frequency",
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
    "Distance between each window's compressed state vector and the previous "
    "window's — spikes mark the windows where the system picture changed."
)
shift = window_vector_shift(matrix_df["window_start"], pca.transformed)
shift_fig = score_line(shift, "score", title="‖window i − window i−1‖")
add_window_boundaries(
    shift_fig, shift["window_start"], y_max=max(float(shift["score"].max()), 1e-9)
)
st.plotly_chart(shift_fig, width="stretch")
