"""Inter-case SOM page: system-level features → optional PCA → SOM cells → trajectory."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.features.inter_case import build_features, describe_cells
from core.pca import fit_pca
from core.som import train_som
from viz.som_grid import som_heatmap
from viz.tables import styled_feature_table
from viz.trajectory import pca_variance_plot, state_timeline

st.set_page_config(page_title="Inter-case SOM", layout="wide")
st.title("4 — Inter-case SOM")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]

with st.sidebar:
    st.header("Controls")
    default_W = int(st.session_state.get("window_minutes", 60))
    options = [30, 60, 120]
    if default_W not in options:
        options = sorted(options + [default_W])
    window_minutes = st.selectbox("Window W (minutes)", options, index=options.index(default_W))
    stall = st.slider("Stall threshold τ (minutes)", min_value=15, max_value=240, value=60, step=15)
    st.session_state["window_minutes"] = window_minutes

matrix_df, spec = build_features(log, window_minutes=window_minutes, stall_minutes=stall)

FEATURE_LABELS = {
    "active_cases": "Active cases",
    "new_arrivals": "New arrivals",
    "completions": "Completions",
    "total_events": "Total events",
    "mean_delta_t": "Mean Δt (ln min)",
    "std_delta_t": "Std Δt (ln min)",
    "stalled_cases": "Stalled cases",
}
with st.sidebar:
    st.subheader("Features for state clustering")
    selected_cols = st.multiselect(
        "Include features",
        options=spec.columns,
        default=spec.columns,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
    )
if not selected_cols:
    st.warning("Select at least one feature.")
    st.stop()

st.subheader("Feature matrix")
st.caption(
    f"{len(matrix_df):,} windows × {len(spec.columns)} features "
    f"(W={spec.window_minutes} min, τ={spec.stall_minutes} min); "
    f"{len(selected_cols)} feed the SOM."
)
styled = styled_feature_table(matrix_df, spec.groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

mat = matrix_df[selected_cols].to_numpy()
use_pca = mat.shape[0] > 20
st.subheader("PCA")
if use_pca:
    pca = fit_pca(mat)
    st.plotly_chart(
        pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim),
        width="stretch",
    )
    som_input = pca.transformed
else:
    st.info(f"PCA skipped — only {mat.shape[0]} windows, feeding directly to SOM.")
    som_input = mat

som = train_som(som_input, grid_h=2, grid_w=2, annotations=None)

centroids = np.zeros((som.grid_h * som.grid_w, len(selected_cols)))
for cell_id in range(som.grid_h * som.grid_w):
    mask = som.state_ids == cell_id
    if mask.any():
        centroids[cell_id] = mat[mask].mean(axis=0)
descriptive = describe_cells(centroids, selected_cols)
som = som.__class__(
    grid_h=som.grid_h, grid_w=som.grid_w,
    bmus=som.bmus, state_ids=som.state_ids,
    cell_labels=descriptive, cell_counts=som.cell_counts,
)
matrix_df = matrix_df.assign(state_id=som.state_ids)
st.session_state["inter_matrix"] = matrix_df
st.session_state["inter_spec"] = spec
st.session_state["inter_som"] = som

col_l, col_r = st.columns([1, 1])
with col_l:
    st.subheader("SOM grid")
    st.plotly_chart(
        som_heatmap(som.grid_h, som.grid_w, som.cell_counts, som.cell_labels, title="System-level states"),
        width="stretch",
    )
with col_r:
    st.subheader("Inter-case state trajectory")
    fig = state_timeline(
        matrix_df["window_start"], som.state_ids, som.cell_labels,
        title=f"State per {spec.window_minutes}-min window",
    )
    st.plotly_chart(fig, width="stretch")
    with st.expander(f"Window feature values (first {min(500, len(matrix_df))} rows)"):
        st.dataframe(matrix_df.head(500), width="stretch", height=240)
