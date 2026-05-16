"""Inter-case SOM page: system-level features → optional PCA → SOM cells → trajectory."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.features.inter_case import build_features, describe_cells
from core.loader import span_label
from core.pca import fit_pca
from core.som import train_som
from core.transitions import find_transitions
from core.windows import SOM_GRID_OPTIONS, som_grid_label, window_minute_choices, window_minute_label
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

with st.sidebar:
    st.header("Controls")
    default_W = int(st.session_state.get("inter_W", 60))
    options = window_minute_choices(default_W)
    window_minutes = st.selectbox(
        "Window W", options, index=options.index(default_W), format_func=window_minute_label
    )
    stall = st.slider("Stall threshold τ (minutes)", min_value=15, max_value=480, value=60, step=15)
    st.session_state["inter_W"] = window_minutes
    grid_default = st.session_state.get("inter_grid", (2, 2))
    if grid_default not in SOM_GRID_OPTIONS:
        grid_default = (2, 2)
    grid_h, grid_w = st.selectbox(
        "SOM grid", SOM_GRID_OPTIONS,
        index=SOM_GRID_OPTIONS.index(grid_default),
        format_func=som_grid_label,
    )
    st.session_state["inter_grid"] = (grid_h, grid_w)
    pca_k = st.number_input(
        "PCA components (0 = auto/elbow)",
        min_value=0, max_value=20, value=int(st.session_state.get("inter_pca_k", 0)), step=1,
    )
    st.session_state["inter_pca_k"] = pca_k

matrix_df, spec = build_features(log, window_minutes=window_minutes, stall_minutes=stall)
if len(matrix_df) < 2:
    st.warning(
        f"Window W is wider than the log span — only {len(matrix_df)} window emerged. "
        "Pick a smaller W to get a trajectory."
    )
    st.stop()

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
    f"{len(matrix_df):,} windows × {len(selected_cols)} features "
    f"(W={spec.window_minutes} min, τ={spec.stall_minutes} min)"
)
preview_cols = ["window_start", *selected_cols]
preview_groups = {g: [c for c in cols if c in selected_cols] for g, cols in spec.groups.items()}
styled = styled_feature_table(matrix_df[preview_cols], preview_groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

mat = matrix_df[selected_cols].to_numpy()
use_pca = mat.shape[0] > 20
st.subheader("PCA")
if use_pca:
    pca = fit_pca(mat, force_k=int(pca_k) if pca_k else None)
    st.plotly_chart(
        pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim),
        width="stretch",
    )
    som_input = pca.transformed
else:
    st.info(f"PCA skipped — only {mat.shape[0]} windows, feeding directly to SOM.")
    som_input = mat

som = train_som(som_input, grid_h=grid_h, grid_w=grid_w, annotations=None)

centroids = np.zeros((som.grid_h * som.grid_w, len(selected_cols)))
for cell_id in range(som.grid_h * som.grid_w):
    mask = som.state_ids == cell_id
    if mask.any():
        centroids[cell_id] = mat[mask].mean(axis=0)
descriptive = describe_cells(centroids, selected_cols)
som = som.__class__(
    grid_h=som.grid_h, grid_w=som.grid_w,
    bmus=som.bmus, state_ids=som.state_ids,
    cell_labels=som.cell_labels, cell_counts=som.cell_counts,
    cell_dominant=descriptive,
)
matrix_df = matrix_df.assign(state_id=som.state_ids)
st.session_state["inter_matrix"] = matrix_df
st.session_state["inter_spec"] = spec
st.session_state["inter_som"] = som

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
        add_transition_markers(fig, transitions["boundary"])
    st.plotly_chart(fig, width="stretch")
    with st.expander(f"Window feature values (first {min(500, len(matrix_df))} rows)"):
        st.dataframe(matrix_df.head(500), width="stretch", height=240)

st.subheader("Transitions")
if transitions.empty:
    st.caption("No state changes in this trajectory.")
else:
    st.caption(f"{len(transitions)} transitions detected.")
    st.dataframe(
        transitions[["boundary", "from", "to", "top_changes"]].head(500).rename(columns={"boundary": "at"}),
        width="stretch", height=min(420, 60 + 36 * len(transitions)),
    )
