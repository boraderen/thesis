"""Resource SOM page: per-window resource workload → PCA → SOM cells → trajectory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.features.resource import build_features
from core.pca import fit_pca
from core.som import train_som
from viz.som_grid import som_heatmap
from viz.tables import styled_feature_table
from viz.trajectory import pca_variance_plot, state_timeline

st.set_page_config(page_title="Resource SOM", layout="wide")
st.title("3 — Resource SOM")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
if "resource" not in log.columns:
    st.error("This log has no `resource` column — the resource SOM is disabled.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    default_W = int(st.session_state.get("window_minutes", 60))
    options = [30, 60, 120]
    if default_W not in options:
        options = sorted(options + [default_W])
    window_minutes = st.selectbox(
        "Window W (minutes)", options, index=options.index(default_W)
    )
    st.session_state["window_minutes"] = window_minutes

matrix_df, spec = build_features(log, window_minutes=window_minutes)
if spec.aggregated:
    st.warning(
        f"More than 8 distinct resources — aggregated to {len(spec.resources)} groups "
        "(via org:group or first-letter role)."
    )

st.subheader("Feature matrix")
st.caption(
    f"{len(matrix_df):,} windows × {len(spec.columns)} columns (W={spec.window_minutes} min, "
    f"{len(spec.resources)} resources)."
)
styled = styled_feature_table(matrix_df, spec.groups, max_rows=30)
st.dataframe(styled, use_container_width=True, height=380)

mat = matrix_df[spec.columns].to_numpy()
pca = fit_pca(mat)
st.subheader("PCA")
st.plotly_chart(pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim), use_container_width=True)

som = train_som(pca.transformed, grid_h=2, grid_w=3, annotations=None)
matrix_df = matrix_df.assign(state_id=som.state_ids)
st.session_state["resource_matrix"] = matrix_df
st.session_state["resource_spec"] = spec
st.session_state["resource_som"] = som

col_l, col_r = st.columns([1, 1])
with col_l:
    st.subheader("SOM grid")
    st.plotly_chart(
        som_heatmap(som.grid_h, som.grid_w, som.cell_counts, som.cell_labels, title="Window states"),
        use_container_width=True,
    )
with col_r:
    st.subheader("Resource state trajectory")
    st.plotly_chart(
        state_timeline(
            matrix_df["window_start"],
            som.state_ids,
            som.cell_labels,
            title=f"State per {spec.window_minutes}-min window",
        ),
        use_container_width=True,
    )
