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
    window_minutes = st.selectbox("Window W (minutes)", options, index=options.index(default_W))
    st.session_state["window_minutes"] = window_minutes

matrix_df, spec = build_features(log, window_minutes=window_minutes)

KIND_LABELS = {
    "events": "Events per resource",
    "active": "Active cases per resource",
    "wait": "Mean wait into resource",
    "ho": "Handover counts",
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

selected_cols = [
    c for k in picked_kinds for c in kind_groups[k]
    if k == "ho" or any(c.endswith(f":{r}") for r in picked_resources)
]
if "ho" in picked_kinds:
    selected_cols = [
        c for c in selected_cols
        if not c.startswith("ho:")
        or any(f"ho:{a}→{b}" == c for a in picked_resources for b in picked_resources if a != b)
    ]
if not selected_cols:
    st.warning("Select at least one feature kind and one resource.")
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
pca = fit_pca(mat)
st.subheader("PCA")
st.plotly_chart(pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim), width="stretch")

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
        width="stretch",
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
        width="stretch",
    )
