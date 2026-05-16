"""Intra-case SOM page: window features → PCA → SOM cells → per-case trajectory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.features.intra_case import build_features
from core.pca import fit_pca
from core.som import train_som
from viz.som_grid import som_heatmap
from viz.tables import styled_feature_table
from viz.trajectory import pca_variance_plot, state_timeline

st.set_page_config(page_title="Intra-case SOM", layout="wide")
st.title("2 — Intra-case SOM")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
numeric_attrs = tuple(st.session_state.get("case_numeric_attrs", []))
categorical_attrs = tuple(st.session_state.get("case_categorical_attrs", []))

GROUP_LABELS = {
    "activity": "Activity one-hot (windowed)",
    "delta": "Δt gaps",
    "elapsed": "Elapsed case time",
    "case_attr": "Case attributes",
}

with st.sidebar:
    st.header("Controls")
    window = st.slider("Window size w (events)", min_value=1, max_value=10, value=3)
    grid_label = st.selectbox("SOM grid", ["2×2", "3×3", "4×4"], index=1)
    grid_h = grid_w = int(grid_label.split("×")[0])

feat, spec = build_features(
    log, window=window, numeric_attrs=numeric_attrs, categorical_attrs=categorical_attrs
)

available_groups = [g for g in GROUP_LABELS if spec.groups.get(g)]
with st.sidebar:
    st.subheader("Features for state clustering")
    picked = st.multiselect(
        "Include groups",
        options=available_groups,
        default=available_groups,
        format_func=lambda g: GROUP_LABELS[g],
    )
selected_cols = [c for g in picked for c in spec.groups[g]]
if not selected_cols:
    st.warning("Select at least one feature group to train the SOM.")
    st.stop()

st.subheader("Feature matrix")
st.caption(
    f"{len(feat):,} events × {len(selected_cols)} feature columns "
    f"(window={spec.window}, |A|={len(spec.activities)})"
)
preview_cols = ["case_id", "activity", "timestamp", *selected_cols]
preview_groups = {g: [c for c in cols if c in selected_cols] for g, cols in spec.groups.items()}
styled = styled_feature_table(feat[preview_cols], preview_groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

matrix = feat[selected_cols].to_numpy()
pca = fit_pca(matrix)

st.subheader("PCA")
st.plotly_chart(pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim), width="stretch")

som = train_som(
    pca.transformed,
    grid_h=grid_h,
    grid_w=grid_w,
    annotations=tuple(feat["activity"].astype(str).tolist()),
)
feat = feat.assign(state_id=som.state_ids)
st.session_state["intra_feat"] = feat
st.session_state["intra_spec"] = spec
st.session_state["intra_som"] = som

col_l, col_r = st.columns([1, 1])
with col_l:
    st.subheader("SOM grid")
    st.plotly_chart(
        som_heatmap(som.grid_h, som.grid_w, som.cell_counts, som.cell_labels, title="States by dominant last activity"),
        width="stretch",
    )
with col_r:
    st.subheader("Case trajectory")
    case_ids = feat["case_id"].drop_duplicates().tolist()
    chosen = st.selectbox("Case", case_ids, index=0)
    sub = feat[feat["case_id"] == chosen].reset_index(drop=True)
    st.plotly_chart(
        state_timeline(sub["timestamp"], sub["state_id"].to_numpy(), som.cell_labels, title=f"Case {chosen}"),
        width="stretch",
    )
