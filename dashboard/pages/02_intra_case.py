"""Intra-case SOM page: window features → PCA → SOM cells → per-case trajectory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.drift import intra_state_distribution
from core.features.intra_case import build_features
from core.pca import fit_pca
from core.som import train_som
from core.transitions import find_transitions
from core.windows import SOM_GRID_OPTIONS, som_grid_label, window_minute_choices, window_minute_label
from viz.drift_signal import add_window_boundaries, stacked_area_intra
from viz.som_grid import som_heatmap
from viz.tables import styled_feature_table
from viz.trajectory import add_transition_markers, pca_variance_plot, state_timeline

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
    grid_default = st.session_state.get("intra_grid", (3, 3))
    if grid_default not in SOM_GRID_OPTIONS:
        grid_default = (3, 3)
    grid_h, grid_w = st.selectbox(
        "SOM grid", SOM_GRID_OPTIONS,
        index=SOM_GRID_OPTIONS.index(grid_default),
        format_func=som_grid_label,
    )
    st.session_state["intra_grid"] = (grid_h, grid_w)
    pca_k = st.number_input(
        "PCA components (0 = auto/elbow)",
        min_value=0, max_value=20, value=int(st.session_state.get("intra_pca_k", 0)), step=1,
    )
    st.session_state["intra_pca_k"] = pca_k
    default_W = int(st.session_state.get("intra_distribution_W", 60))
    win_options = window_minute_choices(default_W)
    distribution_W = st.selectbox(
        "Distribution window W",
        win_options,
        index=win_options.index(default_W),
        format_func=window_minute_label,
        help="Calendar window used to aggregate per-event states into frequency bands.",
    )
    st.session_state["intra_distribution_W"] = distribution_W

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
pca = fit_pca(matrix, force_k=int(pca_k) if pca_k else None)

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
        som_heatmap(
            som.grid_h, som.grid_w, som.cell_counts, som.cell_labels,
            title="States (hover for dominant last activity)",
            dominants=som.cell_dominant,
        ),
        width="stretch",
    )
with col_r:
    st.subheader("Case trajectory")
    case_ids = feat["case_id"].drop_duplicates().tolist()
    chosen = st.selectbox("Case", case_ids, index=0)
    sub = feat[feat["case_id"] == chosen].reset_index(drop=True)
    transitions = find_transitions(
        sub["timestamp"], sub["state_id"].to_numpy(), som.cell_labels, sub[selected_cols]
    )
    fig = state_timeline(
        sub["timestamp"], sub["state_id"].to_numpy(), som.cell_labels,
        title=f"Case {chosen}", cell_dominant=som.cell_dominant, xgap=0,
    )
    if not transitions.empty:
        add_transition_markers(fig, transitions["boundary"])
    st.plotly_chart(fig, width="stretch")

st.subheader("Transitions")
if transitions.empty:
    st.caption("No state changes in this case.")
else:
    st.caption(f"{len(transitions)} transitions in case {chosen}.")
    st.dataframe(
        transitions[["boundary", "from", "to", "top_changes"]].rename(columns={"boundary": "at"}),
        width="stretch", height=min(420, 60 + 36 * len(transitions)),
    )

st.subheader("State frequency distribution over time")
n_states = som.grid_h * som.grid_w
intra_dist = intra_state_distribution(feat, n_states=n_states, window_minutes=distribution_W)
st.caption(
    f"Per {window_minute_label(distribution_W)} window — what fraction of events landed in each state. "
    f"{len(intra_dist):,} windows across all {feat['case_id'].nunique():,} cases."
)
intra_cols = [f"intra_S{i}" for i in range(n_states)]
freq_fig = stacked_area_intra(intra_dist, intra_cols, som.cell_labels)
add_window_boundaries(freq_fig, intra_dist["window_start"])
st.plotly_chart(freq_fig, width="stretch")
