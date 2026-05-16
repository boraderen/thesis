"""Joint drift signal: stack intra-case state fractions with resource + inter-case state indices."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.drift import intra_state_distribution
from core.drift_scores import per_window_scores
from core.features.inter_case import build_features as build_inter
from core.features.resource import build_features as build_resource
from core.loader import span_label
from core.pca import fit_pca
from core.som import train_som
from core.windows import window_minute_choices, window_minute_label
from viz.drift_scores import score_line
from viz.drift_signal import add_window_boundaries, stacked_area_intra, state_index_line

st.set_page_config(page_title="Joint drift signal", layout="wide")
st.title("5 — Joint drift signal")

if "intra_feat" not in st.session_state or "intra_som" not in st.session_state:
    st.warning("Visit the **Intra-case SOM** page first to populate the per-event state trajectory.")
    st.stop()


def _select_W(label: str, key: str) -> int:
    """Render a W selectbox bound to a session_state key."""
    default = int(st.session_state.get(key, 60))
    options = window_minute_choices(default)
    chosen = st.selectbox(label, options, index=options.index(default), format_func=window_minute_label, key=f"{key}_sel")
    st.session_state[key] = chosen
    return chosen


log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))
intra_feat: pd.DataFrame = st.session_state["intra_feat"]
intra_som = st.session_state["intra_som"]
has_resource = "resource" in log.columns

with st.sidebar:
    st.header("Windows")
    intra_W = _select_W("Intra distribution W", "intra_distribution_W")
    resource_W = _select_W("Resource W", "resource_W") if has_resource else None
    inter_W = _select_W("Inter-case W", "inter_W")
    st.caption(
        "Each W is independent. Changing it here also updates the corresponding "
        "single-SOM page on its next load."
    )

# Intra-case distribution (cached, so cheap to recompute)
n_intra = intra_som.grid_h * intra_som.grid_w
intra_dist = intra_state_distribution(intra_feat, n_states=n_intra, window_minutes=intra_W)
intra_cols = [f"intra_S{i}" for i in range(n_intra)]

# Resource pipeline (re-run locally so this page is self-consistent)
resource_pack = None
if has_resource:
    res_grid = st.session_state.get("resource_grid", (2, 3))
    res_pca_k = int(st.session_state.get("resource_pca_k", 0))
    res_matrix, res_spec = build_resource(log, window_minutes=resource_W)
    if len(res_matrix) >= 2:
        res_pca = fit_pca(res_matrix[res_spec.columns].to_numpy(), force_k=res_pca_k if res_pca_k else None)
        res_som = train_som(res_pca.transformed, grid_h=res_grid[0], grid_w=res_grid[1])
        res_states = res_matrix.assign(state_id=res_som.state_ids)[["window_start", "state_id"]].rename(
            columns={"state_id": "resource_state"}
        )
        resource_pack = (res_states, res_som, res_grid)

# Inter-case pipeline (always available)
inter_grid = st.session_state.get("inter_grid", (2, 2))
inter_pca_k = int(st.session_state.get("inter_pca_k", 0))
inter_matrix, inter_spec = build_inter(log, window_minutes=inter_W, stall_minutes=60)
if len(inter_matrix) >= 2:
    inter_mat = inter_matrix[inter_spec.columns].to_numpy()
    inter_pca = fit_pca(inter_mat, force_k=inter_pca_k if inter_pca_k else None)
    inter_som = train_som(inter_pca.transformed, grid_h=inter_grid[0], grid_w=inter_grid[1])
    inter_states_df = inter_matrix.assign(state_id=inter_som.state_ids)[["window_start", "state_id"]].rename(
        columns={"state_id": "inter_state"}
    )
else:
    inter_som = None
    inter_states_df = None

st.subheader("Per-perspective drift attribution")
st.caption(
    "Each line is the divergence of one perspective's per-window distribution from "
    "the full-log baseline. Whichever line spikes is the perspective that drifted; "
    "the others should stay near zero. All four scores share the intra-case window "
    f"({window_minute_label(intra_W)})."
)
score_numeric = tuple(st.session_state.get("case_numeric_attrs", []))
score_categorical = tuple(st.session_state.get("case_categorical_attrs", []))
scores = per_window_scores(log, intra_W, numeric_attrs=score_numeric, categorical_attrs=score_categorical)
st.plotly_chart(
    score_line(scores, "cf_score",
               title="Control-flow — KL(activity dist || baseline)"),
    width="stretch",
)
st.plotly_chart(
    score_line(scores, "resource_score",
               title="Resource — mean KL(per-activity resource dist || baseline)"),
    width="stretch",
)
st.plotly_chart(
    score_line(scores, "inter_score",
               title="Inter-case — |z(events in window)| vs. baseline mean/std"),
    width="stretch",
)
st.plotly_chart(
    score_line(scores, "data_score",
               title="Data attributes — mean(|z(numeric)| + KL(categorical))"),
    width="stretch",
)

st.subheader(f"Intra-case state fractions over time (W = {window_minute_label(intra_W)})")
intra_fig = stacked_area_intra(intra_dist, intra_cols, intra_som.cell_labels)
add_window_boundaries(intra_fig, intra_dist["window_start"])
st.plotly_chart(intra_fig, width="stretch")

st.subheader("Resource state" + (f" (W = {window_minute_label(resource_W)})" if has_resource else ""))
if not has_resource:
    st.info("Log has no resource column.")
elif resource_pack is None:
    st.info("Window is wider than the log span — pick a smaller resource W.")
else:
    res_states, res_som, res_grid = resource_pack
    res_fig = state_index_line(res_states, "resource_state", res_som.cell_labels, title=window_minute_label(resource_W))
    n_res = res_grid[0] * res_grid[1]
    add_window_boundaries(res_fig, res_states["window_start"], y_min=-0.5, y_max=n_res - 0.5)
    st.plotly_chart(res_fig, width="stretch")

st.subheader(f"Inter-case state (W = {window_minute_label(inter_W)})")
if inter_states_df is None:
    st.info("Window is wider than the log span — pick a smaller inter-case W.")
else:
    inter_fig = state_index_line(inter_states_df, "inter_state", inter_som.cell_labels, title=window_minute_label(inter_W))
    n_inter = inter_grid[0] * inter_grid[1]
    add_window_boundaries(inter_fig, inter_states_df["window_start"], y_min=-0.5, y_max=n_inter - 0.5)
    st.plotly_chart(inter_fig, width="stretch")

st.caption(
    "A sustained shift in band proportions indicates concept drift. "
    "Each W can be tuned independently — coarser windows smooth the signal; finer ones expose short events. "
    "Grid size and PCA components are inherited from the per-stream pages."
)
