"""Joint drift signal: stack intra-case state fractions with resource + inter-case state indices."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.drift import intra_state_distribution, joint_signal
from viz.drift_signal import add_window_boundaries, stacked_area_intra, state_index_line

st.set_page_config(page_title="Joint drift signal", layout="wide")
st.title("5 — Joint drift signal")

required = ["intra_feat", "intra_som"]
missing = [k for k in required if k not in st.session_state]
if missing:
    st.warning(
        "Visit the **Intra-case SOM** page first (and Resource / Inter-case pages if their logs apply)."
    )
    st.stop()

intra_feat: pd.DataFrame = st.session_state["intra_feat"]
intra_som = st.session_state["intra_som"]
window_minutes = int(st.session_state.get("window_minutes", 60))

n_states = intra_som.grid_h * intra_som.grid_w
intra_dist = intra_state_distribution(intra_feat, n_states=n_states, window_minutes=window_minutes)
intra_cols = [f"intra_S{i}" for i in range(n_states)]

resource_states = None
if "resource_matrix" in st.session_state:
    rm = st.session_state["resource_matrix"]
    resource_states = rm[["window_start", "state_id"]].rename(columns={"state_id": "resource_state"})

inter_states = None
if "inter_matrix" in st.session_state:
    im = st.session_state["inter_matrix"]
    inter_states = im[["window_start", "state_id"]].rename(columns={"state_id": "inter_state"})

joint = joint_signal(intra_dist, resource_states, inter_states)

st.subheader("Intra-case state fractions over time")
intra_fig = stacked_area_intra(joint, intra_cols, intra_som.cell_labels)
add_window_boundaries(intra_fig, joint["window_start"])
st.plotly_chart(intra_fig, width="stretch")

st.subheader("Resource state")
if resource_states is None:
    st.info("Open the **Resource SOM** page to populate this trace.")
else:
    labels = st.session_state["resource_som"].cell_labels
    res_fig = state_index_line(joint, "resource_state", labels, title=f"W={window_minutes} min")
    n_res = st.session_state["resource_som"].grid_h * st.session_state["resource_som"].grid_w
    add_window_boundaries(res_fig, joint["window_start"], y_min=-0.5, y_max=n_res - 0.5)
    st.plotly_chart(res_fig, width="stretch")

st.subheader("Inter-case state")
if inter_states is None:
    st.info("Open the **Inter-case SOM** page to populate this trace.")
else:
    labels = st.session_state["inter_som"].cell_labels
    inter_fig = state_index_line(joint, "inter_state", labels, title=f"W={window_minutes} min")
    n_inter = st.session_state["inter_som"].grid_h * st.session_state["inter_som"].grid_w
    add_window_boundaries(inter_fig, joint["window_start"], y_min=-0.5, y_max=n_inter - 0.5)
    st.plotly_chart(inter_fig, width="stretch")

st.caption(
    "A sustained shift in band proportions indicates concept drift. "
    "Compare the intra-case distribution before and after any visible change point."
)
