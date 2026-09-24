"""Intra-case features: one row per event, describing its case up to that event."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import log_signature, seed_widget
from kairo.analysis import META
from tables import styled_feature_table

FEATURES = {
    "act_freqs": "Activity frequencies",
    "df_counts": "Directly-follows counts",
    "act_set": "Activity set",
    "case_progress": "Case progress",
    "current_act": "Current activity",
    "past_acts": "Past activities",
}

ui.keep_widgets()
log = ui.intra_log()

for feature in FEATURES:
    seed_widget(f"intra_sel_feature_{feature}", True)
seed_widget("intra_sel_window", 3)

with st.sidebar:
    st.header("Features")
    for feature, label in FEATURES.items():
        st.checkbox(label, key=f"intra_sel_feature_{feature}")
    if st.session_state["intra_sel_feature_past_acts"]:
        st.number_input("Past activities window", min_value=1, max_value=20, step=1,
                        key="intra_sel_window")

st.title("Features")
st.caption("One row per event, describing its case up to and including that event.")

if st.button("Compute features", type="primary", icon=":material/play_arrow:"):
    picked = [f for f in FEATURES if st.session_state[f"intra_sel_feature_{f}"]]
    if not picked:
        st.warning("Select at least one feature group in the sidebar.")
        st.stop()
    with st.spinner("Computing features…"):
        features = kairo.compute_features_intra(log, picked, int(st.session_state["intra_sel_window"]))
    ui.clear_from("features")
    st.session_state["intra_features"] = features
    st.session_state["intra_log_signature"] = log_signature(log, "intra")

features = st.session_state.get("intra_features")
if features is None:
    st.info("Pick the feature groups in the sidebar and compute them.")
    st.stop()

values = features.drop(columns=META)
groups = values.columns.str.split(":").str[0]
columns_per_group = {group: list(values.columns[groups == group]) for group in dict.fromkeys(groups)}

ui.metrics_row([
    ("Events", f"{len(features):,}"),
    ("Cases", f"{features['case:concept:name'].nunique():,}"),
    ("Feature columns", f"{values.shape[1]:,}"),
    ("Feature groups", f"{len(columns_per_group)}"),
])
st.caption("The first 30 rows, every feature group in its own tint.")
st.dataframe(styled_feature_table(features, columns_per_group), width="stretch", height=420)
