"""Intra-case features: one row per event, describing its case up to that event."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import log_signature, seed_widget

FEATURES = {
    "act_freqs": "Activity frequencies",
    "df_counts": "Directly-follows counts",
    "act_set": "Activity set",
    "case_progress": "Case progress",
    "current_act": "Current activity",
    "past_acts": "Past activities",
}
GLOSSARY = {
    "Activity frequencies": "One column per activity — how often the activity occurred in the case so far, "
                            "divided by the number of events so far (the prefix).",
    "Directly-follows counts": "One column per observed directly-follows pair A→B — how often that transition "
                               "occurred in the prefix, divided by the number of transitions so far.",
    "Distinct activity set": "One binary column per activity — 1 if the activity has already occurred in the "
                             "case prefix, 0 otherwise.",
    "Case progress": "Position of the event within its case as a fraction of the total case length (last event = 1).",
    "Current activity": "One binary column per activity — 1 for the activity of this event.",
    "Past activities": "For each of the last n events of the same case, one binary column per activity — 1 for "
                       "that event's activity. Before the case has n predecessors, the missing slots stay 0.",
}

ui.keep_widgets()
log = ui.perspective_log("intra")

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
    ui.feature_glossary(GLOSSARY)

st.title("Features")
st.caption("One row per event, describing its case up to and including that event.")

if st.button("Compute features", type="primary", icon=":material/play_arrow:"):
    picked = [f for f in FEATURES if st.session_state[f"intra_sel_feature_{f}"]]
    if not picked:
        st.warning("Select at least one feature group in the sidebar.")
        st.stop()
    with st.spinner("Computing features…"):
        features = kairo.compute_features_intra(log, picked, int(st.session_state["intra_sel_window"]))
    ui.clear_from("intra", "features")
    st.session_state["intra_features"] = features
    st.session_state["intra_log_signature"] = log_signature(log, "intra")

features = st.session_state.get("intra_features")
if features is None:
    st.info("Pick the feature groups in the sidebar and compute them.")
    st.stop()

groups = ui.feature_groups(features)
ui.metrics_row([
    ("Events", f"{len(features):,}"),
    ("Cases", f"{features['case:concept:name'].nunique():,}"),
    ("Feature columns", f"{sum(map(len, groups.values())):,}"),
    ("Feature groups", f"{len(groups)}"),
])
st.caption("The first 30 rows, every feature group in its own tint.")
ui.show_features(features)
