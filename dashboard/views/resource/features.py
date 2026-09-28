"""Resource features: one row per calendar window, describing how the resources worked in it."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import log_signature, seed_multi, seed_widget

FEATURES = {
    "res_events": "Events per resource",
    "res_cases": "Active cases per resource",
    "res_durations": "Mean event duration per resource",
    "res_waits": "Mean wait into resource",
    "act_res_shares": "Activity-resource event shares",
    "handover_shares": "Handover shares",
}
GLOSSARY = {
    "Events per resource": "Number of events assigned to each resource in the calendar window.",
    "Active cases per resource": "Number of distinct cases touched by each resource in the window.",
    "Mean event duration per resource": "Mean event duration in minutes for each resource in the window "
                                        "(requires a mapped event duration column).",
    "Mean wait into resource": "Computed for each resource r: mean difference between this event's and the "
                               "previous case event's timestamp, in minutes, over r's events in the window — "
                               "counted only for events whose previous event was executed by a different resource.",
    "Activity-resource event shares": "Computed for each activity a and each resource r: the share of a's events "
                                      "in the window executed by r — a-events by r divided by all a-events in the window.",
    "Handover shares": "Computed for each ordered resource pair r1→r2: the share of r1's within-case handovers in "
                       "the window that go to r2 — handovers r1→r2 divided by all handovers from r1 in the window.",
}

ui.keep_widgets()
log = ui.perspective_log("resource")
resources = sorted(log["org:resource"].dropna().unique()) if "org:resource" in log.columns else []

for feature in FEATURES:
    seed_widget(f"resource_sel_feature_{feature}", True)
seed_widget("resource_sel_window", "1D")
seed_multi("resource_sel_resources", [], resources)

with st.sidebar:
    st.header("Features")
    for feature, label in FEATURES.items():
        st.checkbox(label, key=f"resource_sel_feature_{feature}")
    st.selectbox("Window", list(ui.WINDOWS), format_func=ui.WINDOWS.get, key="resource_sel_window")
    st.multiselect("Resources", resources, key="resource_sel_resources", placeholder="All resources",
                   help="The resources that get feature columns. Left empty, all of them.")
    ui.feature_glossary(GLOSSARY)

st.title("Features")
st.caption("One row per calendar window, describing how the resources worked in it.")

if st.button("Compute features", type="primary", icon=":material/play_arrow:"):
    picked = [f for f in FEATURES if st.session_state[f"resource_sel_feature_{f}"]]
    if not picked:
        st.warning("Select at least one feature group in the sidebar.")
        st.stop()
    with st.spinner("Computing features…"):
        try:
            features = kairo.compute_features_resource(log, st.session_state["resource_sel_window"], picked,
                                                       st.session_state["resource_sel_resources"] or None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    ui.clear_from("resource", "features")
    st.session_state["resource_features"] = features
    st.session_state["resource_log_signature"] = log_signature(log, "resource")

features = st.session_state.get("resource_features")
if features is None:
    st.info("Pick the feature groups, the window and the resources in the sidebar and compute them.")
    st.stop()

groups = ui.feature_groups(features)
ui.metrics_row([
    ("Windows", f"{len(features):,}"),
    ("Feature columns", f"{sum(map(len, groups.values())):,}"),
    ("Feature groups", f"{len(groups)}"),
])
st.caption("The first 30 windows, every feature group in its own tint.")
ui.show_features(features)
