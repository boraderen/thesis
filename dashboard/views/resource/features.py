"""Resource features: one row per feature window, describing how the resources worked in it."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import log_signature, seed_multi, seed_widget

# the feature groups with their labels and meanings, as kairo defines them
GLOSSARY = kairo.RESOURCE_GLOSSARY
FEATURES = {feature: label for feature, (label, _) in GLOSSARY.items()}

ui.keep_widgets()
log = ui.perspective_log("resource")
resources = sorted(log["org:resource"].dropna().unique()) if "org:resource" in log.columns else []

for feature in FEATURES:
    seed_widget(f"resource_sel_feature_{feature}", True)
seed_widget("resource_sel_feature_window", "1D")
seed_multi("resource_sel_resources", [], resources)

with st.sidebar:
    st.header("Features")
    for feature, label in FEATURES.items():
        st.checkbox(label, key=f"resource_sel_feature_{feature}")
    st.selectbox("Feature window", list(ui.WINDOWS), format_func=ui.WINDOWS.get, key="resource_sel_feature_window",
                 help="The length of the calendar windows, every feature window is one row of the features.")
    st.multiselect("Resources", resources, key="resource_sel_resources", placeholder="All resources",
                   help="The resources that get feature columns. Left empty, all of them.")
    ui.feature_glossary(GLOSSARY)

st.title("Features")
st.caption("One row per feature window, describing how the resources worked in it.")

if st.button("Compute features", type="primary", icon=":material/play_arrow:"):
    picked = [f for f in FEATURES if st.session_state[f"resource_sel_feature_{f}"]]
    if not picked:
        st.warning("Select at least one feature group in the sidebar.")
        st.stop()
    with st.spinner("Computing features…"):
        try:
            features = kairo.compute_features_resource(
                log, feature_window=st.session_state["resource_sel_feature_window"], features=picked,
                resources=st.session_state["resource_sel_resources"] or None)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    ui.clear_from("resource", "features")
    st.session_state["resource_features"] = features
    st.session_state["resource_feature_window"] = st.session_state["resource_sel_feature_window"]
    st.session_state["resource_log_signature"] = log_signature(log, "resource")

features = st.session_state.get("resource_features")
if features is None:
    st.info("Pick the feature groups, the feature window and the resources in the sidebar and compute them.")
    st.stop()

groups = ui.feature_groups(features)
ui.metrics_row([
    ("Feature windows", f"{len(features):,}"),
    ("Feature columns", f"{sum(map(len, groups.values())):,}"),
    ("Feature groups", f"{len(groups)}"),
])
st.caption("The first 30 feature windows, every feature group in its own tint.")
ui.show_features(features)
