"""Inter-case features: one row per feature window, describing all cases running in it."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import log_signature, seed_widget

# the feature groups with their labels and meanings, as kairo defines them. the attribute
# features come with the kind of case attribute they need
GLOSSARY = kairo.INTER_GLOSSARY
ATTRIBUTE_KINDS = {"attr_means": "numerical", "attr_stds": "numerical", "attr_shares": "categorical"}
FEATURES = {feature: label for feature, (label, _) in GLOSSARY.items() if feature not in ATTRIBUTE_KINDS}
ATTRIBUTE_FEATURES = {feature: (kind, GLOSSARY[feature][0]) for feature, kind in ATTRIBUTE_KINDS.items()}
THRESHOLDS = {"1h": "1 hour", "6h": "6 hours", "12h": "12 hours", "1D": "1 day", "3D": "3 days", "7D": "1 week"}

ui.keep_widgets()
log = ui.perspective_log("inter")
attributes = st.session_state.get("case_attributes", {})
kinds = set(attributes.values())

# the glossary leaves out the attribute features the mapped case attributes do not allow
glossary = {feature: entry for feature, entry in GLOSSARY.items()
            if feature not in ATTRIBUTE_KINDS or ATTRIBUTE_KINDS[feature] in kinds}

for feature in FEATURES | ATTRIBUTE_FEATURES:
    seed_widget(f"inter_sel_feature_{feature}", True)
seed_widget("inter_sel_feature_window", "1D")
seed_widget("inter_sel_stall_threshold", "1D")

with st.sidebar:
    st.header("Features")
    for feature, label in FEATURES.items():
        st.checkbox(label, key=f"inter_sel_feature_{feature}")
    # only the attribute features the mapped case attributes allow
    for feature, (kind, label) in ATTRIBUTE_FEATURES.items():
        if kind in kinds:
            st.checkbox(label, key=f"inter_sel_feature_{feature}")
    st.selectbox("Feature window", list(ui.WINDOWS), format_func=ui.WINDOWS.get, key="inter_sel_feature_window",
                 help="The length of the calendar windows, every feature window is one row of the features.")
    st.selectbox("Stall threshold", list(THRESHOLDS), format_func=THRESHOLDS.get, key="inter_sel_stall_threshold",
                 help="A running case counts as stalled at the end of a feature window when its most recent event is older than this.")
    if attributes:
        st.caption("Case attributes: " + ", ".join(f"{column} ({kind})" for column, kind in attributes.items()))
    else:
        st.caption("No case attributes mapped, pick them on the **Upload log** page.")
    ui.feature_glossary(glossary)

st.title("Features")
st.caption("One row per feature window, describing all cases running in it.")

if st.button("Compute features", type="primary", icon=":material/play_arrow:"):
    picked = [f for f in FEATURES if st.session_state[f"inter_sel_feature_{f}"]]
    picked += [f for f, (kind, _) in ATTRIBUTE_FEATURES.items()
               if kind in kinds and st.session_state[f"inter_sel_feature_{f}"]]
    if not picked:
        st.warning("Select at least one feature group in the sidebar.")
        st.stop()
    with st.spinner("Computing features…"):
        try:
            features = kairo.compute_features_inter(
                log, feature_window=st.session_state["inter_sel_feature_window"], features=picked,
                case_attributes=attributes, stall_threshold=st.session_state["inter_sel_stall_threshold"])
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    ui.clear_from("inter", "features")
    st.session_state["inter_features"] = features
    st.session_state["inter_feature_window"] = st.session_state["inter_sel_feature_window"]
    st.session_state["inter_log_signature"] = log_signature(log, "inter")

features = st.session_state.get("inter_features")
if features is None:
    st.info("Pick the feature groups, the feature window and the stall threshold in the sidebar and compute them.")
    st.stop()

groups = ui.feature_groups(features)
ui.metrics_row([
    ("Feature windows", f"{len(features):,}"),
    ("Feature columns", f"{sum(map(len, groups.values())):,}"),
    ("Feature groups", f"{len(groups)}"),
    ("Case attributes", f"{len(attributes)}"),
])
st.caption("The first 30 feature windows, every feature group in its own tint.")
ui.show_features(features)
