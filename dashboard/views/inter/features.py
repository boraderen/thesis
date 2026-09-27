"""Inter-case features: one row per calendar window, describing all cases running in it."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import log_signature, seed_widget

FEATURES = {
    "active_cases": "Active cases",
    "new_arrivals": "New arrivals",
    "completions": "Completions",
    "events_per_case": "Events per active case",
    "mean_delta_t": "Mean Δt",
    "std_delta_t": "Std Δt",
    "stalled_cases": "Stalled cases",
}
# the attribute features, with the kind of case attribute they need
ATTRIBUTE_FEATURES = {
    "attr_means": ("numerical", "Means of the numerical attributes"),
    "attr_stds": ("numerical", "Stds of the numerical attributes"),
    "attr_shares": ("categorical", "Value shares of the categorical attributes"),
}
THRESHOLDS = {"1h": "1 hour", "6h": "6 hours", "12h": "12 hours", "1D": "1 day", "3D": "3 days", "7D": "1 week"}
GLOSSARY = {
    "Active cases": "Number of distinct cases with at least one event in the calendar window.",
    "New arrivals": "Number of cases whose first event falls inside the window.",
    "Completions": "Number of cases whose last event falls inside the window.",
    "Events per active case": "Events in the window divided by the number of cases active in it.",
    "Mean Δt": "Mean gap between an event and the previous event of the same case, in minutes.",
    "Std Δt": "Standard deviation of those within-case gaps, in minutes.",
    "Stalled cases": "Number of cases still running at the window end whose most recent event is older than "
                     "the stall threshold τ. Completed cases are not counted.",
}

ui.keep_widgets()
log = ui.perspective_log("inter")
attributes = st.session_state.get("case_attributes", {})
kinds = set(attributes.values())

# the attribute features are named after the case attributes mapped for this log
glossary = dict(GLOSSARY)
for column, kind in attributes.items():
    if kind == "numerical":
        glossary[f"Mean {column}"] = f"Mean of {column} over the events in the window."
        glossary[f"Std {column}"] = f"Standard deviation of {column} over the events in the window."
    else:
        glossary[f"{column} value shares"] = (
            f"One column per value of {column} ({log[column].nunique()} in this log) — the share of the window's "
            "events carrying that value. The values come as a set, not one by one.")

for feature in FEATURES | ATTRIBUTE_FEATURES:
    seed_widget(f"inter_sel_feature_{feature}", True)
seed_widget("inter_sel_window", "1D")
seed_widget("inter_sel_stall_threshold", "1D")

with st.sidebar:
    st.header("Features")
    for feature, label in FEATURES.items():
        st.checkbox(label, key=f"inter_sel_feature_{feature}")
    # only the attribute features the mapped case attributes allow
    for feature, (kind, label) in ATTRIBUTE_FEATURES.items():
        if kind in kinds:
            st.checkbox(label, key=f"inter_sel_feature_{feature}")
    st.selectbox("Window", list(ui.WINDOWS), format_func=ui.WINDOWS.get, key="inter_sel_window")
    st.selectbox("Stall threshold", list(THRESHOLDS), format_func=THRESHOLDS.get, key="inter_sel_stall_threshold",
                 help="A running case counts as stalled at a window end when its most recent event is older than this.")
    if attributes:
        st.caption("Case attributes: " + ", ".join(f"{column} ({kind})" for column, kind in attributes.items()))
    else:
        st.caption("No case attributes mapped, pick them on the **Upload log** page.")
    ui.feature_glossary(glossary)

st.title("Features")
st.caption("One row per calendar window, describing all cases running in it.")

if st.button("Compute features", type="primary", icon=":material/play_arrow:"):
    picked = [f for f in FEATURES if st.session_state[f"inter_sel_feature_{f}"]]
    picked += [f for f, (kind, _) in ATTRIBUTE_FEATURES.items()
               if kind in kinds and st.session_state[f"inter_sel_feature_{f}"]]
    if not picked:
        st.warning("Select at least one feature group in the sidebar.")
        st.stop()
    with st.spinner("Computing features…"):
        try:
            features = kairo.compute_features_inter(log, st.session_state["inter_sel_window"], picked, attributes,
                                                    st.session_state["inter_sel_stall_threshold"])
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    ui.clear_from("inter", "features")
    st.session_state["inter_features"] = features
    st.session_state["inter_log_signature"] = log_signature(log, "inter")

features = st.session_state.get("inter_features")
if features is None:
    st.info("Pick the feature groups, the window and the stall threshold in the sidebar and compute them.")
    st.stop()

groups = ui.feature_groups(features)
ui.metrics_row([
    ("Windows", f"{len(features):,}"),
    ("Feature columns", f"{sum(map(len, groups.values())):,}"),
    ("Feature groups", f"{len(groups)}"),
    ("Case attributes", f"{len(attributes)}"),
])
st.caption("The first 30 windows, every feature group in its own tint.")
ui.show_features(features)
