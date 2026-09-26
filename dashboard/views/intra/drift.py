"""Intra-case drift signal: state distributions per calendar window, compared window by window."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import seed_widget

WINDOWS = {
    "1h": "1 hour", "6h": "6 hours", "12h": "12 hours", "1D": "1 day",
    "3D": "3 days", "7D": "1 week", "14D": "2 weeks", "30D": "30 days",
}

ui.keep_widgets()
ui.intra_log()
states = ui.require("intra_states", "Compute the states on the **States & Trajectories** page first.",
                    "views/intra/states.py", "States & Trajectories")

seed_widget("intra_sel_drift_window", "7D")
seed_widget("intra_sel_divergence", "js")
seed_widget("intra_sel_reference", "previous")
seed_widget("intra_sel_lookback", 5)

with st.sidebar:
    st.header("Drift Signal")
    st.subheader("Distributions")
    st.selectbox("Window", list(WINDOWS), format_func=WINDOWS.get, key="intra_sel_drift_window")
    st.subheader("Divergences")
    st.selectbox("Divergence", list(kairo.DIVERGENCES), format_func=kairo.DIVERGENCES.get,
                 key="intra_sel_divergence")
    reference = st.selectbox("Compare against", list(kairo.REFERENCES),
                             format_func=lambda r: kairo.REFERENCES[r].capitalize(), key="intra_sel_reference")
    if reference == "recent":
        st.number_input("Windows to average", min_value=1, step=1, key="intra_sel_lookback")

st.title("Drift Signal")
st.caption(f"The {st.session_state['intra_method']} states counted per calendar window, then every "
           "window compared with a reference. A spike is a change in how the process behaves.")

st.subheader("State distribution per window")
if st.button("Compute distributions", type="primary", icon=":material/play_arrow:"):
    distributions = kairo.compute_state_distributions(states, st.session_state["intra_sel_drift_window"])
    ui.clear_from("drift")
    st.session_state["intra_distributions"] = distributions
    st.session_state["intra_plot_distribution"] = kairo.plot_state_distributions(
        distributions, st.session_state["intra_colors"])
ui.show_plot("intra_plot_distribution", "Pick the window in the sidebar and compute the distributions.")

distributions = st.session_state.get("intra_distributions")
if distributions is not None:
    st.caption(f"{len(distributions):,} windows with events.")

st.subheader("Divergences")
if st.button("Compute divergences", type="primary", icon=":material/monitoring:", disabled=distributions is None):
    config = {
        "divergence": st.session_state["intra_sel_divergence"],
        "reference": st.session_state["intra_sel_reference"],
        "lookback": int(st.session_state["intra_sel_lookback"]),
    }
    divergences = kairo.compute_divergences(distributions, **config)
    st.session_state["intra_divergences"] = divergences
    st.session_state["intra_divergence_config"] = config
    st.session_state["intra_plot_divergence"] = kairo.plot_divergences(divergences, config["divergence"], config["reference"])
ui.show_plot("intra_plot_divergence", "Compute the distributions first, then compare the windows.")
