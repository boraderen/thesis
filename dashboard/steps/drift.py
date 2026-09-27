"""Drift signal: state distributions per calendar window compared window by window, and for the
windowed perspectives the distances between the windows' vectors."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import seed_widget


def reference_inputs(key: str) -> None:
    # what every window is compared against, with the number of windows the recent mean takes
    reference = st.selectbox("Compare against", list(kairo.REFERENCES),
                             format_func=lambda r: kairo.REFERENCES[r].capitalize(), key=f"{key}_reference")
    if reference == "recent":
        st.number_input("Windows to average", min_value=1, step=1, key=f"{key}_lookback")


def show(p: str) -> None:
    ui.keep_widgets()
    ui.perspective_log(p)
    states = ui.require(f"{p}_states", "Compute the states on the **States & Trajectories** page first.",
                        f"views/{p}/states.py", "States & Trajectories")
    windowed = p != "intra"

    # the windowed perspectives count states of feature windows, so their windows are larger
    seed_widget(f"{p}_sel_drift_window", "30D" if windowed else "7D")
    seed_widget(f"{p}_sel_divergence", "kl")
    seed_widget(f"{p}_sel_reference", "recent")
    seed_widget(f"{p}_sel_lookback", 5)
    seed_widget(f"{p}_sel_distance", "euclidean")
    seed_widget(f"{p}_sel_distance_reference", "recent")
    seed_widget(f"{p}_sel_distance_lookback", 5)

    with st.sidebar:
        st.header("Drift Signal")
        st.subheader("Distributions")
        st.selectbox("Window", list(ui.WINDOWS), format_func=ui.WINDOWS.get, key=f"{p}_sel_drift_window")
        st.subheader("Divergences")
        st.selectbox("Divergence", list(kairo.DIVERGENCES), format_func=kairo.DIVERGENCES.get,
                     key=f"{p}_sel_divergence")
        reference_inputs(f"{p}_sel")
        if windowed:
            st.subheader("Window distances")
            st.selectbox("Distance", list(kairo.DISTANCES), format_func=kairo.DISTANCES.get, key=f"{p}_sel_distance")
            reference_inputs(f"{p}_sel_distance")

    st.title("Drift Signal")
    st.caption(f"The {st.session_state[f'{p}_method']} states counted per calendar window, then every "
               "window compared with a reference. A spike is a change in how the process behaves.")

    st.subheader("State distribution per window")
    if windowed:
        st.caption("Every state is a state of a feature window, so pick a larger window here: "
                   "each one then holds the states of several feature windows.")
    if st.button("Compute distributions", type="primary", icon=":material/play_arrow:"):
        distributions = kairo.compute_state_distributions(states, st.session_state[f"{p}_sel_drift_window"])
        ui.clear_from(p, "drift")
        st.session_state[f"{p}_distributions"] = distributions
        st.session_state[f"{p}_plot_distribution"] = kairo.plot_state_distributions(
            distributions, st.session_state[f"{p}_colors"])
    ui.show_plot(f"{p}_plot_distribution", "Pick the window in the sidebar and compute the distributions.")

    distributions = st.session_state.get(f"{p}_distributions")
    if distributions is not None:
        st.caption(f"{len(distributions):,} windows with {ui.ROWS[p]}.")

    st.subheader("Divergences")
    if st.button("Compute divergences", type="primary", icon=":material/monitoring:", disabled=distributions is None):
        config = {
            "divergence": st.session_state[f"{p}_sel_divergence"],
            "reference": st.session_state[f"{p}_sel_reference"],
            "lookback": int(st.session_state[f"{p}_sel_lookback"]),
        }
        divergences = kairo.compute_divergences(distributions, **config)
        st.session_state[f"{p}_divergences"] = divergences
        st.session_state[f"{p}_divergence_config"] = config
        st.session_state[f"{p}_plot_divergence"] = kairo.plot_divergences(
            divergences, config["divergence"], config["reference"])
    ui.show_plot(f"{p}_plot_divergence", "Compute the distributions first, then compare the windows.")

    if not windowed:
        return

    st.subheader("Window distances")
    st.caption("The vector every feature window got on the PCA page, as the states were computed from it, "
               "compared with a reference. No states involved, the vectors are compared directly.")
    if st.button("Compute window distances", type="primary", icon=":material/straighten:"):
        config = {
            "distance": st.session_state[f"{p}_sel_distance"],
            "reference": st.session_state[f"{p}_sel_distance_reference"],
            "lookback": int(st.session_state[f"{p}_sel_distance_lookback"]),
        }
        distances = kairo.compute_window_distances(st.session_state[f"{p}_compressed"], **config)
        st.session_state[f"{p}_window_distances"] = distances
        st.session_state[f"{p}_window_distance_config"] = config
        st.session_state[f"{p}_plot_window_distance"] = kairo.plot_window_distances(
            distances, config["distance"], config["reference"])
    ui.show_plot(f"{p}_plot_window_distance", "Pick the distance in the sidebar and compare the windows.")
