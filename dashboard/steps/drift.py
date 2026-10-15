"""Drift signal: the states counted per signal window, and every signal window's state distribution
compared with a reference, its divergence being the drift score of the signal window."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import kairo
import ui
from controls import seed_choice, seed_widget


def signal_windows(p: str) -> list:
    # the signal window lengths to pick from. the resource and inter-case states belong to feature
    # windows, so a signal window has to be at least as long as the feature window
    if p == "intra":
        return list(ui.WINDOWS)

    feature_window = pd.Timedelta(st.session_state[f"{p}_feature_window"])

    return [window for window in ui.WINDOWS if pd.Timedelta(window) >= feature_window]


def show(p: str) -> None:
    ui.keep_widgets()
    ui.perspective_log(p)
    states = ui.require(f"{p}_states", "Compute the states on the **States & Trajectories** page first.",
                        f"views/{p}/states.py", "States & Trajectories")
    options = signal_windows(p)

    if not options:
        st.warning("The feature window is the longest window there is, so no signal window can be longer. "
                   "Compute the features with a shorter feature window.")
        st.stop()

    # a picked signal window that is now shorter than the feature window falls back
    seed_choice(f"{p}_sel_signal_window", "14D" if "14D" in options else options[0], options)
    seed_widget(f"{p}_sel_divergence", "js")
    seed_widget(f"{p}_sel_reference", "previous")
    seed_widget(f"{p}_sel_lookback", 5)

    with st.sidebar:
        st.header("Drift Signal")
        st.selectbox("Signal window", options, format_func=ui.WINDOWS.get, key=f"{p}_sel_signal_window",
                     help="The length of the calendar windows the drift signal has one drift score for."
                          + ("" if p == "intra" else " At least as long as the feature window."))
        st.selectbox("Divergence", list(kairo.DIVERGENCES), format_func=kairo.DIVERGENCES.get,
                     key=f"{p}_sel_divergence")
        # what every signal window is compared against, with the number of signal windows the recent mean takes
        reference = st.selectbox("Compare against", list(kairo.REFERENCES),
                                 format_func=lambda r: kairo.REFERENCES[r].capitalize(), key=f"{p}_sel_reference")
        if reference == "recent":
            st.number_input("Signal windows to average", min_value=1, step=1, key=f"{p}_sel_lookback")

    st.title("Drift Signal")
    st.caption(f"The {st.session_state[f'{p}_method']} states counted per signal window, then every signal "
               "window's state distribution compared with a reference. The divergence is the drift score of "
               "the signal window, a spike is a change in how the process behaves.")

    if st.button("Compute drift signal", type="primary", icon=":material/monitoring:"):
        config = {
            "divergence": st.session_state[f"{p}_sel_divergence"],
            "reference": st.session_state[f"{p}_sel_reference"],
            "lookback": int(st.session_state[f"{p}_sel_lookback"]),
        }
        distributions = kairo.compute_state_distributions(states, st.session_state[f"{p}_sel_signal_window"])
        drift_signal = kairo.compute_drift_signal(distributions, **config)
        ui.clear_from(p, "drift")
        st.session_state[f"{p}_distributions"] = distributions
        st.session_state[f"{p}_drift_signal"] = drift_signal
        st.session_state[f"{p}_drift_signal_config"] = config
        st.session_state[f"{p}_plot_distribution"] = kairo.plot_state_distributions(
            distributions, st.session_state[f"{p}_colors"])
        st.session_state[f"{p}_plot_drift_signal"] = kairo.plot_drift_signal(
            drift_signal, config["divergence"], config["reference"])

    distributions = st.session_state.get(f"{p}_distributions")
    if distributions is None:
        st.info("Pick the signal window, the divergence and the reference in the sidebar and compute the drift signal.")
        st.stop()

    st.subheader("Drift signal")
    st.caption("Over time, the drift score of every signal window: the divergence of its state distribution "
               "from the reference.")
    ui.show_plot(f"{p}_plot_drift_signal", "")

    st.subheader("State distribution per signal window")
    st.caption(f"{len(distributions):,} signal windows with {ui.ROWS[p]}, the share of every state in each.")
    ui.show_plot(f"{p}_plot_distribution", "")
