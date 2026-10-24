"""Drift signal: the states counted per signal window, and every signal window's state distribution
compared with a reference, its divergence being the drift score of the signal window. The mean and
quantiles of the drift scores and the rows behind every drift score help to read the signal."""
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
    seed_widget(f"{p}_sel_quantile", 0.9)

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
        signal_window_rows = kairo.compute_signal_window_rows(states, st.session_state[f"{p}_sel_signal_window"])
        ui.clear_from(p, "drift")
        # a new drift signal has other drift scores, its threshold starts again at the 0.9-quantile
        st.session_state.pop(f"{p}_sel_threshold", None)
        st.session_state[f"{p}_distributions"] = distributions
        st.session_state[f"{p}_drift_signal"] = drift_signal
        st.session_state[f"{p}_drift_signal_config"] = config
        st.session_state[f"{p}_signal_window_rows"] = signal_window_rows
        st.session_state[f"{p}_plot_distribution"] = kairo.plot_state_distributions(
            distributions, st.session_state[f"{p}_colors"])
        st.session_state[f"{p}_plot_drift_signal"] = kairo.plot_drift_signal(
            drift_signal, config["divergence"], config["reference"])
        st.session_state[f"{p}_plot_signal_window_rows"] = kairo.plot_signal_window_rows(signal_window_rows)

    distributions = st.session_state.get(f"{p}_distributions")
    if distributions is None:
        st.info("Pick the signal window, the divergence and the reference in the sidebar and compute the drift signal.")
        st.stop()

    st.subheader("Drift signal")
    st.caption("Over time, the drift score of every signal window: the divergence of its state distribution "
               "from the reference.")

    drift_signal = st.session_state[f"{p}_drift_signal"]
    scores = drift_signal["drift_score"].dropna()

    # a threshold to play with, drawn into the plot as a dashed line. it starts at the 0.9-quantile
    # of the drift scores, and the plot the copilot gets shows it too
    if len(scores) and scores.max() > 0:
        highest = float(scores.max())
        seed_widget(f"{p}_sel_threshold", kairo.compute_drift_signal_quantile(drift_signal, 0.9))
        slider_column, _ = st.columns([1, 2])
        threshold = slider_column.slider(
            "Threshold", min_value=0.0, max_value=highest, step=highest / 200, format="%.3f",
            key=f"{p}_sel_threshold", help="The drift score above which a signal window counts as a drift. "
                                          "It starts at the 0.9-quantile of the drift scores.")
        slider_column.caption(f"{int((scores >= threshold).sum()):,} of {len(scores):,} drift scores reach it.")
        config = st.session_state[f"{p}_drift_signal_config"]
        st.session_state[f"{p}_plot_drift_signal"] = kairo.plot_drift_signal(
            drift_signal, config["divergence"], config["reference"], threshold=threshold)

    ui.show_plot(f"{p}_plot_drift_signal", "")

    signal_window_rows = st.session_state.get(f"{p}_signal_window_rows")

    # drift signals computed before the rows existed have none until they are computed again
    if signal_window_rows is not None:
        st.subheader("Drift scores and rows")
        st.caption(f"The mean and the t-quantile of the drift scores and of the {ui.ROWS[p]} per signal window. "
                   "The median (t = 0.5) shows the drift scores without a drift, an upper quantile such as "
                   "t = 0.9 is a starting point for the threshold of a drift. A low quantile of the rows "
                   "shows how thin the thinnest signal windows are.")
        t = st.number_input("Quantile t", min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                            key=f"{p}_sel_quantile", help="The share of the signal windows that stays at or "
                                                         "below the quantile, e.g. 0.5 for the median.")
        ui.metrics_row([
            ("Mean drift score", f"{kairo.compute_drift_signal_mean(drift_signal):.3f}",
             "Over the signal windows that have a drift score."),
            (f"{t:g}-quantile drift score", f"{kairo.compute_drift_signal_quantile(drift_signal, t):.3f}",
             f"A share {t:g} of the drift scores stays at or below it."),
            (f"Mean {ui.ROWS[p]}", f"{kairo.compute_signal_window_rows_mean(signal_window_rows):,.1f}",
             f"The mean number of {ui.ROWS[p]} per signal window."),
            (f"{t:g}-quantile {ui.ROWS[p]}", f"{kairo.compute_signal_window_rows_quantile(signal_window_rows, t):,.1f}",
             f"A share {t:g} of the signal windows holds at most this many {ui.ROWS[p]}."),
        ])

        st.subheader("Rows per signal window")
        st.caption(f"Over time, the {ui.ROWS[p]} behind the drift score of every signal window. A drift score "
                   "of a thin signal window rests on few rows.")
        ui.show_plot(f"{p}_plot_signal_window_rows", "")
    else:
        st.caption("Compute the drift signal again to see the mean, the quantiles and the rows per signal window.")

    st.subheader("State distribution per signal window")
    st.caption(f"{len(distributions):,} signal windows with {ui.ROWS[p]}, the share of every state in each.")
    ui.show_plot(f"{p}_plot_distribution", "")
