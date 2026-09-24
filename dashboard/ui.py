"""Shared helpers for the pipeline pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from controls import log_signature

# the results of the intra-case pipeline, by the step that computes them. when a step
# runs again, the results of every later step were computed from its old output
INTRA_STEPS = {
    "features": ["intra_features", "intra_log_signature"],
    "pca": ["intra_scaled", "intra_pca", "intra_plot_variance"],
    "cut": ["intra_cut", "intra_compressed", "intra_plot_cut", "intra_plot_kdistance"],
    "states": ["intra_method", "intra_model", "intra_states", "intra_colors", "intra_distances",
               "intra_plot_colors", "intra_plot_frequencies", "intra_plot_distances",
               "intra_trajectories", "intra_last_case"],
}

# the plots a page stores, by the name the copilot offers them under
INTRA_PLOTS = {
    "intra_plot_variance": "PCA explained variance",
    "intra_plot_cut": "PCA explained variance with the cut",
    "intra_plot_kdistance": "DBSCAN k-distance curve",
    "intra_plot_colors": "State colors",
    "intra_plot_frequencies": "State frequencies",
    "intra_plot_distances": "Distances between states",
}


def require_log() -> pd.DataFrame:
    """The mapped log from session state, or a friendly stop."""
    if "log" not in st.session_state:
        st.info("No event log loaded yet — start on the **Upload log** page.")
        st.page_link("views/upload.py", label="Go to Upload", icon=":material/upload_file:")
        st.stop()
    return st.session_state["log"]


def intra_log() -> pd.DataFrame:
    # the loaded log, dropping every intra-case result that was computed on another one
    log = require_log()

    if st.session_state.get("intra_log_signature") not in (None, log_signature(log, "intra")):
        clear_from("features")

    return log


def keep_widgets() -> None:
    # streamlit forgets a widget's value once its page is left. assigning the value to
    # itself keeps it, so every page shows its sidebar parameters as they were left
    for key in list(st.session_state):
        if key.startswith("intra_sel_"):
            st.session_state[key] = st.session_state[key]


def clear_from(step: str) -> None:
    steps = list(INTRA_STEPS)

    for later in steps[steps.index(step):]:
        for key in INTRA_STEPS[later]:
            st.session_state.pop(key, None)


def require(key: str, message: str, page: str, label: str):
    """A result of an earlier step, or a stop pointing at the page that computes it."""
    if st.session_state.get(key) is None:
        st.info(message)
        st.page_link(page, label=f"Go to {label}", icon=":material/arrow_back:")
        st.stop()
    return st.session_state[key]


def show_plot(key: str, hint: str) -> None:
    if key in st.session_state:
        st.plotly_chart(st.session_state[key], width="stretch")
    else:
        st.caption(hint)


def stored_plots() -> dict:
    plots = {name: st.session_state[key] for key, name in INTRA_PLOTS.items() if key in st.session_state}

    for case_id, trajectory in st.session_state.get("intra_trajectories", {}).items():
        plots[f"Trajectory of case {case_id}"] = trajectory["plot"]

    return plots


def metrics_row(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)
