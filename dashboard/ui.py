"""Shared helpers for the pipeline pages of the three perspectives."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from controls import log_signature
from kairo.analysis import META
from tables import styled_feature_table

PERSPECTIVES = {"intra": "Intra-case", "resource": "Resource", "inter": "Inter-case"}

# what a row of the features stands for
ROWS = {"intra": "events", "resource": "windows", "inter": "windows"}

WINDOWS = {
    "1h": "1 hour", "6h": "6 hours", "12h": "12 hours", "1D": "1 day",
    "3D": "3 days", "7D": "1 week", "14D": "2 weeks", "30D": "30 days",
}

# the results of a perspective's pipeline by the step that computes them, stored under the
# perspective's prefix, e.g. intra_features. when a step runs again, the results of every
# later step were computed from its old output
STEPS = {
    "features": ["features", "log_signature"],
    "pca": ["scaled", "pca", "plot_variance"],
    "cut": ["cut", "pca_skipped", "compressed", "plot_cut", "plot_kdistance"],
    "states": ["method", "model", "states", "frequencies", "colors", "distances",
               "plot_colors", "plot_distances", "plot_frequencies_1", "plot_frequencies_2",
               "trajectories", "last_case", "range_trajectories", "plot_trajectories",
               "log_trajectory", "plot_log_trajectory", "scores"],
    # the window distances only need the compressed rows, they come before the distributions,
    # so computing the distributions again keeps them
    "distances": ["window_distances", "window_distance_config", "plot_window_distance"],
    "drift": ["distributions", "divergences", "divergence_config", "plot_distribution", "plot_divergence"],
}

# the plots a page stores, by the name the copilot offers them under
PLOTS = {
    "plot_variance": "PCA explained variance",
    "plot_cut": "PCA explained variance with the cut",
    "plot_kdistance": "DBSCAN k-distance curve",
    "plot_colors": "State colors",
    "plot_distances": "Distances between states",
    "plot_frequencies_1": "State frequencies, first range",
    "plot_frequencies_2": "State frequencies, second range",
    "plot_trajectories": "Trajectories of the cases in a range",
    "plot_log_trajectory": "Trajectory of the log",
    "plot_distribution": "State distribution per window",
    "plot_divergence": "Divergence per window",
    "plot_window_distance": "Window distances",
}


def require_log() -> pd.DataFrame:
    """The mapped log from session state, or a friendly stop."""
    if "log" not in st.session_state:
        st.info("No event log loaded yet — start on the **Upload log** page.")
        st.page_link("views/upload.py", label="Go to Upload", icon=":material/upload_file:")
        st.stop()
    return st.session_state["log"]


def perspective_log(p: str) -> pd.DataFrame:
    # the loaded log, dropping every result of the perspective that was computed on another one
    log = require_log()

    if st.session_state.get(f"{p}_log_signature") not in (None, log_signature(log, p)):
        clear_from(p, "features")

    return log


def keep_widgets() -> None:
    # streamlit forgets a widget's value once its page is left. assigning the value to
    # itself keeps it, so every page shows its sidebar parameters as they were left
    for key in list(st.session_state):
        if "_sel_" in key:
            st.session_state[key] = st.session_state[key]


def clear_from(p: str, step: str) -> None:
    steps = list(STEPS)

    for later in steps[steps.index(step):]:
        for key in STEPS[later]:
            st.session_state.pop(f"{p}_{key}", None)


def require(key: str, message: str, page: str, label: str):
    """A result of an earlier step, or a stop pointing at the page that computes it."""
    if st.session_state.get(key) is None:
        st.info(message)
        st.page_link(page, label=f"Go to {label}", icon=":material/arrow_back:")
        st.stop()
    return st.session_state[key]


def show_plot(key: str, hint: str) -> None:
    # the key keeps two identical plots, e.g. frequencies of the same range, from colliding
    if key in st.session_state:
        st.plotly_chart(st.session_state[key], width="stretch", key=f"{key}_chart")
    else:
        st.caption(hint)


def stored_plots(p: str) -> dict:
    plots = {name: st.session_state[f"{p}_{key}"] for key, name in PLOTS.items() if f"{p}_{key}" in st.session_state}

    for case_id, trajectory in st.session_state.get(f"{p}_trajectories", {}).items():
        plots[f"Trajectory of case {case_id}"] = trajectory["plot"]

    return plots


def feature_groups(features: pd.DataFrame) -> dict:
    # the feature columns by their group, the part of the name before the first colon
    values = features.drop(columns=META, errors="ignore")
    groups = values.columns.str.split(":").str[0]

    return {group: list(values.columns[groups == group]) for group in dict.fromkeys(groups)}


def show_features(features: pd.DataFrame) -> None:
    # the first 30 rows, every feature group in its own tint. a very wide matrix, e.g. of
    # hundreds of resources, is shown plain, styling it would take long
    if features.shape[1] <= 2000:
        st.dataframe(styled_feature_table(features, feature_groups(features)), width="stretch", height=420)
    else:
        st.dataframe(features.head(30), width="stretch", height=420)


def feature_glossary(entries: dict) -> None:
    # what every feature means, collapsed below the parameters of a features page
    with st.expander("Feature glossary"):
        for term, meaning in entries.items():
            st.markdown(f"**{term}:** {meaning}")


def metrics_row(items: list[tuple]) -> None:
    # every item is a label and a value, and optionally a hover explanation
    columns = st.columns(len(items))
    for column, (label, value, *help) in zip(columns, items):
        column.metric(label, value, help=help[0] if help else None)
