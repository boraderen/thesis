"""States: cluster the rows into states, then follow the cases, or the log itself, through them."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import kairo
import ui
from controls import seed_widget
from kairo.analysis import META

METHODS = ["SOM", "k-means", "DBSCAN"]


def range_inputs(key: str, log: pd.DataFrame) -> None:
    # value=None, the default of today would have to lie within the dates of the log
    first_day = log["time:timestamp"].min().date()
    last_day = log["time:timestamp"].max().date()

    start, end = st.columns(2)
    start.date_input("From", value=None, min_value=first_day, max_value=last_day, key=f"{key}_start")
    end.date_input("To", value=None, min_value=first_day, max_value=last_day, key=f"{key}_end")


def picked_range(key: str) -> tuple:
    # the picked dates as text, with the end day counting as a whole
    start = st.session_state[f"{key}_start"]
    end = st.session_state[f"{key}_end"]

    return (str(start) if start else None, f"{end} 23:59:59" if end else None)


def compute_states(p: str, method: str, compressed: pd.DataFrame) -> dict:
    """The fitted model and the state of every row, for the method in the sidebar."""
    meta = compressed[[c for c in META if c in compressed.columns]]

    if method == "SOM":
        size = (int(st.session_state[f"{p}_sel_rows"]), int(st.session_state[f"{p}_sel_cols"]))
        distance = st.session_state[f"{p}_sel_som_distance"]
        model = kairo.compute_som(compressed, size, float(st.session_state[f"{p}_sel_learning_rate"]), distance)
        states = meta.join(kairo.get_som_winners(compressed, model))
        frequencies = kairo.get_som_state_frequencies(states)
        colors = kairo.compute_som_color_mapping(size)
        distances = kairo.get_som_state_distances(model, distance)
    elif method == "k-means":
        k = int(st.session_state[f"{p}_sel_k"])
        model = kairo.compute_kmeans(compressed, k)
        states = meta.join(kairo.get_kmeans_clusters(compressed, model))
        frequencies = kairo.get_kmeans_state_frequencies(states)
        colors = kairo.compute_kmeans_color_mapping(k)
        distances = kairo.get_kmeans_state_distances(model)
    else:
        model = kairo.compute_dbscan(compressed, float(st.session_state[f"{p}_sel_eps"]),
                                     int(st.session_state[f"{p}_sel_min_samples"]),
                                     st.session_state[f"{p}_sel_dbscan_distance"])
        states = meta.join(kairo.get_dbscan_clusters(compressed, model))
        frequencies = kairo.get_dbscan_state_frequencies(states)
        colors = kairo.compute_dbscan_color_mapping(model)
        distances = kairo.get_dbscan_state_distances(model)

    return {"method": method, "model": model, "states": states, "frequencies": frequencies,
            "colors": colors, "distances": distances}


def show(p: str) -> None:
    ui.keep_widgets()
    log = ui.perspective_log(p)
    compressed = ui.require(f"{p}_compressed", "Apply or skip PCA on the **PCA** page first.",
                            f"views/{p}/pca.py", "PCA")
    rows = ui.ROWS[p]

    # the windowed perspectives have a few hundred rows, a smaller grid fits them better
    seed_widget(f"{p}_sel_method", "SOM")
    seed_widget(f"{p}_sel_rows", 5 if p == "intra" else 3)
    seed_widget(f"{p}_sel_cols", 5 if p == "intra" else 3)
    seed_widget(f"{p}_sel_learning_rate", 0.5)
    seed_widget(f"{p}_sel_som_distance", "euclidean")
    seed_widget(f"{p}_sel_k", 5)
    seed_widget(f"{p}_sel_eps", 0.5)
    seed_widget(f"{p}_sel_min_samples", 5)
    seed_widget(f"{p}_sel_dbscan_distance", "euclidean")
    seed_widget(f"{p}_sel_traj_count", 1000)

    with st.sidebar:
        st.header("States")
        method = st.radio("Method", METHODS, key=f"{p}_sel_method", horizontal=True)
        if method == "SOM":
            grid_rows, grid_cols = st.columns(2)
            grid_rows.number_input("Grid rows", min_value=1, max_value=30, step=1, key=f"{p}_sel_rows")
            grid_cols.number_input("Grid columns", min_value=1, max_value=30, step=1, key=f"{p}_sel_cols")
            st.number_input("Learning rate", min_value=0.01, max_value=1.0, step=0.05,
                            key=f"{p}_sel_learning_rate")
            st.selectbox("Distance", list(kairo.DISTANCES), key=f"{p}_sel_som_distance")
        elif method == "k-means":
            st.number_input("Clusters k", min_value=2, max_value=30, step=1, key=f"{p}_sel_k")
        else:
            st.number_input("eps", min_value=0.001, step=0.05, format="%.3f", key=f"{p}_sel_eps")
            st.number_input("Min samples", min_value=1, step=1, key=f"{p}_sel_min_samples")
            st.selectbox("Distance", list(kairo.DISTANCES), key=f"{p}_sel_dbscan_distance")

    st.title("States & Trajectories")
    if p == "intra":
        st.caption("Every event gets a state from its compressed features, then cases are followed through them.")
    else:
        st.caption("Every calendar window gets a state from its compressed features, "
                   "then the log is followed through them window by window.")

    if method == "DBSCAN":
        st.subheader("k-distance curve")
        if st.button("Plot k-distance", icon=":material/show_chart:"):
            with st.spinner("Finding the neighbours of every row…"):
                st.session_state[f"{p}_plot_kdistance"] = kairo.plot_dbscan_k_distance(
                    compressed, int(st.session_state[f"{p}_sel_min_samples"]),
                    st.session_state[f"{p}_sel_dbscan_distance"])
        if p == "intra":
            hint = ("The knee of the curve is a good eps. Repeating prefixes keep it at 0 for most "
                    "events, so the knee sits at the far right end.")
        else:
            hint = "The knee of the curve is a good eps."
        ui.show_plot(f"{p}_plot_kdistance", hint)

    st.subheader("States")
    if st.button("Compute states", type="primary", icon=":material/play_arrow:"):
        with st.spinner(f"Computing the {method} states…"):
            try:
                result = compute_states(p, method, compressed)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
        ui.clear_from(p, "states")
        st.session_state.update({f"{p}_{key}": value for key, value in result.items()})

    states = st.session_state.get(f"{p}_states")
    if states is None:
        st.info("Pick a method and its parameters in the sidebar, then compute the states.")
        st.stop()

    # from here on everything shows the computed states, even if the sidebar has moved on
    computed = st.session_state[f"{p}_method"]
    model = st.session_state[f"{p}_model"]
    colors = st.session_state[f"{p}_colors"]

    ui.metrics_row([
        ("Method", computed),
        ("States", f"{len(st.session_state[f'{p}_distances']):,}"),
        (rows.capitalize(), f"{len(states):,}"),
    ])
    if computed != method:
        st.caption(f"These are the {computed} states — compute again to switch to {method}.")

    # the colors of k-means and dbscan clusters show in their frequency bars, the som grid
    # gets its own plot
    if computed == "SOM":
        colors_column, distances_column = st.columns(2)
        with colors_column:
            st.subheader("State colors")
            if st.button("Plot state colors", icon=":material/palette:"):
                st.session_state[f"{p}_plot_colors"] = kairo.plot_som_colors(states, colors)
            ui.show_plot(f"{p}_plot_colors", f"The color of every cell, with how many {rows} it holds.")
    else:
        distances_column = st.container()

    with distances_column:
        st.subheader("Distances between states")
        if st.button("Plot distances", icon=":material/grid_on:"):
            if computed == "SOM":
                fig = kairo.plot_som_u_matrix(model)
            elif computed == "k-means":
                fig = kairo.plot_kmeans_distances(st.session_state[f"{p}_distances"])
            else:
                fig = kairo.plot_dbscan_distances(st.session_state[f"{p}_distances"])
            st.session_state[f"{p}_plot_distances"] = fig
        ui.show_plot(f"{p}_plot_distances",
                     "For a SOM the u-matrix, the distance of every neuron to its neighbours. "
                     "For k-means and DBSCAN the distance between every two clusters.")

    st.subheader("Frequencies")
    st.caption("Two date ranges side by side, e.g. before and after a suspected drift.")
    for column, number in zip(st.columns(2), (1, 2)):
        with column:
            range_inputs(f"{p}_sel_freq{number}", log)
            if st.button("Plot frequencies", icon=":material/bar_chart:", key=f"{p}_freq_button_{number}"):
                start_date, end_date = picked_range(f"{p}_sel_freq{number}")
                if computed == "SOM":
                    fig = kairo.plot_som_heatmap(states, model.get_weights().shape[:2], start_date, end_date)
                elif computed == "k-means":
                    fig = kairo.plot_kmeans_frequencies(
                        kairo.get_kmeans_state_frequencies(states, start_date, end_date), colors)
                else:
                    fig = kairo.plot_dbscan_frequencies(
                        kairo.get_dbscan_state_frequencies(states, start_date, end_date), colors)
                if computed != "SOM":
                    # the range was picked for the getter, the title shows it like the som heatmap's
                    fig.update_layout(title=f"{fig.layout.title.text} ({start_date or 'start'} to {end_date or 'end'})")
                st.session_state[f"{p}_plot_frequencies_{number}"] = fig
            ui.show_plot(f"{p}_plot_frequencies_{number}", f"{rows.capitalize()} per state between the two dates.")

    if p == "intra":
        case_trajectories(p, log, states, computed, colors)
    else:
        log_trajectory(p, log, states, computed, colors)


def case_trajectories(p: str, log: pd.DataFrame, states: pd.DataFrame, computed: str, colors: dict) -> None:
    st.subheader("Case trajectory")
    case_input, case_button = st.columns([3, 1], vertical_alignment="bottom")
    case_input.text_input("Case id", key=f"{p}_sel_case")
    if case_button.button("Compute trajectory", icon=":material/route:", width="stretch"):
        case_id = st.session_state[f"{p}_sel_case"].strip()
        if not case_id:
            st.warning("Enter a case id first.")
        else:
            try:
                if computed == "SOM":
                    visits = kairo.get_som_case_trajectory(states, case_id)
                    fig = kairo.plot_som_case_trajectory(visits, colors, case_id)
                elif computed == "k-means":
                    visits = kairo.get_kmeans_case_trajectory(states, case_id)
                    fig = kairo.plot_kmeans_case_trajectory(visits, colors, case_id)
                else:
                    visits = kairo.get_dbscan_case_trajectory(states, case_id)
                    fig = kairo.plot_dbscan_case_trajectory(visits, colors, case_id)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.setdefault(f"{p}_trajectories", {})[case_id] = {"visits": visits, "plot": fig}
                st.session_state[f"{p}_last_case"] = case_id

    case_id = st.session_state.get(f"{p}_last_case")
    if case_id is None:
        st.caption("Enter a case id and compute its trajectory.")
    else:
        trajectory = st.session_state[f"{p}_trajectories"][case_id]
        st.plotly_chart(trajectory["plot"], width="stretch")

        st.markdown(f"**Transitions of case {case_id}**")
        st.caption("One row per visit of a state. The case moves into every row's state at its start.")
        visits = trajectory["visits"]
        st.dataframe(visits.assign(duration=visits["duration"].astype(str)), width="stretch", hide_index=True)

    st.subheader("Trajectories of all cases in a range")
    st.caption("Every case whose first and last event lie between the two dates, one row each. "
               "With more cases than the number set, the ones that started first.")
    range_inputs(f"{p}_sel_traj", log)
    count_column, button_column = st.columns([1, 4], vertical_alignment="bottom")
    count_column.number_input("Trajectories", min_value=1, step=100, key=f"{p}_sel_traj_count")
    if button_column.button("Plot trajectories", icon=":material/view_timeline:"):
        start_date, end_date = picked_range(f"{p}_sel_traj")
        max_cases = int(st.session_state[f"{p}_sel_traj_count"])
        with st.spinner("Following the cases through their states…"):
            if computed == "SOM":
                trajectories = kairo.get_som_trajectories(states, start_date, end_date, max_cases)
                fig = kairo.plot_som_trajectories(trajectories, colors)
            elif computed == "k-means":
                trajectories = kairo.get_kmeans_trajectories(states, start_date, end_date, max_cases)
                fig = kairo.plot_kmeans_trajectories(trajectories, colors)
            else:
                trajectories = kairo.get_dbscan_trajectories(states, start_date, end_date, max_cases)
                fig = kairo.plot_dbscan_trajectories(trajectories, colors)
        st.session_state[f"{p}_range_trajectories"] = trajectories
        st.session_state[f"{p}_plot_trajectories"] = fig
    ui.show_plot(f"{p}_plot_trajectories", "Pick the dates and plot the trajectories.")


def log_trajectory(p: str, log: pd.DataFrame, states: pd.DataFrame, computed: str, colors: dict) -> None:
    st.subheader("Trajectory of the log")
    st.caption("The state of every calendar window between the two dates, left open for the whole log. "
               "Consecutive windows in the same state are one visit.")
    range_inputs(f"{p}_sel_log_traj", log)
    if st.button("Plot trajectory", icon=":material/view_timeline:"):
        start_date, end_date = picked_range(f"{p}_sel_log_traj")
        if computed == "SOM":
            trajectory = kairo.get_som_log_trajectory(states, start_date, end_date)
        elif computed == "k-means":
            trajectory = kairo.get_kmeans_log_trajectory(states, start_date, end_date)
        else:
            trajectory = kairo.get_dbscan_log_trajectory(states, start_date, end_date)

        if trajectory.empty:
            st.warning("No windows start between the two dates.")
        else:
            if computed == "SOM":
                fig = kairo.plot_som_log_trajectory(trajectory, colors)
            elif computed == "k-means":
                fig = kairo.plot_kmeans_log_trajectory(trajectory, colors)
            else:
                fig = kairo.plot_dbscan_log_trajectory(trajectory, colors)
            st.session_state[f"{p}_log_trajectory"] = trajectory
            st.session_state[f"{p}_plot_log_trajectory"] = fig
    ui.show_plot(f"{p}_plot_log_trajectory", "Pick the dates, or leave them open, and plot the trajectory.")

    trajectory = st.session_state.get(f"{p}_log_trajectory")
    if trajectory is not None:
        st.markdown("**Transitions of the log**")
        st.caption("One row per visit of a state. The log moves into every row's state at its start.")
        st.dataframe(trajectory.assign(duration=trajectory["duration"].astype(str)), width="stretch", hide_index=True)
