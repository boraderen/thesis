"""Intra-case states: cluster the compressed events into states, then follow cases through them."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import seed_widget
from kairo.analysis import META

METHODS = ["SOM", "k-means", "DBSCAN"]
DISTANCES = ["euclidean", "cosine", "manhattan", "chebyshev"]

ui.keep_widgets()
log = ui.intra_log()
compressed = ui.require("intra_compressed", "Apply PCA on the **PCA** page first.",
                        "views/intra/pca.py", "PCA")

seed_widget("intra_sel_method", "SOM")
seed_widget("intra_sel_rows", 5)
seed_widget("intra_sel_cols", 5)
seed_widget("intra_sel_learning_rate", 0.5)
seed_widget("intra_sel_som_distance", "euclidean")
seed_widget("intra_sel_k", 5)
seed_widget("intra_sel_eps", 0.5)
seed_widget("intra_sel_min_samples", 5)
seed_widget("intra_sel_dbscan_distance", "euclidean")
seed_widget("intra_sel_traj_count", 1000)
first_day = log["time:timestamp"].min().date()
last_day = log["time:timestamp"].max().date()

with st.sidebar:
    st.header("States")
    method = st.radio("Method", METHODS, key="intra_sel_method", horizontal=True)
    if method == "SOM":
        rows, cols = st.columns(2)
        rows.number_input("Grid rows", min_value=1, max_value=30, step=1, key="intra_sel_rows")
        cols.number_input("Grid columns", min_value=1, max_value=30, step=1, key="intra_sel_cols")
        st.number_input("Learning rate", min_value=0.01, max_value=1.0, step=0.05,
                        key="intra_sel_learning_rate")
        st.selectbox("Distance", DISTANCES, key="intra_sel_som_distance")
    elif method == "k-means":
        st.number_input("Clusters k", min_value=2, max_value=30, step=1, key="intra_sel_k")
    else:
        st.number_input("eps", min_value=0.001, step=0.05, format="%.3f", key="intra_sel_eps")
        st.number_input("Min samples", min_value=1, step=1, key="intra_sel_min_samples")
        st.selectbox("Distance", DISTANCES, key="intra_sel_dbscan_distance")


def range_inputs(key: str) -> None:
    # value=None, the default of today would have to lie within the dates of the log
    start, end = st.columns(2)
    start.date_input("From", value=None, min_value=first_day, max_value=last_day, key=f"{key}_start")
    end.date_input("To", value=None, min_value=first_day, max_value=last_day, key=f"{key}_end")


def picked_range(key: str) -> tuple:
    # the picked dates as text, with the end day counting as a whole
    start = st.session_state[f"{key}_start"]
    end = st.session_state[f"{key}_end"]

    return (str(start) if start else None, f"{end} 23:59:59" if end else None)


def compute_states() -> dict:
    """The fitted model and the state of every event, for the method in the sidebar."""
    meta = compressed[META]

    if method == "SOM":
        size = (int(st.session_state["intra_sel_rows"]), int(st.session_state["intra_sel_cols"]))
        distance = st.session_state["intra_sel_som_distance"]
        model = kairo.compute_som(compressed, size, float(st.session_state["intra_sel_learning_rate"]), distance)
        states = meta.join(kairo.get_som_winners(compressed, model))
        frequencies = kairo.get_som_state_frequencies(states)
        colors = kairo.compute_som_color_mapping(size)
        distances = kairo.get_som_state_distances(model, distance)
    elif method == "k-means":
        k = int(st.session_state["intra_sel_k"])
        model = kairo.compute_kmeans(compressed, k)
        states = meta.join(kairo.get_kmeans_clusters(compressed, model))
        frequencies = kairo.get_kmeans_state_frequencies(states)
        colors = kairo.compute_kmeans_color_mapping(k)
        distances = kairo.get_kmeans_state_distances(model)
    else:
        model = kairo.compute_dbscan(compressed, float(st.session_state["intra_sel_eps"]),
                                     int(st.session_state["intra_sel_min_samples"]),
                                     st.session_state["intra_sel_dbscan_distance"])
        states = meta.join(kairo.get_dbscan_clusters(compressed, model))
        frequencies = kairo.get_dbscan_state_frequencies(states)
        colors = kairo.compute_dbscan_color_mapping(model)
        distances = kairo.get_dbscan_state_distances(model)

    return {"intra_method": method, "intra_model": model, "intra_states": states, "intra_frequencies": frequencies,
            "intra_colors": colors, "intra_distances": distances}


st.title("States & Trajectories")
st.caption("Every event gets a state from its compressed features, then cases are followed through them.")

if method == "DBSCAN":
    st.subheader("k-distance curve")
    if st.button("Plot k-distance", icon=":material/show_chart:"):
        with st.spinner("Finding the neighbours of every event…"):
            st.session_state["intra_plot_kdistance"] = kairo.plot_dbscan_k_distance(
                compressed, int(st.session_state["intra_sel_min_samples"]),
                st.session_state["intra_sel_dbscan_distance"])
    ui.show_plot("intra_plot_kdistance",
                 "The knee of the curve is a good eps. Repeating prefixes keep it at 0 for most "
                 "events, so the knee sits at the far right end.")

st.subheader("States")
if st.button("Compute states", type="primary", icon=":material/play_arrow:"):
    with st.spinner(f"Computing the {method} states…"):
        try:
            result = compute_states()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    ui.clear_from("states")
    st.session_state.update(result)

states = st.session_state.get("intra_states")
if states is None:
    st.info("Pick a method and its parameters in the sidebar, then compute the states.")
    st.stop()

# from here on everything shows the computed states, even if the sidebar has moved on
computed = st.session_state["intra_method"]
model = st.session_state["intra_model"]
colors = st.session_state["intra_colors"]

ui.metrics_row([
    ("Method", computed),
    ("States", f"{len(st.session_state['intra_distances']):,}"),
    ("Events", f"{len(states):,}"),
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
            st.session_state["intra_plot_colors"] = kairo.plot_som_colors(states, colors)
        ui.show_plot("intra_plot_colors", "The color of every cell, with how many events it holds.")
else:
    distances_column = st.container()

with distances_column:
    st.subheader("Distances between states")
    if st.button("Plot distances", icon=":material/grid_on:"):
        if computed == "SOM":
            fig = kairo.plot_som_u_matrix(model)
        elif computed == "k-means":
            fig = kairo.plot_kmeans_distances(model)
        else:
            fig = kairo.plot_dbscan_distances(model)
        st.session_state["intra_plot_distances"] = fig
    ui.show_plot("intra_plot_distances",
                 "For a SOM the u-matrix, the distance of every neuron to its neighbours. "
                 "For k-means and DBSCAN the distance between every two clusters.")

st.subheader("Frequencies")
st.caption("Two date ranges side by side, e.g. before and after a suspected drift.")
for column, number in zip(st.columns(2), (1, 2)):
    with column:
        range_inputs(f"intra_sel_freq{number}")
        if st.button("Plot frequencies", icon=":material/bar_chart:", key=f"intra_freq_button_{number}"):
            start_date, end_date = picked_range(f"intra_sel_freq{number}")
            if computed == "SOM":
                fig = kairo.plot_som_heatmap(states, model.get_weights().shape[:2], start_date, end_date)
            elif computed == "k-means":
                fig = kairo.plot_kmeans_frequencies(states, colors, start_date, end_date)
            else:
                fig = kairo.plot_dbscan_frequencies(states, colors, start_date, end_date)
            st.session_state[f"intra_plot_frequencies_{number}"] = fig
        ui.show_plot(f"intra_plot_frequencies_{number}", "Events per state between the two dates.")

st.subheader("Case trajectory")
case_input, case_button = st.columns([3, 1], vertical_alignment="bottom")
case_input.text_input("Case id", key="intra_sel_case")
if case_button.button("Compute trajectory", icon=":material/route:", width="stretch"):
    case_id = st.session_state["intra_sel_case"].strip()
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
            st.session_state.setdefault("intra_trajectories", {})[case_id] = {"visits": visits, "plot": fig}
            st.session_state["intra_last_case"] = case_id

case_id = st.session_state.get("intra_last_case")
if case_id is None:
    st.caption("Enter a case id and compute its trajectory.")
else:
    trajectory = st.session_state["intra_trajectories"][case_id]
    st.plotly_chart(trajectory["plot"], width="stretch")

    st.markdown(f"**Transitions of case {case_id}**")
    st.caption("One row per visit of a state. The case moves into every row's state at its start.")
    visits = trajectory["visits"]
    st.dataframe(visits.assign(duration=visits["duration"].astype(str)), width="stretch", hide_index=True)

st.subheader("Trajectories of all cases in a range")
st.caption("Every case whose first and last event lie between the two dates, one row each. "
           "With more cases than the number set, the ones that started first.")
range_inputs("intra_sel_traj")
count_column, button_column = st.columns([1, 4], vertical_alignment="bottom")
count_column.number_input("Trajectories", min_value=1, step=100, key="intra_sel_traj_count")
if button_column.button("Plot trajectories", icon=":material/view_timeline:"):
    start_date, end_date = picked_range("intra_sel_traj")
    max_cases = int(st.session_state["intra_sel_traj_count"])
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
    st.session_state["intra_range_trajectories"] = trajectories
    st.session_state["intra_plot_trajectories"] = fig
ui.show_plot("intra_plot_trajectories", "Pick the dates and plot the trajectories.")
