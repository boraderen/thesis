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
seed_widget("intra_sel_start", None)
seed_widget("intra_sel_end", None)
seed_widget("intra_sel_case", "")

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

    st.subheader("Frequencies")
    # value=None, the default of today would have to lie within the dates of the log
    start, end = st.columns(2)
    start.date_input("From", value=None, min_value=first_day, max_value=last_day, key="intra_sel_start")
    end.date_input("To", value=None, min_value=first_day, max_value=last_day, key="intra_sel_end")

    st.subheader("Trajectory")
    st.text_input("Case id", key="intra_sel_case")


def compute_states() -> dict:
    """The fitted model and the state of every event, for the method in the sidebar."""
    meta = compressed[META]

    if method == "SOM":
        size = (int(st.session_state["intra_sel_rows"]), int(st.session_state["intra_sel_cols"]))
        distance = st.session_state["intra_sel_som_distance"]
        model = kairo.compute_som(compressed, size, float(st.session_state["intra_sel_learning_rate"]), distance)
        states = meta.join(kairo.get_som_winners(compressed, model))
        colors = kairo.compute_som_color_mapping(size)
        distances = kairo.get_som_state_distances(model, distance)
    elif method == "k-means":
        k = int(st.session_state["intra_sel_k"])
        model = kairo.compute_kmeans(compressed, k)
        states = meta.join(kairo.get_kmeans_clusters(compressed, model))
        colors = kairo.compute_kmeans_color_mapping(k)
        distances = kairo.get_kmeans_state_distances(model)
    else:
        model = kairo.compute_dbscan(compressed, float(st.session_state["intra_sel_eps"]),
                                     int(st.session_state["intra_sel_min_samples"]),
                                     st.session_state["intra_sel_dbscan_distance"])
        states = meta.join(kairo.get_dbscan_clusters(compressed, model))
        colors = kairo.compute_dbscan_color_mapping(model)
        distances = kairo.get_dbscan_state_distances(model)

    return {"intra_method": method, "intra_model": model, "intra_states": states,
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
    ("Noise", f"{(states['cluster'] == -1).mean():.1%}" if computed == "DBSCAN" else "—"),
])
if computed != method:
    st.caption(f"These are the {computed} states — compute again to switch to {method}.")

start_date = st.session_state["intra_sel_start"]
end_date = st.session_state["intra_sel_end"]
# the end day counts as a whole
start_date = str(start_date) if start_date else None
end_date = f"{end_date} 23:59:59" if end_date else None

left, right = st.columns(2)
with left:
    st.subheader("State colors")
    if st.button("Plot state colors", icon=":material/palette:"):
        if computed == "SOM":
            fig = kairo.plot_som_colors(states, colors)
        elif computed == "k-means":
            fig = kairo.plot_kmeans_colors(states, colors)
        else:
            fig = kairo.plot_dbscan_colors(states, colors)
        st.session_state["intra_plot_colors"] = fig
    ui.show_plot("intra_plot_colors", "The color of every state, with how many events it holds.")

with right:
    st.subheader("Frequencies")
    if st.button("Plot frequencies", icon=":material/bar_chart:"):
        if computed == "SOM":
            fig = kairo.plot_som_heatmap(states, model.get_weights().shape[:2], start_date, end_date)
        elif computed == "k-means":
            fig = kairo.plot_kmeans_frequencies(states, colors, start_date, end_date)
        else:
            fig = kairo.plot_dbscan_frequencies(states, colors, start_date, end_date)
        st.session_state["intra_plot_frequencies"] = fig
    ui.show_plot("intra_plot_frequencies", "Events per state, between the dates in the sidebar.")

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

st.subheader("Case trajectory")
if st.button("Compute trajectory", icon=":material/route:"):
    case_id = st.session_state["intra_sel_case"].strip()
    if not case_id:
        st.warning("Enter a case id in the sidebar.")
    else:
        try:
            if computed == "SOM":
                visits = kairo.get_som_case_trajectory(states, case_id)
                fig = kairo.plot_som_case_trajectory(states, colors, case_id)
            elif computed == "k-means":
                visits = kairo.get_kmeans_case_trajectory(states, case_id)
                fig = kairo.plot_kmeans_case_trajectory(states, colors, case_id)
            else:
                visits = kairo.get_dbscan_case_trajectory(states, case_id)
                fig = kairo.plot_dbscan_case_trajectory(states, colors, case_id)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state.setdefault("intra_trajectories", {})[case_id] = {"visits": visits, "plot": fig}
            st.session_state["intra_last_case"] = case_id

case_id = st.session_state.get("intra_last_case")
if case_id is None:
    st.caption("Enter a case id in the sidebar and compute its trajectory.")
else:
    trajectory = st.session_state["intra_trajectories"][case_id]
    st.plotly_chart(trajectory["plot"], width="stretch")

    st.markdown(f"**Transitions of case {case_id}**")
    st.caption("One row per visit of a state. The case moves into every row's state at its start.")
    visits = trajectory["visits"]
    st.dataframe(visits.assign(duration=visits["duration"].astype(str)), width="stretch", hide_index=True)
