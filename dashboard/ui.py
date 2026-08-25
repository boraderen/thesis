"""Shared chrome and controls for the three pipeline pages.

Everything here renders Streamlit widgets or figures; all computation lives in
kairo (memoized through cache.py where it runs on every rerun).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import cache
import kairo
from controls import seed_choice, seed_widget
from tables import styled_feature_table

CLUSTERING_LABELS = {"som": "SOM", "dbscan": "DBSCAN", "kmeans": "k-means"}
GRID_TITLES = {"som": "SOM grid", "dbscan": "DBSCAN clusters", "kmeans": "k-means clusters"}


# ---------------------------------------------------------------------------
# Page scaffolding
# ---------------------------------------------------------------------------

def require_log() -> pd.DataFrame:
    """The mapped log from session state, or a friendly stop."""
    if "log" not in st.session_state:
        st.info("No event log loaded yet — start on the **Upload log** page.")
        st.page_link("views/upload.py", label="Go to Upload", icon=":material/upload_file:")
        st.stop()
    return st.session_state["log"]


def page_header(title: str, subtitle: str, log: pd.DataFrame | None = None) -> None:
    st.title(title)
    caption = subtitle
    if log is not None:
        caption += "  \n" + kairo.data.span_label(log)
    st.caption(caption)


def run_banner(prefix: str) -> None:
    """One-line summary of the stored run's configuration, as chips."""
    cfg = st.session_state.get(f"{prefix}_run_config")
    if not cfg:
        return
    chips = [f"**{CLUSTERING_LABELS.get(cfg.get('clustering'), cfg.get('clustering'))}**"]
    if cfg.get("clustering") == "som":
        h, w = cfg.get("grid", (0, 0))
        chips.append(f"grid {h}×{w}")
    elif cfg.get("clustering") == "kmeans":
        chips.append(f"k={cfg.get('kmeans_k')}")
    else:
        chips.append(f"eps={cfg.get('eps')}, minPts={cfg.get('min_samples')}")
    chips.append("no PCA" if cfg.get("skip_pca") else f"PCA k={cfg.get('pca_k') or 'auto'}")
    chips.append(f"metric {cfg.get('metric')}")
    chips.append(f"W {kairo.data.window_minute_label(int(cfg.get('window_minutes', 0)))}")
    st.caption("Last run · " + " · ".join(chips))


def metrics_row(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


# ---------------------------------------------------------------------------
# Sidebar control groups (all widgets keyed and pre-seeded)
# ---------------------------------------------------------------------------

def seed_common(prefix: str, run_cfg: dict, log: pd.DataFrame, grid_default: tuple[int, int]) -> None:
    """Seed every widget slot the three pipelines share."""
    seed_widget(f"{prefix}_skip_pca_sel", bool(run_cfg.get("skip_pca", False)))
    seed_widget(f"{prefix}_pca_k_sel", int(run_cfg.get("pca_k", 0)))
    seed_choice(f"{prefix}_scaling_sel", run_cfg.get("scaling", "none"), tuple(kairo.SCALING))
    seed_choice(f"{prefix}_cluster_sel", run_cfg.get("clustering", "som"), tuple(CLUSTERING_LABELS))
    seed_choice(f"{prefix}_init_sel", run_cfg.get("som_init", "random"), tuple(kairo.SOM_INIT))
    seed_choice(
        f"{prefix}_metric_sel",
        run_cfg.get("metric", "euclidean"),
        kairo.SUPPORTED_DISTANCES[st.session_state[f"{prefix}_cluster_sel"]],
    )
    seed_widget(f"{prefix}_eps_sel", float(run_cfg.get("eps", 0.5)))
    seed_widget(f"{prefix}_minpts_sel", int(run_cfg.get("min_samples", 5)))
    seed_widget(f"{prefix}_kdist_sel", int(run_cfg.get("kdist_k", 5)))
    seed_widget(f"{prefix}_kmeans_k_sel", int(run_cfg.get("kmeans_k", 6)))
    grid = tuple(run_cfg.get("grid", grid_default))
    seed_widget(f"{prefix}_grid_h_sel", int(grid[0]))
    seed_widget(f"{prefix}_grid_w_sel", int(grid[1]))
    seed_widget(
        f"{prefix}_W_sel",
        int(run_cfg.get("window_minutes",
                        kairo.data.default_window_minutes(kairo.data.log_span_minutes(log)))),
    )
    # The window selectbox takes hand-typed values, which arrive as strings.
    st.session_state[f"{prefix}_W_sel"] = kairo.data.as_window_minutes(
        st.session_state[f"{prefix}_W_sel"],
        kairo.data.default_window_minutes(kairo.data.log_span_minutes(log)),
    )


def reduction_controls(prefix: str) -> tuple[bool, int, str]:
    st.subheader("2 · Reduction")
    skip_pca = st.toggle("Skip PCA", key=f"{prefix}_skip_pca_sel")
    if not skip_pca:
        st.number_input("PCA components (0 = auto)", min_value=0, max_value=20, step=1,
                        key=f"{prefix}_pca_k_sel")
    scaling = st.radio("Scaling", options=tuple(kairo.SCALING), key=f"{prefix}_scaling_sel",
                       format_func=lambda s: kairo.SCALING[s], horizontal=True)
    return skip_pca, int(st.session_state[f"{prefix}_pca_k_sel"]), scaling


def clustering_controls(prefix: str) -> dict:
    st.subheader("3 · Clustering")
    clustering = st.radio("Method", options=tuple(CLUSTERING_LABELS), key=f"{prefix}_cluster_sel",
                          format_func=lambda m: CLUSTERING_LABELS[m], horizontal=True)
    metric = st.selectbox("Distance", kairo.SUPPORTED_DISTANCES[clustering],
                          key=f"{prefix}_metric_sel", format_func=lambda d: kairo.DISTANCES[d])
    if clustering == "som":
        col_h, col_w = st.columns(2)
        col_h.number_input("Grid height", min_value=1, max_value=50, step=1, key=f"{prefix}_grid_h_sel")
        col_w.number_input("Grid width", min_value=1, max_value=50, step=1, key=f"{prefix}_grid_w_sel")
        st.radio("SOM init", options=tuple(kairo.SOM_INIT), key=f"{prefix}_init_sel",
                 format_func=lambda i: kairo.SOM_INIT[i], horizontal=True)
    elif clustering == "dbscan":
        st.number_input("eps", min_value=0.05, max_value=100.0, step=0.05, key=f"{prefix}_eps_sel")
        st.number_input("min samples", min_value=1, step=1, key=f"{prefix}_minpts_sel")
        st.number_input("k for the k-distance curve", min_value=1, step=1, key=f"{prefix}_kdist_sel")
    else:
        st.number_input("Clusters k", min_value=2, max_value=25, step=1, key=f"{prefix}_kmeans_k_sel")
    return {
        "clustering": clustering,
        "metric": metric,
        "grid": (int(st.session_state[f"{prefix}_grid_h_sel"]),
                 int(st.session_state[f"{prefix}_grid_w_sel"])),
        "som_init": st.session_state[f"{prefix}_init_sel"],
        "eps": float(st.session_state[f"{prefix}_eps_sel"]),
        "min_samples": int(st.session_state[f"{prefix}_minpts_sel"]),
        "kdist_k": int(st.session_state[f"{prefix}_kdist_sel"]),
        "kmeans_k": int(st.session_state[f"{prefix}_kmeans_k_sel"]),
    }


def window_control(prefix: str, log: pd.DataFrame, label: str) -> int:
    st.subheader("4 · Windows")
    return kairo.data.as_window_minutes(
        st.selectbox(
            label,
            kairo.data.window_minute_choices(st.session_state[f"{prefix}_W_sel"]),
            key=f"{prefix}_W_sel",
            format_func=kairo.data.window_minute_label,
            accept_new_options=True,
        ),
        kairo.data.default_window_minutes(kairo.data.log_span_minutes(log)),
    )


def cluster_params(controls: dict) -> dict:
    """The kairo cluster() parameters for the chosen method."""
    method = controls["clustering"]
    if method == "som":
        return {"grid": controls["grid"], "metric": controls["metric"],
                "init": controls["som_init"], "seed": 7}
    if method == "kmeans":
        return {"n_clusters": controls["kmeans_k"], "metric": controls["metric"], "seed": 7}
    return {"eps": controls["eps"], "min_samples": controls["min_samples"],
            "metric": controls["metric"]}


# ---------------------------------------------------------------------------
# Result sections
# ---------------------------------------------------------------------------

def render_feature_matrix(fs: kairo.features.FeatureSet, selected: list[str], caption: str) -> None:
    st.caption(caption)
    preview = fs.frame[[*fs.index.columns, *selected]]
    groups = {g: [c for c in cols if c in selected] for g, cols in fs.groups.items()}
    st.dataframe(styled_feature_table(preview, groups), width="stretch", height=380)


def render_pca(pca) -> None:
    if pca is None:
        st.caption("PCA was skipped — clustering ran on the raw feature columns.")
        return
    st.plotly_chart(kairo.plot_pca_variance(pca), width="stretch")


def render_k_distance(reduced, kdist_k: int, metric: str) -> None:
    st.plotly_chart(
        kairo.plot_k_distance(cache.k_distances(reduced, kdist_k, metric), kdist_k),
        width="stretch",
    )
    st.caption("The knee of this curve is the usual candidate for DBSCAN's eps.")


def render_state_grid(model, prefix: str, method: str) -> None:
    st.markdown(f"**{GRID_TITLES.get(method, 'States')}**")
    view = st.radio("Grid view", ("State colors", "Frequency"), horizontal=True,
                    label_visibility="collapsed", key=f"{prefix}_grid_style")
    st.plotly_chart(
        kairo.plot_state_grid(model, monochrome=view == "Frequency",
                              title=f"{CLUSTERING_LABELS[method]} states"),
        width="stretch",
    )


def render_transitions(transitions: pd.DataFrame, empty_text: str) -> None:
    if transitions.empty:
        st.caption(empty_text)
        return
    st.caption(f"{len(transitions)} transitions.")
    show = transitions[["timestamp", "from", "to", "top_changes"]].head(500)
    st.dataframe(show.rename(columns={"timestamp": "at"}), width="stretch",
                 height=min(420, 60 + 36 * len(show)))


REFERENCE_PHRASES = {
    "previous": "window i−1",
    "recent": "mean of the {l} windows before i",
    "baseline": "full-log baseline",
}


def render_vector_shift(prefix: str, window_starts: pd.Series, reduced) -> None:
    """The window-to-window vector-distance drift signal, with distance and reference pickers."""
    seed_widget(f"{prefix}_shift_lookback_sel", 5)
    pick_metric, pick_ref, pick_l = st.columns(3)
    metric = pick_metric.selectbox("Distance", tuple(kairo.DISTANCES),
                                   key=f"{prefix}_shift_metric_sel",
                                   format_func=lambda d: kairo.DISTANCES[d])
    reference = pick_ref.selectbox("Compare against", tuple(kairo.REFERENCES),
                                   key=f"{prefix}_shift_reference_sel",
                                   format_func=lambda r: kairo.REFERENCES[r])
    lookback = (
        pick_l.number_input("Windows to average (l)", min_value=1, step=1,
                            key=f"{prefix}_shift_lookback_sel")
        if reference == "recent" else st.session_state.get(f"{prefix}_shift_lookback_sel", 5)
    )
    phrase = REFERENCE_PHRASES[reference].format(l=int(lookback))
    st.caption(f"Each window's compressed state vector compared with the {phrase}.")
    shift = cache.window_vector_shift(window_starts, reduced, metric, reference, int(lookback))
    fig = kairo.plot_drift_signal(
        shift, title=f"{kairo.DISTANCES[metric]} distance: window i vs. {phrase}")
    kairo.add_window_boundaries(fig, shift["window_start"],
                                y_max=max(float(shift["score"].max()), 1e-9))
    st.plotly_chart(fig, width="stretch")
