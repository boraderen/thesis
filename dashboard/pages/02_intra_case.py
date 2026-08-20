"""Intra-case state page: prefix features → PCA → SOM / DBSCAN / k-means states → per-case trajectory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.controls import (
    INTRA_SCHEMA_VERSION,
    log_signature,
    seed_choice,
    seed_widget,
)
from core.dbscan import cluster_dbscan, k_distances
from core.distance import DISTANCE_LABELS, SUPPORTED_DISTANCES
from core.drift import intra_state_distribution
from core.features.intra_case import build_features
from core.kmeans import cluster_kmeans
from core.loader import span_label
from core.pca import SCALING_LABELS, fit_pca, standardize
from core.schema import INTRA_FEATURE_LABELS as GROUP_LABELS
from core.som import cell_distances, train_som
from core.state_attribution import (
    DIVERGENCE_LABELS,
    REFERENCE_LABELS,
    intra_state_shift,
)
from core.transitions import find_transitions
from core.windows import (
    as_window_minutes,
    default_window_minutes,
    log_span_minutes,
    window_minute_choices,
    window_minute_label,
)
from viz.drift_scores import score_line
from viz.drift_signal import add_window_boundaries, stacked_area_intra
from viz.som_grid import cell_distance_heatmap, som_heatmap
from viz.tables import styled_feature_table
from viz.trajectory import (
    add_transition_markers,
    k_distance_plot,
    pca_variance_plot,
    state_timeline,
)

st.set_page_config(page_title="Intra-case states", layout="wide")
st.title("2 — Intra-case states")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))

GROUP_DESCRIPTIONS = {
    "activity_freq": "One column per activity — how often the activity occurred in the case so far, divided by the number of events so far (the prefix).",
    "bigram": "One column per observed directly-follows pair A→B — how often that transition occurred in the prefix, divided by the number of transitions so far.",
    "vocab": "One binary column per activity — 1 if the activity has already occurred in the case prefix, 0 otherwise.",
    "progress": "Position of the event within its case as a fraction of the total case length (last event = 1).",
    "current": "One binary column per activity — 1 for the activity of this event.",
    "history": "For each of the last n events of the same case, one binary column per activity — 1 for that event's activity. Before the case has n predecessors, the missing slots stay 0.",
}

CLUSTERING_LABELS = {
    "som": "SOM",
    "dbscan": "DBSCAN",
    "kmeans": "k-means",
}

GRID_TITLES = {
    "som": "SOM grid",
    "dbscan": "DBSCAN clusters",
    "kmeans": "k-means clusters",
}

INIT_LABELS = {
    "random": "Random samples",
    "pca": "PCA plane",
}

INTRA_RESULT_KEYS = (
    "intra_feat",
    "intra_spec",
    "intra_som",
    "intra_pca",
    "intra_reduced",
    "intra_selected_cols",
    "intra_run_config",
    "intra_log_signature",
)

current_log_signature = log_signature(log, INTRA_SCHEMA_VERSION)
if (
    st.session_state.get("intra_log_signature") is not None
    and st.session_state.get("intra_log_signature") != current_log_signature
):
    # Window choices depend on the log span, so the W widget re-seeds too.
    for key in (*INTRA_RESULT_KEYS, "intra_W_sel"):
        st.session_state.pop(key, None)

run_cfg = st.session_state.get("intra_run_config") or {}
default_groups = run_cfg.get("groups", list(GROUP_LABELS))
for group in GROUP_LABELS:
    seed_widget(f"intra_group_{group}", group in default_groups)
seed_widget("intra_history_sel", int(run_cfg.get("history", 3)))
seed_widget("intra_skip_pca_sel", bool(run_cfg.get("skip_pca", False)))
seed_widget("intra_pca_k_sel", int(run_cfg.get("pca_k", 0)))
seed_choice("intra_scaling_sel", run_cfg.get("scaling", "none"), tuple(SCALING_LABELS))
seed_choice("intra_cluster_sel", run_cfg.get("clustering", "som"), tuple(CLUSTERING_LABELS))
seed_choice("intra_init_sel", run_cfg.get("som_init", "random"), tuple(INIT_LABELS))
seed_choice(
    "intra_metric_sel",
    run_cfg.get("metric", "euclidean"),
    SUPPORTED_DISTANCES[st.session_state["intra_cluster_sel"]],
)
seed_widget("intra_eps_sel", float(run_cfg.get("eps", 0.5)))
seed_widget("intra_minpts_sel", int(run_cfg.get("min_samples", 5)))
seed_widget("intra_kdist_sel", int(run_cfg.get("kdist_k", 5)))
seed_widget("intra_lookback_sel", 5)  # l for the rolling-reference divergence
seed_widget("intra_kmeans_k_sel", int(run_cfg.get("kmeans_k", 6)))
grid_default = tuple(run_cfg.get("grid", (3, 3)))
seed_widget("intra_grid_h_sel", int(grid_default[0]))
seed_widget("intra_grid_w_sel", int(grid_default[1]))
seed_widget(
    "intra_W_sel",
    int(run_cfg.get("distribution_W", default_window_minutes(log_span_minutes(log)))),
)
# The window selectbox takes hand-typed values, which arrive as strings.
st.session_state["intra_W_sel"] = as_window_minutes(
    st.session_state["intra_W_sel"], default_window_minutes(log_span_minutes(log))
)

with st.sidebar:
    st.header("Controls")
    st.subheader("Select features")
    picked = [
        group
        for group, label in GROUP_LABELS.items()
        if st.checkbox(label, key=f"intra_group_{group}")
    ]
    with st.expander("Feature glossary"):
        for group, label in GROUP_LABELS.items():
            st.markdown(f"**{label}:** {GROUP_DESCRIPTIONS[group]}")
    history = st.slider("Past activities window n (events)", min_value=1, max_value=20, step=1,
                        key="intra_history_sel")

    st.subheader("Dimensionality reduction")
    skip_pca = st.checkbox("Skip PCA", key="intra_skip_pca_sel")
    if not skip_pca:
        st.number_input("PCA components (0 = auto)", min_value=0, max_value=20, step=1,
                        key="intra_pca_k_sel")
    scaling = st.radio(
        "Scaling",
        options=tuple(SCALING_LABELS),
        key="intra_scaling_sel",
        format_func=lambda s: SCALING_LABELS[s],
    )

    st.subheader("Clustering")
    clustering = st.radio(
        "Method",
        options=tuple(CLUSTERING_LABELS),
        key="intra_cluster_sel",
        format_func=lambda m: CLUSTERING_LABELS[m],
    )
    metric = st.selectbox(
        "Distance",
        SUPPORTED_DISTANCES[clustering],
        key="intra_metric_sel",
        format_func=lambda d: DISTANCE_LABELS[d],
    )
    if clustering == "som":
        col_grid_h, col_grid_w = st.columns(2)
        col_grid_h.number_input("SOM grid height", min_value=1, max_value=50, step=1,
                                key="intra_grid_h_sel")
        col_grid_w.number_input("SOM grid width", min_value=1, max_value=50, step=1,
                                key="intra_grid_w_sel")
        st.radio("SOM init", options=tuple(INIT_LABELS), key="intra_init_sel",
                 format_func=lambda i: INIT_LABELS[i])
    elif clustering == "dbscan":
        st.number_input("DBSCAN eps", min_value=0.05, max_value=100.0, step=0.05,
                        key="intra_eps_sel")
        st.number_input("DBSCAN min samples", min_value=1, step=1,
                        key="intra_minpts_sel")
        st.number_input("k for the k-distance curve", min_value=1, step=1,
                        key="intra_kdist_sel")
    else:
        st.number_input("k-means clusters", min_value=2, max_value=25, step=1,
                        key="intra_kmeans_k_sel")

    st.subheader("Windows")
    distribution_W = as_window_minutes(
        st.selectbox(
            "Distribution window W",
            window_minute_choices(st.session_state["intra_W_sel"]),
            key="intra_W_sel",
            format_func=window_minute_label,
            accept_new_options=True,
        ),
        default_window_minutes(log_span_minutes(log)),
    )
    run_pipeline = st.button("Run intra-case pipeline", width="stretch", type="primary")

# The parameters of the unselected clustering methods are not rendered, so they
# are read from their seeded slots rather than from a widget's return value.
pca_k = int(st.session_state["intra_pca_k_sel"])
grid_h = int(st.session_state["intra_grid_h_sel"])
grid_w = int(st.session_state["intra_grid_w_sel"])
som_init = st.session_state["intra_init_sel"]
eps = float(st.session_state["intra_eps_sel"])
min_samples = int(st.session_state["intra_minpts_sel"])
kdist_k = int(st.session_state["intra_kdist_sel"])
kmeans_k = int(st.session_state["intra_kmeans_k_sel"])

if run_pipeline:
    if not picked:
        st.warning("Select at least one feature group.")
        st.stop()
    with st.spinner("Building features, reducing dimensions, clustering states…"):
        feat, spec = build_features(log, history=int(history))
        selected_cols = [c for g in picked for c in spec.groups[g]]
        matrix = feat[selected_cols].to_numpy()
        # PCA is optional; standardization, when asked for, applies to whatever
        # goes into the clustering — the raw columns, or the PCA output.
        pca = None if skip_pca else fit_pca(matrix, force_k=pca_k or None)
        reduced = matrix if pca is None else pca.transformed
        if scaling == "standardize":
            reduced = standardize(reduced)
        annotations = tuple(feat["concept:name"].astype(str).tolist())
        if clustering == "dbscan":
            som = cluster_dbscan(
                reduced, eps=float(eps), min_samples=int(min_samples), annotations=annotations,
                metric=metric
            )
        elif clustering == "kmeans":
            som = cluster_kmeans(
                reduced, n_clusters=int(kmeans_k), annotations=annotations, metric=metric
            )
        else:
            som = train_som(
                reduced, grid_h=grid_h, grid_w=grid_w, annotations=annotations,
                init=som_init, metric=metric
            )
        feat = feat.assign(state_id=som.state_ids)
    st.session_state["intra_feat"] = feat
    st.session_state["intra_spec"] = spec
    st.session_state["intra_pca"] = pca
    st.session_state["intra_reduced"] = reduced
    st.session_state["intra_som"] = som
    st.session_state["intra_selected_cols"] = selected_cols
    st.session_state["intra_log_signature"] = current_log_signature
    st.session_state["intra_run_config"] = {
        "grid": (grid_h, grid_w),
        "clustering": clustering,
        "som_init": som_init,
        "metric": metric,
        "eps": float(eps),
        "min_samples": int(min_samples),
        "kdist_k": kdist_k,
        "kmeans_k": int(kmeans_k),
        "pca_k": int(pca_k),
        "skip_pca": skip_pca,
        "scaling": scaling,
        "distribution_W": int(distribution_W),
        "groups": list(picked),
        "history": int(history),
    }

if (
    any(key not in st.session_state for key in INTRA_RESULT_KEYS)
    or st.session_state.get("intra_log_signature") != current_log_signature
):
    st.info("Set the sidebar controls and run the intra-case pipeline.")
    st.stop()

feat = st.session_state["intra_feat"]
spec = st.session_state["intra_spec"]
pca = st.session_state["intra_pca"]
reduced = st.session_state["intra_reduced"]
som = st.session_state["intra_som"]
selected_cols = st.session_state["intra_selected_cols"]
distribution_W = st.session_state["intra_run_config"]["distribution_W"]
ran_clustering = st.session_state["intra_run_config"].get("clustering", "som")
ran_metric = st.session_state["intra_run_config"].get("metric", "euclidean")

st.subheader("Feature matrix")
st.caption(
    f"{len(feat):,} events × {len(selected_cols)} feature columns "
    f"(|A|={len(spec.activities)}, |A→B|={len(spec.transitions)})"
)
preview_cols = ["case:concept:name", "concept:name", "time:timestamp", *selected_cols]
preview_groups = {g: [c for c in cols if c in selected_cols] for g, cols in spec.groups.items()}
styled = styled_feature_table(feat[preview_cols], preview_groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

if pca is not None:
    st.subheader("PCA")
    st.plotly_chart(
        pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim),
        width="stretch",
    )

# The knee of the k-distance curve is read as the candidate DBSCAN eps.
if ran_clustering == "dbscan":
    st.subheader("k-distance curve")
    st.plotly_chart(
        k_distance_plot(k_distances(reduced, kdist_k, ran_metric), kdist_k), width="stretch"
    )

col_l, col_r = st.columns([1, 1])
with col_l:
    st.subheader(GRID_TITLES.get(ran_clustering, "States"))
    grid_view = st.radio(
        "Grid view", ("State colors", "Frequency"),
        horizontal=True, label_visibility="collapsed", key="intra_grid_style",
    )
    st.plotly_chart(
        som_heatmap(
            som.grid_h, som.grid_w, som.cell_counts, som.cell_labels,
            title=f"{CLUSTERING_LABELS[ran_clustering]} states",
            dominants=som.cell_dominant,
            monochrome=grid_view == "Frequency",
        ),
        width="stretch",
    )
with col_r:
    st.subheader("Case trajectory")
    case_ids = feat["case:concept:name"].drop_duplicates().tolist()
    chosen = st.selectbox("Case", case_ids, index=0)
    sub = feat[feat["case:concept:name"] == chosen].reset_index(drop=True)
    transitions = find_transitions(
        sub["time:timestamp"], sub["state_id"].to_numpy(), som.cell_labels, sub[selected_cols]
    )
    fig = state_timeline(
        sub["time:timestamp"], sub["state_id"].to_numpy(), som.cell_labels,
        title=f"Case {chosen}", cell_dominant=som.cell_dominant,
    )
    if not transitions.empty:
        add_transition_markers(fig, transitions["timestamp"])
    st.plotly_chart(fig, width="stretch")

# The SOM cell spacing, read as the scale at which its states separate.
codebook = getattr(som, "codebook", None)
if ran_clustering == "som" and codebook is not None:
    st.subheader("Cell distances")
    st.plotly_chart(
        cell_distance_heatmap(cell_distances(codebook, ran_metric), som.cell_labels),
        width="stretch",
    )

st.subheader("Transitions")
if transitions.empty:
    st.caption("No state changes in this case.")
else:
    st.caption(f"{len(transitions)} transitions in case {chosen}.")
    st.dataframe(
        transitions[["timestamp", "from", "to", "top_changes"]].rename(columns={"timestamp": "at"}),
        width="stretch", height=min(420, 60 + 36 * len(transitions)),
    )

st.subheader("State frequency distribution over time")
n_states = som.grid_h * som.grid_w
intra_dist = intra_state_distribution(feat, n_states=n_states, window_minutes=distribution_W)
st.caption(
    f"Per {window_minute_label(distribution_W)} window — what fraction of events landed in each state. "
    f"{len(intra_dist):,} windows across all {feat['case:concept:name'].nunique():,} cases."
)
intra_cols = [f"intra_S{i}" for i in range(n_states)]
freq_fig = stacked_area_intra(intra_dist, intra_cols, som.cell_labels)
add_window_boundaries(freq_fig, intra_dist["window_start"])
st.plotly_chart(freq_fig, width="stretch")

st.subheader("Freq. distri divergences")
pick_div, pick_ref, pick_l = st.columns([1, 1, 1])
divergence = pick_div.selectbox(
    "Divergence",
    tuple(DIVERGENCE_LABELS),
    key="intra_divergence_sel",
    format_func=lambda d: DIVERGENCE_LABELS[d],
)
reference = pick_ref.selectbox(
    "Compare against",
    tuple(REFERENCE_LABELS),
    key="intra_reference_sel",
    format_func=lambda r: REFERENCE_LABELS[r],
)
lookback = (
    pick_l.number_input("Windows to average (l)", min_value=1, step=1, key="intra_lookback_sel")
    if reference == "recent"
    else st.session_state.get("intra_lookback_sel", 5)
)
REFERENCE_PHRASES = {
    "previous": "window i−1",
    "recent": f"mean of the {int(lookback)} windows before i",
    "baseline": "full-log baseline",
}
st.caption("Each window's state distribution compared with its reference.")
shift = intra_state_shift(intra_dist, divergence, reference, int(lookback))
shift_fig = score_line(
    shift,
    "score",
    title=f"{DIVERGENCE_LABELS[divergence]}: window i vs. {REFERENCE_PHRASES[reference]}",
)
add_window_boundaries(
    shift_fig, shift["window_start"], y_max=max(float(shift["score"].max()), 1e-9)
)
st.plotly_chart(shift_fig, width="stretch")
