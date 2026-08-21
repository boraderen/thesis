"""Resource states: windowed workload features → PCA → clustered states → drift."""
from __future__ import annotations

import streamlit as st

import cache
import kairo
import ui
from controls import log_signature, seed_multi, seed_widget
from kairo.schema import FEATURE_DESCRIPTIONS, RESOURCE, RESOURCE_FEATURES

PREFIX = "resource"
SCHEMA_VERSION = "resource_window_v3"
RESULT_KEYS = ("resource_result", "resource_run_config", "resource_log_signature")

log = ui.require_log()
ui.page_header("Resource states",
               "One state per calendar window, summarising workload, waits, and handovers "
               "across the resource pool.", log)
if RESOURCE not in log.columns:
    st.error("No resource column was mapped on the **Upload** page — this page is disabled.")
    st.stop()

available, disabled = kairo.schema.feature_availability(log.columns, "resource")
all_resources = sorted(log[RESOURCE].astype(str).dropna().unique().tolist())
all_activities = sorted(log["concept:name"].astype(str).dropna().unique().tolist())

current_signature = log_signature(log, SCHEMA_VERSION)
if (
    st.session_state.get("resource_log_signature") is not None
    and st.session_state.get("resource_log_signature") != current_signature
):
    for key in (*RESULT_KEYS, "resource_W_sel"):
        st.session_state.pop(key, None)

run_cfg = st.session_state.get("resource_run_config") or {}
default_kinds = run_cfg.get("kinds", available)
for kind in RESOURCE_FEATURES:
    seed_widget(f"resource_kind_{kind}", kind in default_kinds and kind in available)
seed_multi("resource_res_sel", run_cfg.get("resources", all_resources), all_resources)
seed_multi("resource_act_sel", run_cfg.get("activities", all_activities), all_activities)
ui.seed_common(PREFIX, run_cfg, log, grid_default=(2, 3))

with st.sidebar:
    st.header("Pipeline")
    st.subheader("1 · Features")
    picked_kinds = [
        kind for kind, label in RESOURCE_FEATURES.items()
        if st.checkbox(label, key=f"resource_kind_{kind}", disabled=kind in disabled,
                       help=disabled.get(kind) and f"needs: {disabled[kind]}")
    ]
    with st.expander("Feature glossary"):
        for kind, label in RESOURCE_FEATURES.items():
            st.markdown(f"**{label}:** {FEATURE_DESCRIPTIONS[kind]}")
    picked_resources = st.multiselect("Resources", options=all_resources, key="resource_res_sel")
    picked_activities = st.multiselect("Activities", options=all_activities, key="resource_act_sel")
    skip_pca, pca_k, scaling = ui.reduction_controls(PREFIX)
    controls = ui.clustering_controls(PREFIX)
    window_minutes = ui.window_control(PREFIX, log, "Window W")
    st.divider()
    run_pipeline = st.button("Run resource pipeline", width="stretch", type="primary",
                             icon=":material/play_arrow:")

if run_pipeline:
    if not picked_kinds or not picked_resources:
        st.warning("Select at least one feature kind and one resource.")
        st.stop()
    with st.spinner("Building features, fitting PCA, clustering states…"):
        try:
            fs = cache.build_features(
                log, "resource", features=tuple(picked_kinds), window_minutes=int(window_minutes),
                resources=tuple(picked_resources), activities=tuple(picked_activities),
            )
        except ValueError as exc:
            st.warning(str(exc))
            st.stop()
        if len(fs.matrix) < 2:
            st.warning("Window W is wider than the log span — pick a smaller W to get a trajectory.")
            st.stop()
        if not fs.columns:
            st.warning("The picked kinds, resources, and activities select no columns.")
            st.stop()
        reduced, pca = cache.reduce_matrix(fs.values(), skip_pca, pca_k or None, scaling)
        model = cache.cluster(reduced, controls["clustering"], None, ui.cluster_params(controls))
        window_starts = fs.index["window_start"]
        result = kairo.StateResult(
            perspective="resource",
            config=kairo.ResourceConfig(
                features=tuple(picked_kinds), resources=tuple(picked_resources),
                activities=tuple(picked_activities), window_minutes=int(window_minutes),
                skip_pca=skip_pca, pca_components=pca_k or None, scaling=scaling,
                clustering=controls["clustering"], metric=controls["metric"],
                grid=controls["grid"], som_init=controls["som_init"],
                n_clusters=controls["kmeans_k"], eps=controls["eps"],
                min_samples=controls["min_samples"],
            ),
            features=fs, pca=pca, reduced=reduced, states=model,
            trajectories=kairo.trajectories(fs, model),
            transitions=cache.find_transitions(window_starts, model.state_ids,
                                               tuple(model.labels), fs.matrix),
            distribution=cache.state_distribution(window_starts, model.state_ids,
                                                  model.n_states, int(window_minutes)),
            signal=cache.window_vector_shift(window_starts, reduced, "euclidean"),
        )
    st.session_state["resource_result"] = result
    st.session_state["resource_log_signature"] = current_signature
    st.session_state["resource_run_config"] = {
        **controls, "pca_k": int(pca_k), "skip_pca": skip_pca, "scaling": scaling,
        "window_minutes": int(window_minutes), "kinds": list(picked_kinds),
        "resources": list(picked_resources), "activities": list(picked_activities),
    }
    st.toast("Resource pipeline finished.", icon=":material/check_circle:")

if (
    any(key not in st.session_state for key in RESULT_KEYS)
    or st.session_state.get("resource_log_signature") != current_signature
):
    st.info("Set the sidebar controls and run the resource pipeline.")
    st.stop()

result: kairo.StateResult = st.session_state["resource_result"]
fs, model, pca = result.features, result.states, result.pca
ran = st.session_state["resource_run_config"]
ui.run_banner(PREFIX)
ui.metrics_row([
    ("Windows", f"{len(fs.matrix):,}"),
    ("Feature columns", f"{len(fs.columns)}"),
    ("Resources", f"{len(fs.meta['picked_resources'])}"),
    ("PCA →", "—" if pca is None else f"{pca.n_components}D"),
    ("States", f"{model.n_states}"),
])

tab_features, tab_states, tab_transitions, tab_drift = st.tabs(
    ["Features & PCA", "States & trajectory", "Transitions", "Drift signal"]
)

with tab_features:
    ui.render_feature_matrix(
        fs, fs.columns,
        f"{len(fs.matrix):,} windows × {len(fs.columns)} columns "
        f"(W={fs.meta['window_minutes']} min, {len(fs.meta['resources'])} resources)",
    )
    ui.render_pca(pca)
    if ran["clustering"] == "dbscan":
        ui.render_k_distance(result.reduced, ran["kdist_k"], ran["metric"])

with tab_states:
    col_l, col_r = st.columns(2)
    with col_l:
        ui.render_state_grid(model, PREFIX, ran["clustering"])
    with col_r:
        st.markdown("**Resource state trajectory**")
        fig = kairo.plot_trajectory(
            fs.index["window_start"], model.state_ids, model,
            title=f"State per {fs.meta['window_minutes']}-min window", window_ticks=True,
        )
        if not result.transitions.empty:
            kairo.add_transition_markers(fig, result.transitions["timestamp"])
        st.plotly_chart(fig, width="stretch")
    if ran["clustering"] == "som" and model.codebook is not None:
        st.plotly_chart(kairo.plot_state_distances(model, ran["metric"]), width="stretch")

with tab_transitions:
    ui.render_transitions(result.transitions, "No state changes in this trajectory.")

with tab_drift:
    st.markdown("**State vector distances**")
    ui.render_vector_shift(PREFIX, fs.index["window_start"], result.reduced)
