"""Intra-case states: prefix features → PCA → clustered states → trajectories → drift."""
from __future__ import annotations

import streamlit as st

import cache
import kairo
import ui
from controls import log_signature, seed_widget
from kairo.schema import FEATURE_DESCRIPTIONS, INTRA_FEATURES

PREFIX = "intra"
SCHEMA_VERSION = "intra_prefix_v2"
RESULT_KEYS = ("intra_result", "intra_run_config", "intra_log_signature")

log = ui.require_log()
ui.page_header("Intra-case states",
               "One state per event, derived from its case prefix — then per-case "
               "trajectories and a divergence-based drift signal.", log)

current_signature = log_signature(log, SCHEMA_VERSION)
if (
    st.session_state.get("intra_log_signature") is not None
    and st.session_state.get("intra_log_signature") != current_signature
):
    for key in (*RESULT_KEYS, "intra_W_sel"):
        st.session_state.pop(key, None)

run_cfg = st.session_state.get("intra_run_config") or {}
default_groups = run_cfg.get("groups", list(INTRA_FEATURES))
for group in INTRA_FEATURES:
    seed_widget(f"intra_group_{group}", group in default_groups)
seed_widget("intra_history_sel", int(run_cfg.get("history", 3)))
seed_widget("intra_lookback_sel", 5)
ui.seed_common(PREFIX, run_cfg, log, grid_default=(3, 3))

with st.sidebar:
    st.header("Pipeline")
    st.subheader("1 · Features")
    picked = [
        group for group, label in INTRA_FEATURES.items()
        if st.checkbox(label, key=f"intra_group_{group}")
    ]
    with st.expander("Feature glossary"):
        for group, label in INTRA_FEATURES.items():
            st.markdown(f"**{label}:** {FEATURE_DESCRIPTIONS[group]}")
    history = st.slider("Past activities window n (events)", min_value=1, max_value=20, step=1,
                        key="intra_history_sel")
    skip_pca, pca_k, scaling = ui.reduction_controls(PREFIX)
    controls = ui.clustering_controls(PREFIX)
    distribution_W = ui.window_control(PREFIX, log, "Distribution window W")
    st.divider()
    run_pipeline = st.button("Run intra-case pipeline", width="stretch", type="primary",
                             icon=":material/play_arrow:")

if run_pipeline:
    if not picked:
        st.warning("Select at least one feature group.")
        st.stop()
    with st.spinner("Building features, reducing dimensions, clustering states…"):
        fs = cache.build_features(log, "intra_case", features=tuple(picked), history=int(history))
        reduced, pca = cache.reduce_matrix(fs.values(), skip_pca, pca_k or None, scaling)
        annotations = tuple(fs.index["concept:name"].astype(str))
        model = cache.cluster(reduced, controls["clustering"], annotations, ui.cluster_params(controls))
        dist = cache.state_distribution(fs.index["time:timestamp"], model.state_ids,
                                        model.n_states, int(distribution_W))
        result = kairo.StateResult(
            perspective="intra_case",
            config=kairo.IntraConfig(
                features=tuple(picked), history=int(history), skip_pca=skip_pca,
                pca_components=pca_k or None, scaling=scaling,
                clustering=controls["clustering"], metric=controls["metric"],
                grid=controls["grid"], som_init=controls["som_init"],
                n_clusters=controls["kmeans_k"], eps=controls["eps"],
                min_samples=controls["min_samples"], window_minutes=int(distribution_W),
            ),
            features=fs, pca=pca, reduced=reduced, states=model,
            trajectories=kairo.trajectories(fs, model),
            transitions=kairo.case_transitions(fs, model),
            distribution=dist,
            signal=cache.drift_signal(dist, "js", "previous", 5),
        )
    st.session_state["intra_result"] = result
    st.session_state["intra_log_signature"] = current_signature
    st.session_state["intra_run_config"] = {
        **controls, "pca_k": int(pca_k), "skip_pca": skip_pca, "scaling": scaling,
        "window_minutes": int(distribution_W), "groups": list(picked), "history": int(history),
    }
    st.toast("Intra-case pipeline finished.", icon=":material/check_circle:")

if (
    any(key not in st.session_state for key in RESULT_KEYS)
    or st.session_state.get("intra_log_signature") != current_signature
):
    st.info("Set the sidebar controls and run the intra-case pipeline.")
    st.stop()

result: kairo.StateResult = st.session_state["intra_result"]
fs, model, pca = result.features, result.states, result.pca
selected = fs.columns
ran = st.session_state["intra_run_config"]
ui.run_banner(PREFIX)
ui.metrics_row([
    ("Events", f"{len(fs.matrix):,}"),
    ("Feature columns", f"{len(selected)}"),
    ("PCA →", "—" if pca is None else f"{pca.n_components}D"),
    ("States", f"{model.n_states}"),
    ("Transitions", f"{len(result.transitions):,}"),
])

tab_features, tab_states, tab_transitions, tab_drift = st.tabs(
    ["Features & PCA", "States & trajectory", "Transitions", "Drift signal"]
)

with tab_features:
    ui.render_feature_matrix(
        fs, selected,
        f"{len(fs.matrix):,} events × {len(selected)} feature columns "
        f"(|A|={len(fs.meta['activities'])}, |A→B|={len(fs.meta['transitions'])})",
    )
    ui.render_pca(pca)
    if ran["clustering"] == "dbscan":
        ui.render_k_distance(result.reduced, ran["kdist_k"], ran["metric"])

with tab_states:
    col_l, col_r = st.columns(2)
    with col_l:
        ui.render_state_grid(model, PREFIX, ran["clustering"])
    with col_r:
        st.markdown("**Case trajectory**")
        case_ids = fs.index["case:concept:name"].drop_duplicates().tolist()
        chosen = st.selectbox("Case", case_ids, index=0)
        sub = result.trajectories[result.trajectories["case:concept:name"] == chosen]
        sub_features = fs.matrix.loc[sub.index, selected]
        case_transitions = cache.find_transitions(
            sub["time:timestamp"], sub["state_id"].to_numpy(), tuple(model.labels), sub_features
        )
        fig = kairo.plot_trajectory(sub["time:timestamp"], sub["state_id"].to_numpy(), model,
                                    title=f"Case {chosen}")
        if not case_transitions.empty:
            kairo.add_transition_markers(fig, case_transitions["timestamp"])
        st.plotly_chart(fig, width="stretch")
    if ran["clustering"] == "som" and model.codebook is not None:
        st.plotly_chart(kairo.plot_state_distances(model, ran["metric"]), width="stretch")

with tab_transitions:
    st.markdown(f"**Transitions in case {chosen}**")
    ui.render_transitions(case_transitions, "No state changes in this case.")

with tab_drift:
    W = ran["window_minutes"]
    st.markdown("**State frequency distribution over time**")
    st.caption(
        f"Per {kairo.windows.window_minute_label(W)} window — the fraction of events in each state. "
        f"{len(result.distribution):,} windows across {fs.index['case:concept:name'].nunique():,} cases."
    )
    freq_fig = kairo.plot_state_distribution(result.distribution, model)
    kairo.add_window_boundaries(freq_fig, result.distribution["window_start"])
    st.plotly_chart(freq_fig, width="stretch")

    st.markdown("**Frequency-distribution divergences**")
    pick_div, pick_ref, pick_l = st.columns(3)
    div = pick_div.selectbox("Divergence", tuple(kairo.DIVERGENCES), key="intra_divergence_sel",
                             format_func=lambda d: kairo.DIVERGENCES[d])
    ref = pick_ref.selectbox("Compare against", tuple(kairo.REFERENCES), key="intra_reference_sel",
                             format_func=lambda r: kairo.REFERENCES[r])
    lookback = (
        pick_l.number_input("Windows to average (l)", min_value=1, step=1, key="intra_lookback_sel")
        if ref == "recent" else st.session_state.get("intra_lookback_sel", 5)
    )
    phrases = {"previous": "window i−1", "recent": f"mean of the {int(lookback)} windows before i",
               "baseline": "full-log baseline"}
    shift = cache.drift_signal(result.distribution, div, ref, int(lookback))
    shift_fig = kairo.plot_drift_signal(
        shift, title=f"{kairo.DIVERGENCES[div]}: window i vs. {phrases[ref]}")
    kairo.add_window_boundaries(shift_fig, shift["window_start"],
                                y_max=max(float(shift["score"].max()), 1e-9))
    st.plotly_chart(shift_fig, width="stretch")
