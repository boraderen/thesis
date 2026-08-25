"""Inter-case states: windowed system features → PCA → clustered states → drift."""
from __future__ import annotations

from dataclasses import replace

import streamlit as st

import cache
import kairo
import ui
from controls import log_signature, seed_widget
from kairo.data.schema import FEATURE_DESCRIPTIONS, INTER_FEATURES

PREFIX = "inter"
SCHEMA_VERSION = "inter_window_v4"
RESULT_KEYS = ("inter_result", "inter_run_config", "inter_log_signature")

log = ui.require_log()
ui.page_header("Inter-case states",
               "One state per calendar window, capturing the global process situation: "
               "load, arrivals, pacing, stalls, and case-attribute mixes.", log)


def attr_description(key: str, columns: list[str]) -> str:
    kind, target = key.split(":", 1)
    if kind == "attr_mean":
        return f"Mean of {target} over the events in the window."
    if kind == "attr_std":
        return f"Standard deviation of {target} over the events in the window."
    return (
        f"One column per value of {target} ({len(columns)} in this log) — the share of the "
        "window's events carrying that value. The values come as a set, not one by one."
    )


numeric_attrs = tuple(st.session_state.get("case_numeric_attrs", []))
categorical_attrs = tuple(st.session_state.get("case_categorical_attrs", []))
ATTR_CATALOG = kairo.features.inter.attribute_feature_catalog(log, numeric_attrs, categorical_attrs)
FEATURE_LABELS = {**INTER_FEATURES, **{k: label for k, (label, _) in ATTR_CATALOG.items()}}
DESCRIPTIONS = {
    **{k: FEATURE_DESCRIPTIONS[k] for k in INTER_FEATURES},
    **{k: attr_description(k, cols) for k, (_, cols) in ATTR_CATALOG.items()},
}
ALL_FEATURES = list(FEATURE_LABELS)

attr_columns = sorted(c for _, cols in ATTR_CATALOG.values() for c in cols)
current_signature = log_signature(log, f"{SCHEMA_VERSION}:{attr_columns}")
if (
    st.session_state.get("inter_log_signature") is not None
    and st.session_state.get("inter_log_signature") != current_signature
):
    for key in (*RESULT_KEYS, "inter_W_sel"):
        st.session_state.pop(key, None)

run_cfg = st.session_state.get("inter_run_config") or {}
default_features = run_cfg.get("features", ALL_FEATURES)
for feature in ALL_FEATURES:
    seed_widget(f"inter_feat_{feature}", feature in default_features)
seed_widget("inter_stall_sel", int(run_cfg.get("stall_minutes", 60)))
ui.seed_common(PREFIX, run_cfg, log, grid_default=(2, 2))

with st.sidebar:
    st.header("Pipeline")
    st.subheader("1 · Features")
    picked = [
        feature for feature in ALL_FEATURES
        if st.checkbox(FEATURE_LABELS.get(feature, feature), key=f"inter_feat_{feature}")
    ]
    with st.expander("Feature glossary"):
        for feature, label in FEATURE_LABELS.items():
            st.markdown(f"**{label}.** {DESCRIPTIONS[feature]}")
    stall = st.slider("Stall threshold τ (minutes)", min_value=0, max_value=1000, step=5,
                      key="inter_stall_sel")
    skip_pca, pca_k, scaling = ui.reduction_controls(PREFIX)
    controls = ui.clustering_controls(PREFIX)
    window_minutes = ui.window_control(PREFIX, log, "Window W")
    st.divider()
    run_pipeline = st.button("Run inter-case pipeline", width="stretch", type="primary",
                             icon=":material/play_arrow:")

if run_pipeline:
    if not picked:
        st.warning("Select at least one feature.")
        st.stop()
    with st.spinner("Building features, fitting PCA, clustering states…"):
        fs = cache.build_features(
            log, "inter_case", features=tuple(picked), window_minutes=int(window_minutes),
            stall_minutes=int(stall), numeric_attrs=numeric_attrs,
            categorical_attrs=categorical_attrs,
        )
        if len(fs.matrix) < 2:
            st.warning("Window W is wider than the log span — pick a smaller W to get a trajectory.")
            st.stop()
        reduced, pca = cache.reduce_matrix(fs.values(), skip_pca, pca_k or None, scaling)
        model = cache.cluster(reduced, controls["clustering"], None, ui.cluster_params(controls))
        raw_centroids = kairo.analysis.state_means(fs.values(), model.state_ids, model.n_states)
        model = replace(model, dominant=kairo.features.inter.describe_states(raw_centroids, fs.columns))
        window_starts = fs.index["window_start"]
        result = kairo.StateResult(
            perspective="inter_case",
            config=kairo.InterConfig(
                features=tuple(picked), window_minutes=int(window_minutes),
                stall_minutes=int(stall), numeric_attrs=numeric_attrs,
                categorical_attrs=categorical_attrs, skip_pca=skip_pca,
                pca_components=pca_k or None, scaling=scaling,
                clustering=controls["clustering"], metric=controls["metric"],
                grid=controls["grid"], som_init=controls["som_init"],
                n_clusters=controls["kmeans_k"], eps=controls["eps"],
                min_samples=controls["min_samples"],
                signal_metric=st.session_state.get(f"{PREFIX}_shift_metric_sel", "euclidean"),
                signal_reference=st.session_state.get(f"{PREFIX}_shift_reference_sel", "previous"),
                signal_lookback=int(st.session_state.get(f"{PREFIX}_shift_lookback_sel", 5)),
            ),
            features=fs, pca=pca, reduced=reduced, states=model,
            trajectories=kairo.analysis.trajectories(fs, model),
            transitions=cache.find_transitions(window_starts, model.state_ids,
                                               tuple(model.labels), fs.matrix),
            distribution=cache.state_distribution(window_starts, model.state_ids,
                                                  model.n_states, int(window_minutes)),
            signal=cache.window_vector_shift(
                window_starts, reduced,
                st.session_state.get(f"{PREFIX}_shift_metric_sel", "euclidean"),
                st.session_state.get(f"{PREFIX}_shift_reference_sel", "previous"),
                int(st.session_state.get(f"{PREFIX}_shift_lookback_sel", 5)),
            ),
        )
    st.session_state["inter_result"] = result
    st.session_state["inter_log_signature"] = current_signature
    st.session_state["inter_run_config"] = {
        **controls, "pca_k": int(pca_k), "skip_pca": skip_pca, "scaling": scaling,
        "window_minutes": int(window_minutes), "stall_minutes": int(stall),
        "features": list(picked),
    }
    st.toast("Inter-case pipeline finished.", icon=":material/check_circle:")

if (
    any(key not in st.session_state for key in RESULT_KEYS)
    or st.session_state.get("inter_log_signature") != current_signature
):
    st.info("Set the sidebar controls and run the inter-case pipeline.")
    st.stop()

result: kairo.StateResult = st.session_state["inter_result"]
fs, model, pca = result.features, result.states, result.pca
ran = st.session_state["inter_run_config"]
ui.run_banner(PREFIX)
ui.metrics_row([
    ("Windows", f"{len(fs.matrix):,}"),
    ("Feature columns", f"{len(fs.columns)}"),
    ("τ (min)", f"{fs.meta['stall_minutes']}"),
    ("PCA →", "—" if pca is None else f"{pca.n_components}D"),
    ("States", f"{model.n_states}"),
])

tab_features, tab_states, tab_transitions, tab_drift = st.tabs(
    ["Features & PCA", "States & trajectory", "Transitions", "Drift signal"]
)

with tab_features:
    ui.render_feature_matrix(
        fs, fs.columns,
        f"{len(fs.matrix):,} windows × {len(fs.columns)} features "
        f"(W={fs.meta['window_minutes']} min, τ={fs.meta['stall_minutes']} min)",
    )
    ui.render_pca(pca)
    if ran["clustering"] == "dbscan":
        ui.render_k_distance(result.reduced, ran["kdist_k"], ran["metric"])

with tab_states:
    col_l, col_r = st.columns(2)
    with col_l:
        ui.render_state_grid(model, PREFIX, ran["clustering"])
    with col_r:
        st.markdown("**Inter-case state trajectory**")
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
