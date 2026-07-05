"""Intra-case SOM page: prefix features → PCA or pretrained autoencoder → SOM cells → per-case trajectory."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.autoencoder import (
    AE_ACTIVITIES,
    artifacts_available,
    encode_matrix,
    fixed_matrix,
)
from core.controls import (
    INTRA_SCHEMA_VERSION,
    log_signature,
    seed_choice,
    seed_multi,
    seed_widget,
)
from core.drift import intra_state_distribution
from core.features.intra_case import build_features
from core.loader import span_label
from core.pca import fit_pca
from core.som import train_som
from core.transitions import find_transitions
from core.windows import (
    SOM_GRID_OPTIONS,
    default_window_minutes,
    log_span_minutes,
    som_grid_label,
    window_minute_choices,
    window_minute_label,
)
from viz.drift_signal import add_window_boundaries, stacked_area_intra
from viz.som_grid import som_heatmap
from viz.tables import styled_feature_table
from viz.trajectory import (
    add_transition_markers,
    ae_error_histogram,
    pca_variance_plot,
    state_timeline,
)

st.set_page_config(page_title="Intra-case SOM", layout="wide")
st.title("2 — Intra-case SOM")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))

GROUP_LABELS = {
    "activity_freq": "Activity frequency vector",
    "bigram": "Bigram transition counts",
    "vocab": "Distinct activity set",
    "progress": "Progress ratio / trace length",
}
GROUP_DESCRIPTIONS = {
    "activity_freq": "One column per activity — how often the activity occurred in the case so far, divided by the number of events so far (the prefix).",
    "bigram": "One column per observed directly-follows pair A→B — how often that transition occurred in the prefix, divided by the number of transitions so far.",
    "vocab": "One binary column per activity — 1 if the activity has already occurred in the case prefix, 0 otherwise.",
    "progress": "Position of the event within its case as a fraction of the total case length (last event = 1).",
}

REDUCTION_LABELS = {
    "pca": "PCA",
    "autoencoder": "Autoencoder",
}

INTRA_RESULT_KEYS = (
    "intra_feat",
    "intra_spec",
    "intra_som",
    "intra_pca",
    "intra_ae",
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
available_groups = list(GROUP_LABELS)
seed_multi("intra_groups_sel", run_cfg.get("groups", available_groups), available_groups)
seed_choice("intra_method_sel", run_cfg.get("method", "pca"), tuple(REDUCTION_LABELS))
seed_widget("intra_pca_k_sel", int(run_cfg.get("pca_k", 0)))
grid_default = tuple(run_cfg.get("grid", (3, 3)))
seed_choice(
    "intra_grid_sel",
    grid_default if grid_default in SOM_GRID_OPTIONS else (3, 3),
    SOM_GRID_OPTIONS,
)
seed_widget(
    "intra_W_sel",
    int(run_cfg.get("distribution_W", default_window_minutes(log_span_minutes(log)))),
)

with st.sidebar:
    st.header("Controls")
    with st.form("intra_pipeline_controls"):
        st.subheader("Features for state clustering")
        picked = st.multiselect(
            "Include groups",
            options=available_groups,
            key="intra_groups_sel",
            format_func=lambda g: GROUP_LABELS[g],
        )
        with st.expander("Feature glossary"):
            st.caption(
                "All groups are prefix-based: computed per event from the case's "
                "events up to and including the current one."
            )
            for group, label in GROUP_LABELS.items():
                st.markdown(f"**{label}.** {GROUP_DESCRIPTIONS[group]}")
        st.subheader("Dimensionality reduction")
        method = st.radio(
            "Method",
            options=tuple(REDUCTION_LABELS),
            key="intra_method_sel",
            format_func=lambda m: REDUCTION_LABELS[m],
            help="The autoencoder needs all feature groups selected and a log "
            f"with at most {len(AE_ACTIVITIES)} distinct activities.",
        )
        pca_k = st.number_input(
            "PCA components",
            min_value=0,
            max_value=20,
            step=1,
            key="intra_pca_k_sel",
        )
        st.subheader("SOM")
        grid_h, grid_w = st.selectbox(
            "SOM grid",
            SOM_GRID_OPTIONS,
            key="intra_grid_sel",
            format_func=som_grid_label,
        )
        st.subheader("Windows")
        distribution_W = st.selectbox(
            "Distribution window W",
            window_minute_choices(st.session_state["intra_W_sel"]),
            key="intra_W_sel",
            format_func=window_minute_label,
            help="Calendar window used to aggregate per-event states into frequency bands.",
        )
        run_pipeline = st.form_submit_button(
            "Run intra-case pipeline",
            width="stretch",
        )

if run_pipeline:
    if not picked:
        st.warning("Select at least one feature group.")
        st.stop()
    if method == "autoencoder":
        if set(picked) != set(available_groups):
            st.warning(
                "The pretrained autoencoder uses the full feature space — select "
                "**all** feature groups, or switch to PCA."
            )
            st.stop()
        if not artifacts_available():
            st.warning(
                "No trained autoencoder found under `data/autoencoder/` — run "
                "`notebooks/train_autoencoder.ipynb` first."
            )
            st.stop()
        n_activities = int(log["activity"].nunique())
        if n_activities > len(AE_ACTIVITIES):
            st.warning(
                f"The pretrained autoencoder supports at most {len(AE_ACTIVITIES)} "
                f"distinct activities; this log has {n_activities}."
            )
            st.stop()
    with st.spinner("Building features, reducing dimensions, training SOM…"):
        feat, spec = build_features(log)
        selected_cols = [c for g in picked for c in spec.groups[g]]
        matrix = feat[selected_cols].to_numpy()
        if method == "autoencoder":
            pca = None
            ae = encode_matrix(fixed_matrix(feat, spec))
            reduced = ae.transformed
        else:
            pca = fit_pca(matrix, force_k=int(pca_k) if pca_k else None)
            ae = None
            reduced = pca.transformed
        som = train_som(
            reduced,
            grid_h=grid_h,
            grid_w=grid_w,
            annotations=tuple(feat["activity"].astype(str).tolist()),
        )
        feat = feat.assign(state_id=som.state_ids)
    st.session_state["intra_feat"] = feat
    st.session_state["intra_spec"] = spec
    st.session_state["intra_pca"] = pca
    st.session_state["intra_ae"] = ae
    st.session_state["intra_som"] = som
    st.session_state["intra_selected_cols"] = selected_cols
    st.session_state["intra_log_signature"] = current_log_signature
    st.session_state["intra_run_config"] = {
        "grid": (grid_h, grid_w),
        "method": method,
        "pca_k": int(pca_k),
        "distribution_W": int(distribution_W),
        "groups": list(picked),
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
ae = st.session_state["intra_ae"]
som = st.session_state["intra_som"]
selected_cols = st.session_state["intra_selected_cols"]
distribution_W = st.session_state["intra_run_config"]["distribution_W"]
ran_method = st.session_state["intra_run_config"].get("method", "pca")

st.subheader("Feature matrix")
st.caption(
    f"{len(feat):,} events × {len(selected_cols)} feature columns "
    f"(|A|={len(spec.activities)}, |A→B|={len(spec.transitions)})"
)
preview_cols = ["case_id", "activity", "timestamp", *selected_cols]
preview_groups = {g: [c for c in cols if c in selected_cols] for g, cols in spec.groups.items()}
styled = styled_feature_table(feat[preview_cols], preview_groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

if ran_method == "autoencoder":
    st.subheader("Autoencoder")
    st.markdown(
        "<style>[data-testid='stMetricValue'] { font-size: 1.4rem; }</style>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Reduction", f"{ae.raw_dim}D → {ae.latent_dim}D")
    m2.metric(
        "Explained variance",
        f"{ae.explained_variance:.1%}",
        help="Reconstruction R² = 1 − SSE/SST on this log — same footing as PCA's "
        "cumulative explained-variance ratio. Negative means the pretrained model "
        "reconstructs this log worse than its per-feature means would.",
    )
    m3.metric("Parameters", f"{ae.n_parameters:,}")
    m4.metric("Mean recon error", f"{ae.recon_errors.mean():.3f}")
    m5.metric("P95 recon error", f"{np.quantile(ae.recon_errors, 0.95):.3f}")
    st.plotly_chart(ae_error_histogram(ae.recon_errors), width="stretch")
else:
    st.subheader("PCA")
    st.plotly_chart(pca_variance_plot(pca.explained_variance_ratio, pca.chosen_k, pca.raw_dim), width="stretch")

col_l, col_r = st.columns([1, 1])
with col_l:
    st.subheader("SOM grid")
    st.plotly_chart(
        som_heatmap(
            som.grid_h, som.grid_w, som.cell_counts, som.cell_labels,
            title="States (hover for dominant last activity)",
            dominants=som.cell_dominant,
        ),
        width="stretch",
    )
with col_r:
    st.subheader("Case trajectory")
    case_ids = feat["case_id"].drop_duplicates().tolist()
    chosen = st.selectbox("Case", case_ids, index=0)
    sub = feat[feat["case_id"] == chosen].reset_index(drop=True)
    transitions = find_transitions(
        sub["timestamp"], sub["state_id"].to_numpy(), som.cell_labels, sub[selected_cols]
    )
    fig = state_timeline(
        sub["timestamp"], sub["state_id"].to_numpy(), som.cell_labels,
        title=f"Case {chosen}", cell_dominant=som.cell_dominant, xgap=0,
    )
    if not transitions.empty:
        add_transition_markers(fig, transitions["timestamp"])
    st.plotly_chart(fig, width="stretch")

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
    f"{len(intra_dist):,} windows across all {feat['case_id'].nunique():,} cases."
)
intra_cols = [f"intra_S{i}" for i in range(n_states)]
freq_fig = stacked_area_intra(intra_dist, intra_cols, som.cell_labels)
add_window_boundaries(freq_fig, intra_dist["window_start"])
st.plotly_chart(freq_fig, width="stretch")
