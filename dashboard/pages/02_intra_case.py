"""Intra-case SOM page: prefix features → PCA → SOM cells → per-case trajectory."""
from __future__ import annotations

import pandas as pd
import streamlit as st

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
from viz.trajectory import add_transition_markers, pca_variance_plot, state_timeline

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

FEATURE_SCHEMA_VERSION = "intra_prefix_v1"


def _log_signature(df: pd.DataFrame) -> tuple[object, ...]:
    return (
        FEATURE_SCHEMA_VERSION,
        len(df),
        tuple(df.columns),
        df["case_id"].nunique(),
        df["activity"].nunique(),
        str(df["timestamp"].min()),
        str(df["timestamp"].max()),
    )


def _available_groups() -> list[str]:
    return list(GROUP_LABELS)


INTRA_RESULT_KEYS = (
    "intra_feat",
    "intra_spec",
    "intra_som",
    "intra_pca",
    "intra_selected_cols",
    "intra_run_config",
    "intra_log_signature",
)

current_log_signature = _log_signature(log)
if (
    st.session_state.get("intra_log_signature") is not None
    and st.session_state.get("intra_log_signature") != current_log_signature
):
    for key in (
        *INTRA_RESULT_KEYS,
        "intra_selected_groups",
    ):
        st.session_state.pop(key, None)

with st.sidebar:
    st.header("Controls")
    with st.form("intra_pipeline_controls"):
        grid_default = st.session_state.get("intra_grid", (3, 3))
        if grid_default not in SOM_GRID_OPTIONS:
            grid_default = (3, 3)
        grid_h, grid_w = st.selectbox(
            "SOM grid",
            SOM_GRID_OPTIONS,
            index=SOM_GRID_OPTIONS.index(grid_default),
            format_func=som_grid_label,
        )
        pca_k = st.number_input(
            "PCA components (0 = auto/elbow)",
            min_value=0,
            max_value=20,
            value=int(st.session_state.get("intra_pca_k", 0)),
            step=1,
        )
        default_W = int(
            st.session_state.get(
                "intra_distribution_W",
                default_window_minutes(log_span_minutes(log)),
            )
        )
        win_options = window_minute_choices(default_W)
        distribution_W = st.selectbox(
            "Distribution window W",
            win_options,
            index=win_options.index(default_W),
            format_func=window_minute_label,
            help="Calendar window used to aggregate per-event states into frequency bands.",
        )

        st.subheader("Features for state clustering")
        available_groups = _available_groups()
        default_groups = st.session_state.get("intra_selected_groups", available_groups)
        default_groups = [g for g in default_groups if g in available_groups]
        if not default_groups:
            default_groups = available_groups
        picked = st.multiselect(
            "Include groups",
            options=available_groups,
            default=default_groups,
            format_func=lambda g: GROUP_LABELS[g],
        )
        run_pipeline = st.form_submit_button(
            "Run intra-case pipeline",
            width="stretch",
        )

if run_pipeline:
    feat, spec = build_features(log)
    selected_cols = [c for g in picked for c in spec.groups[g]]
    if not selected_cols:
        st.warning("Select at least one feature group that produces columns for the current settings.")
        st.stop()

    matrix = feat[selected_cols].to_numpy()
    pca = fit_pca(matrix, force_k=int(pca_k) if pca_k else None)
    som = train_som(
        pca.transformed,
        grid_h=grid_h,
        grid_w=grid_w,
        annotations=tuple(feat["activity"].astype(str).tolist()),
    )
    feat = feat.assign(state_id=som.state_ids)

    st.session_state["intra_grid"] = (grid_h, grid_w)
    st.session_state["intra_pca_k"] = pca_k
    st.session_state["intra_distribution_W"] = distribution_W
    st.session_state["intra_selected_groups"] = picked
    st.session_state["intra_selected_cols"] = selected_cols
    st.session_state["intra_feat"] = feat
    st.session_state["intra_spec"] = spec
    st.session_state["intra_pca"] = pca
    st.session_state["intra_som"] = som
    st.session_state["intra_log_signature"] = current_log_signature
    st.session_state["intra_run_config"] = {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "pca_k": int(pca_k),
        "distribution_W": distribution_W,
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
som = st.session_state["intra_som"]
selected_cols = st.session_state["intra_selected_cols"]
distribution_W = st.session_state["intra_run_config"]["distribution_W"]

st.subheader("Feature matrix")
st.caption(
    f"{len(feat):,} events × {len(selected_cols)} feature columns "
    f"(|A|={len(spec.activities)}, |A→B|={len(spec.transitions)})"
)
preview_cols = ["case_id", "activity", "timestamp", *selected_cols]
preview_groups = {g: [c for c in cols if c in selected_cols] for g, cols in spec.groups.items()}
styled = styled_feature_table(feat[preview_cols], preview_groups, max_rows=30)
st.dataframe(styled, width="stretch", height=380)

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
