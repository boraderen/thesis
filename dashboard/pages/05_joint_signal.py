"""Joint drift signal: stack intra-case state fractions with resource + inter-case state indices."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.controls import INTRA_SCHEMA_VERSION, log_signature, seed_widget
from core.drift import intra_state_distribution
from core.features.inter_case import build_features as build_inter
from core.features.resource import build_features as build_resource
from core.loader import span_label
from core.pca import fit_pca
from core.som import train_som
from core.state_attribution import intra_state_shift, state_id_shift
from core.windows import (
    default_window_minutes,
    log_span_minutes,
    window_minute_choices,
    window_minute_label,
)
from viz.drift_scores import score_line
from viz.drift_signal import add_window_boundaries, stacked_area_intra, state_index_line

st.set_page_config(page_title="Joint drift signal", layout="wide")
st.title("5 — Joint drift signal")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))

current_intra_signature = log_signature(log, INTRA_SCHEMA_VERSION)
if (
    "intra_feat" not in st.session_state
    or "intra_som" not in st.session_state
    or st.session_state.get("intra_log_signature") != current_intra_signature
):
    st.warning(
        "Run the **Intra-case SOM** pipeline first — the joint signal builds on its per-event states."
    )
    st.stop()

intra_feat: pd.DataFrame = st.session_state["intra_feat"]
intra_som = st.session_state["intra_som"]
has_resource = "resource" in log.columns

intra_cfg = st.session_state.get("intra_run_config") or {}
resource_cfg = st.session_state.get("resource_run_config") or {}
inter_cfg = st.session_state.get("inter_run_config") or {}
auto_W = default_window_minutes(log_span_minutes(log))

if st.session_state.get("joint_log_signature") not in (None, current_intra_signature):
    for key in (
        "joint_pack",
        "joint_log_signature",
        "joint_intra_W_sel",
        "joint_resource_W_sel",
        "joint_inter_W_sel",
    ):
        st.session_state.pop(key, None)

seed_widget("joint_intra_W_sel", int(intra_cfg.get("distribution_W", auto_W)))
if has_resource:
    seed_widget("joint_resource_W_sel", int(resource_cfg.get("window_minutes", auto_W)))
seed_widget("joint_inter_W_sel", int(inter_cfg.get("window_minutes", auto_W)))

with st.sidebar:
    st.header("Controls")
    with st.form("joint_controls"):
        st.subheader("Windows")
        intra_W = st.selectbox(
            "Intra distribution W",
            window_minute_choices(st.session_state["joint_intra_W_sel"]),
            key="joint_intra_W_sel",
            format_func=window_minute_label,
        )
        resource_W = (
            st.selectbox(
                "Resource W",
                window_minute_choices(st.session_state["joint_resource_W_sel"]),
                key="joint_resource_W_sel",
                format_func=window_minute_label,
            )
            if has_resource
            else None
        )
        inter_W = st.selectbox(
            "Inter-case W",
            window_minute_choices(st.session_state["joint_inter_W_sel"]),
            key="joint_inter_W_sel",
            format_func=window_minute_label,
        )
        st.caption(
            "Only the window sizes change here — grid, PCA, feature and τ choices "
            "are inherited from each pipeline's last run on its own page."
        )
        run_all = st.form_submit_button("Run all pipelines", width="stretch")

if run_all:
    with st.spinner("Recomputing all three pipelines…"):
        n_intra = intra_som.grid_h * intra_som.grid_w
        intra_dist = intra_state_distribution(
            intra_feat, n_states=n_intra, window_minutes=int(intra_W)
        )
        intra_pack = {
            "dist": intra_dist,
            "labels": list(intra_som.cell_labels),
            "n_states": n_intra,
            "W": int(intra_W),
        }

        if not has_resource:
            resource_pack = {"error": "Log has no resource column."}
        else:
            res_matrix, res_spec = build_resource(log, window_minutes=int(resource_W))
            if len(res_matrix) < 2:
                resource_pack = {
                    "error": "Window is wider than the log span — pick a smaller resource W.",
                }
            else:
                grid_h, grid_w = tuple(resource_cfg.get("grid", (2, 3)))
                stored_cols = st.session_state.get("resource_selected_cols") or []
                cols = [c for c in stored_cols if c in res_spec.columns] or list(res_spec.columns)
                pca_k = int(resource_cfg.get("pca_k", 0))
                res_pca = fit_pca(res_matrix[cols].to_numpy(), force_k=pca_k if pca_k else None)
                res_som = train_som(res_pca.transformed, grid_h=grid_h, grid_w=grid_w)
                resource_pack = {
                    "states": res_matrix.assign(state_id=res_som.state_ids)[["window_start", "state_id"]],
                    "labels": list(res_som.cell_labels),
                    "n_states": grid_h * grid_w,
                    "W": int(resource_W),
                }

        inter_matrix, inter_spec = build_inter(
            log,
            window_minutes=int(inter_W),
            stall_minutes=int(inter_cfg.get("stall_minutes", 60)),
        )
        if len(inter_matrix) < 2:
            inter_pack = {
                "error": "Window is wider than the log span — pick a smaller inter-case W.",
            }
        else:
            grid_h, grid_w = tuple(inter_cfg.get("grid", (2, 2)))
            stored_cols = st.session_state.get("inter_selected_cols") or []
            cols = [c for c in stored_cols if c in inter_spec.columns] or list(inter_spec.columns)
            pca_k = int(inter_cfg.get("pca_k", 0))
            inter_pca = fit_pca(inter_matrix[cols].to_numpy(), force_k=pca_k if pca_k else None)
            inter_som = train_som(inter_pca.transformed, grid_h=grid_h, grid_w=grid_w)
            inter_pack = {
                "states": inter_matrix.assign(state_id=inter_som.state_ids)[["window_start", "state_id"]],
                "labels": list(inter_som.cell_labels),
                "n_states": grid_h * grid_w,
                "W": int(inter_W),
            }

    st.session_state["joint_pack"] = {
        "intra_cfg": dict(intra_cfg),
        "intra": intra_pack,
        "resource": resource_pack,
        "inter": inter_pack,
    }
    st.session_state["joint_log_signature"] = current_intra_signature

if (
    "joint_pack" not in st.session_state
    or st.session_state.get("joint_log_signature") != current_intra_signature
):
    st.info("Pick the window sizes in the sidebar and run all pipelines.")
    st.stop()

pack = st.session_state["joint_pack"]
if pack["intra_cfg"] != intra_cfg:
    st.warning(
        "The intra-case pipeline was re-run since this signal was computed — "
        "hit **Run all pipelines** to refresh."
    )

intra_p = pack["intra"]
intra_cols = [f"intra_S{i}" for i in range(intra_p["n_states"])]
st.subheader(f"Intra-case state fractions over time (W = {window_minute_label(intra_p['W'])})")
intra_fig = stacked_area_intra(intra_p["dist"], intra_cols, intra_p["labels"])
add_window_boundaries(intra_fig, intra_p["dist"]["window_start"])
st.plotly_chart(intra_fig, width="stretch")

res_p = pack["resource"]
st.subheader("Resource state" + (f" (W = {window_minute_label(res_p['W'])})" if "states" in res_p else ""))
if "states" not in res_p:
    st.info(res_p["error"])
else:
    res_fig = state_index_line(
        res_p["states"], "state_id", res_p["labels"], title=window_minute_label(res_p["W"])
    )
    add_window_boundaries(
        res_fig, res_p["states"]["window_start"], y_min=-0.5, y_max=res_p["n_states"] - 0.5
    )
    st.plotly_chart(res_fig, width="stretch")

inter_p = pack["inter"]
st.subheader("Inter-case state" + (f" (W = {window_minute_label(inter_p['W'])})" if "states" in inter_p else ""))
if "states" not in inter_p:
    st.info(inter_p["error"])
else:
    inter_fig = state_index_line(
        inter_p["states"], "state_id", inter_p["labels"], title=window_minute_label(inter_p["W"])
    )
    add_window_boundaries(
        inter_fig, inter_p["states"]["window_start"], y_min=-0.5, y_max=inter_p["n_states"] - 0.5
    )
    st.plotly_chart(inter_fig, width="stretch")

st.caption(
    "A sustained shift in band proportions indicates concept drift. "
    "Each W can be tuned independently — coarser windows smooth the signal; finer ones expose short events. "
    "Grid size, PCA components, and feature choices are inherited from the per-stream pages."
)

st.divider()
st.subheader("State-based perspective attribution")
st.caption(
    "How much each SOM's state distribution shifts away from its own full-log "
    "baseline, per window. Stays within the state-based method: no raw activity / "
    "resource / attribute distributions are consulted here. The SOM with the "
    "highest spike is the perspective that drove the drift. Cross-talk between "
    "SOMs is still possible — see the **Statistical drift detection** page for "
    "a perspective-isolated reference."
)

intra_shift = intra_state_shift(intra_p["dist"])
res_shift = (
    state_id_shift(res_p["states"], n_states=res_p["n_states"]) if "states" in res_p else None
)
inter_shift = (
    state_id_shift(inter_p["states"], n_states=inter_p["n_states"]) if "states" in inter_p else None
)
shift_max = max(
    [s["score"].max() for s in (intra_shift, res_shift, inter_shift) if s is not None and not s.empty],
    default=1.0,
)


def _shift_plot(df: pd.DataFrame, title: str) -> None:
    fig = score_line(df, "score", title=title)
    fig.update_yaxes(range=[0, max(shift_max, 1e-9)])
    add_window_boundaries(fig, df["window_start"], y_min=0, y_max=max(shift_max, 1e-9))
    st.plotly_chart(fig, width="stretch")


_shift_plot(intra_shift, "Intra-case SOM — KL(window state dist || baseline)")
if res_shift is not None:
    _shift_plot(res_shift, "Resource SOM — rolling KL(state mix || baseline)")
if inter_shift is not None:
    _shift_plot(inter_shift, "Inter-case SOM — rolling KL(state mix || baseline)")
