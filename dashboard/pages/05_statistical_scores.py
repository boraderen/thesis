"""Statistical drift detection — per-perspective baseline-relative scores.

This page is a classical concept-drift detection view, conceptually separate
from the state-based SOM pipeline on the other pages. Each line measures how
far a window's *raw* per-perspective distribution sits from the full-log
baseline.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.drift_scores import per_window_scores
from core.loader import span_label
from core.windows import default_window_minutes, log_span_minutes, window_minute_label
from viz.drift_scores import score_line
from viz.drift_signal import add_window_boundaries

st.set_page_config(page_title="Statistical drift detection", layout="wide")
st.title("5 — Statistical drift detection")

if "log" not in st.session_state:
    st.warning("Load a log on the **Upload** page first.")
    st.stop()

log: pd.DataFrame = st.session_state["log"]
st.caption(span_label(log))

st.markdown(
    "These four lines compare each window's **raw distribution** to the "
    "full-log baseline — independent of the SOM views. Whichever line spikes "
    "is the perspective that drifted; the others should stay near zero."
)

W = default_window_minutes(log_span_minutes(log))
st.caption(f"Analysis window: {window_minute_label(W)} — chosen automatically from the log span.")

numeric_attrs = tuple(st.session_state.get("case_numeric_attrs", []))
categorical_attrs = tuple(st.session_state.get("case_categorical_attrs", []))
scores = per_window_scores(log, W, numeric_attrs=numeric_attrs, categorical_attrs=categorical_attrs)

KL_TITLES = [
    ("cf_score", "Control-flow — KL(activity dist || baseline)"),
    ("resource_score", "Resource — mean KL(per-activity resource dist || baseline)"),
    ("data_score", "Data attributes — mean(|z(numeric)| + KL(categorical))"),
]
kl_max = max(
    [float(scores[c].max()) for c, _ in KL_TITLES if c in scores.columns and not scores.empty],
    default=1.0,
)
for col, title in KL_TITLES:
    fig = score_line(scores, col, title=title)
    fig.update_yaxes(range=[0, max(kl_max, 1e-9)])
    add_window_boundaries(fig, scores["window_start"], y_min=0, y_max=max(kl_max, 1e-9))
    st.plotly_chart(fig, width="stretch")

# Inter-case uses a z-score (unbounded, different unit) so it gets its own axis.
inter_fig = score_line(scores, "inter_score",
                       title="Inter-case — |z(events in window)| vs. baseline mean/std")
inter_max = float(scores["inter_score"].max()) if not scores.empty else 1.0
add_window_boundaries(inter_fig, scores["window_start"], y_min=0, y_max=max(inter_max, 1e-9))
st.plotly_chart(inter_fig, width="stretch")

st.caption(
    "Different units across plots: KL divergence (CF/Resource/Data) typically "
    "sits in [0, ~2]; the inter-case z-score is unbounded — read each line "
    "against its own baseline level, not across plots."
)
