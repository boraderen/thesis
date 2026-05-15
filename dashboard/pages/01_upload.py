"""Upload page: pick an event log (XES / CSV) or use the synthetic demo log."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.loader import (
    case_attributes,
    load_uploaded,
    summary_stats,
    synthetic_log,
)

st.set_page_config(page_title="Upload", layout="wide")
st.title("1 — Upload event log")
st.caption("Drop an XES or CSV file, or use the bundled synthetic log.")

col_upload, col_demo = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("Event log", type=["xes", "csv"], accept_multiple_files=False)
with col_demo:
    use_demo = st.button("Load synthetic demo", use_container_width=True)

log: pd.DataFrame | None = None
if uploaded is not None:
    log = load_uploaded(uploaded.name, uploaded.read())
elif use_demo or "log" not in st.session_state:
    if use_demo or "log" not in st.session_state:
        log = synthetic_log()

if log is None and "log" in st.session_state:
    log = st.session_state["log"]

if log is None:
    st.info("Upload a file or click *Load synthetic demo* to continue.")
    st.stop()

st.session_state["log"] = log
numeric_attrs, categorical_attrs = case_attributes(log)
st.session_state["case_numeric_attrs"] = numeric_attrs
st.session_state["case_categorical_attrs"] = categorical_attrs

stats = summary_stats(log)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cases", f"{stats['cases']:,}")
m2.metric("Events", f"{stats['events']:,}")
m3.metric("Activities", f"{stats['activities']:,}")
m4.metric("Resources", f"{stats['resources']:,}")
m5.metric("Span (h)", f"{(stats['end'] - stats['start']).total_seconds() / 3600:,.1f}")

st.subheader("Preview (first 20 rows)")
preview = log.head(20).copy()
cases = preview["case_id"].unique().tolist()
color_map = {c: f"hsl({(i * 53) % 360}, 60%, 92%)" for i, c in enumerate(cases)}
styled = preview.style.apply(
    lambda row: [f"background-color: {color_map[row['case_id']]}"] * len(row), axis=1
)
st.dataframe(styled, use_container_width=True)

st.subheader("Activity frequency")
counts = log["activity"].value_counts().sort_values(ascending=True).reset_index()
counts.columns = ["activity", "count"]
fig = px.bar(counts, x="count", y="activity", orientation="h", height=max(220, 28 * len(counts)))
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

if numeric_attrs or categorical_attrs:
    st.caption(
        f"Case-level attributes detected — numeric: {numeric_attrs or '—'}; "
        f"categorical: {categorical_attrs or '—'}"
    )
