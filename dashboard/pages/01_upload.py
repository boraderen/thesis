"""Upload page: pick an event log (XES / CSV)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.loader import case_attributes, load_uploaded, summary_stats
from core.schema import render_schema_help

st.set_page_config(page_title="Upload", layout="wide")
st.title("1 — Upload event log")
st.caption("Drop an XES or CSV file.")

render_schema_help()

uploaded = st.file_uploader("Event log", type=["xes", "csv"], accept_multiple_files=False)

log: pd.DataFrame | None = None
if uploaded is not None:
    log = load_uploaded(uploaded.name, uploaded.read())
elif "log" in st.session_state:
    log = st.session_state["log"]

if log is None:
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
cases = preview["case:concept:name"].unique().tolist()
color_map = {c: f"hsl({(i * 53) % 360}, 60%, 92%)" for i, c in enumerate(cases)}
styled = preview.style.apply(
    lambda row: [f"background-color: {color_map[row['case:concept:name']]}"] * len(row), axis=1
)
st.dataframe(styled, width="stretch")

st.subheader("Activity frequency")
counts = log["concept:name"].value_counts().sort_values(ascending=True).reset_index()
counts.columns = ["activity", "count"]
fig = px.bar(counts, x="count", y="activity", orientation="h", height=max(220, 28 * len(counts)))
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")

if numeric_attrs or categorical_attrs:
    st.caption(
        f"Case-level attributes detected — numeric: {numeric_attrs or '—'}; "
        f"categorical: {categorical_attrs or '—'}"
    )
