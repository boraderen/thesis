"""Intra-case drift signal: placeholder until the drift functions are in kairo."""
from __future__ import annotations

import streamlit as st

import ui

ui.keep_widgets()
ui.intra_log()

with st.sidebar:
    st.header("Drift Signal")
    st.caption("No parameters yet.")

st.title("Drift Signal")
st.info(
    "Not built yet. It will continue with the states from **States & Trajectories**: "
    "the state frequencies per time window, compared window by window as the drift signal.",
    icon=":material/construction:",
)
