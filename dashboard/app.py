"""Streamlit entry point for the state-based process monitoring dashboard."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="State-based process monitoring",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("State-based process monitoring")
st.write(
    "A three-SOM pipeline that turns an event log into intra-case, resource, "
    "and inter-case state trajectories, then combines them into a single drift signal."
)

if "log" in st.session_state:
    log = st.session_state["log"]
    st.success(f"Log loaded — {len(log):,} events across {log['case_id'].nunique():,} cases.")
else:
    st.info("Open **1 — Upload** in the sidebar to load a log (or use the synthetic demo).")

st.markdown(
    """
    **Pipeline**

    1. **Upload** — parse an XES or CSV event log.
    2. **Intra-case SOM** — windowed activity features → PCA or pretrained autoencoder → SOM cells per event.
    3. **Resource SOM** — per-window resource workload features → SOM cells per window.
    4. **Inter-case SOM** — arrivals, completions, stalls per window → SOM cells per window.
    5. **Joint drift signal** — stack the three views per window, look for sustained shifts.
    """
)
