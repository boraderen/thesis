"""Overview page: what the dashboard does and where to start."""
from __future__ import annotations

import streamlit as st

st.title("State-based process monitoring for concept-drift detection")
st.caption(
    "Compute intra-case, resource, and inter-case states from an event log, follow how "
    "they evolve over time, and read significant changes as concept-drift signals. "
)

has_log = "log" in st.session_state
if has_log:
    st.success("An event log is loaded — pick a perspective below.", icon=":material/check_circle:")
else:
    st.info("Upload an event log.", icon=":material/upload_file:")

cards = st.columns(4)
with cards[0], st.container(border=True):
    st.markdown("**1 · Upload**")
    st.caption("Load an XES / CSV log and map its columns")
    st.page_link("views/upload.py", label="Upload log", icon=":material/upload_file:")
with cards[1], st.container(border=True):
    st.markdown("**2 · Intra-case**")
    st.caption("Features per event → PCA → clustered states → per-case trajectories.")
    st.page_link("views/intra.py", label="Intra-case states", icon=":material/route:")
with cards[2], st.container(border=True):
    st.markdown("**3 · Resource**")
    st.caption("Windowed features → PCA → clustered states → log-level trajectories.")
    st.page_link("views/resource.py", label="Resource states", icon=":material/group:")
with cards[3], st.container(border=True):
    st.markdown("**4 · Inter-case**")
    st.caption("Windowed features → PCA → clustered states → log-level trajectories.")
    st.page_link("views/inter.py", label="Inter-case states", icon=":material/hub:")

st.title("Overview of the pipelines")
st.graphviz_chart(
        """
digraph {
    rankdir=LR
    bgcolor=transparent
    node [shape=box style=rounded fontsize=11 color="#888888" fontcolor="#888888"]
    edge [color="#888888" arrowsize=0.7]

    log [label="Event log"]

    subgraph cluster_intra {
        label="Intra-case" fontcolor="#888888" color="#bbbbbb"
        intra_feat [label="Intra-case features" width=2.1]
        intra_red [label="PCA"]
        intra_som [label="SOM / clustering states"]
        intra_div [label="Freq. distri divergences"]
        intra_sig [label="Drift signals"]
        intra_feat -> intra_red -> intra_som
        intra_som -> intra_div
        intra_som -> intra_sig
        intra_div -> intra_sig
    }
    subgraph cluster_resource {
        label="Resource" fontcolor="#888888" color="#bbbbbb"
        res_feat [label="Windowed resource features" width=2.1]
        res_red [label="PCA"]
        res_som [label="SOM / clustering states"]
        res_dist [label="State vector distances"]
        res_sig [label="Drift signals"]
        res_feat -> res_red -> res_som
        res_red -> res_dist
        res_som -> res_sig
        res_dist -> res_sig
    }
    subgraph cluster_inter {
        label="Inter-case" fontcolor="#888888" color="#bbbbbb"
        inter_feat [label="Windowed inter-case features" width=2.1]
        inter_red [label="PCA"]
        inter_som [label="SOM / clustering states"]
        inter_dist [label="State vector distances"]
        inter_sig [label="Drift signals"]
        inter_feat -> inter_red -> inter_som
        inter_red -> inter_dist
        inter_som -> inter_sig
        inter_dist -> inter_sig
    }

    log -> intra_feat
    log -> res_feat
    log -> inter_feat
}
        """
)
