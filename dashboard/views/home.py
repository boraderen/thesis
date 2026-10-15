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
    st.page_link("views/intra/features.py", label="Intra-case states", icon=":material/route:")
with cards[2], st.container(border=True):
    st.markdown("**3 · Resource**")
    st.caption("A feature vector per feature window → PCA → clustered states → the log's trajectory.")
    st.page_link("views/resource/features.py", label="Resource states", icon=":material/group:")
with cards[3], st.container(border=True):
    st.markdown("**4 · Inter-case**")
    st.caption("A feature vector per feature window → PCA → clustered states → the log's trajectory.")
    st.page_link("views/inter/features.py", label="Inter-case states", icon=":material/hub:")

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
        intra_feat [label="Intra-case features per event" width=2.3]
        intra_red [label="Scaling and PCA"]
        intra_som [label="States: SOM, k-means or DBSCAN"]
        intra_dist [label="State distribution per signal window"]
        intra_sig [label="Drift signal: divergence per signal window"]
        intra_feat -> intra_red -> intra_som -> intra_dist -> intra_sig
    }
    subgraph cluster_resource {
        label="Resource" fontcolor="#888888" color="#bbbbbb"
        res_feat [label="Resource features per feature window" width=2.3]
        res_red [label="Scaling and PCA"]
        res_som [label="States: SOM, k-means or DBSCAN"]
        res_dist [label="State distribution per signal window"]
        res_sig [label="Drift signal: divergence per signal window"]
        res_feat -> res_red -> res_som -> res_dist -> res_sig
    }
    subgraph cluster_inter {
        label="Inter-case" fontcolor="#888888" color="#bbbbbb"
        inter_feat [label="Inter-case features per feature window" width=2.3]
        inter_red [label="Scaling and PCA"]
        inter_som [label="States: SOM, k-means or DBSCAN"]
        inter_dist [label="State distribution per signal window"]
        inter_sig [label="Drift signal: divergence per signal window"]
        inter_feat -> inter_red -> inter_som -> inter_dist -> inter_sig
    }

    log -> intra_feat
    log -> res_feat
    log -> inter_feat
}
        """
)
