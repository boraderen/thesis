"""Streamlit entry point for the state-based process monitoring dashboard."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="State-based process monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("State-based process monitoring for concept drift detection")
st.write(
    "Pipelines that compute drift signals using state definitions for drifts from intra-case, resource and "
    "inter-case perspectives from an event log."
)

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
        res_feat [label="Windowed Resource features" width=2.1]
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
