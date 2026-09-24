"""Intra-case PCA: scale the features, fit the components and look at them, then cut."""
from __future__ import annotations

import streamlit as st

import kairo
import ui
from controls import seed_widget
from kairo.analysis import META

ui.keep_widgets()
ui.intra_log()
features = ui.require("intra_features", "Compute the features on the **Features** page first.",
                      "views/intra/features.py", "Features")

columns = features.shape[1] - len(META)
pca = st.session_state.get("intra_pca")

seed_widget("intra_sel_scaling", "zscore")
seed_widget("intra_sel_components", min(20, columns))
seed_widget("intra_sel_cut", 5)

# the bounds come from the features and the fitted pca, so a value left from an
# earlier run can lie outside of them
st.session_state["intra_sel_components"] = min(st.session_state["intra_sel_components"], columns)
max_cut = pca.n_components_ if pca is not None else st.session_state["intra_sel_components"]
st.session_state["intra_sel_cut"] = min(st.session_state["intra_sel_cut"], max_cut)

with st.sidebar:
    st.header("PCA")
    st.radio("Scaling", ["zscore", "minmax"], key="intra_sel_scaling", horizontal=True,
             format_func=lambda s: {"zscore": "z-score", "minmax": "min-max"}[s])
    st.number_input("Components to fit", min_value=1, max_value=columns, step=1,
                    key="intra_sel_components")
    st.number_input("Cut after component", min_value=1, max_value=max_cut, step=1,
                    key="intra_sel_cut")

st.title("PCA")
st.caption("Scale the features and fit the components to see their variances, then cut.")

st.subheader("1 · Explained variance")
if st.button("Compute PCA", type="primary", icon=":material/play_arrow:"):
    with st.spinner("Scaling the features and fitting the components…"):
        scaled = kairo.standardize(features, st.session_state["intra_sel_scaling"])
        pca = kairo.compute_pca(scaled, int(st.session_state["intra_sel_components"]))
    ui.clear_from("pca")
    st.session_state["intra_scaled"] = scaled
    st.session_state["intra_pca"] = pca
    st.session_state["intra_plot_variance"] = kairo.plot_pca_variances(pca)
    # the cut in the sidebar was drawn before the fit, its maximum changed
    st.rerun()
ui.show_plot("intra_plot_variance", "Fit the components to see how much variance each one carries.")

st.subheader("2 · Cut")
if st.button("Apply PCA", type="primary", icon=":material/content_cut:", disabled=pca is None):
    cut = int(st.session_state["intra_sel_cut"])
    ui.clear_from("cut")
    st.session_state["intra_cut"] = cut
    st.session_state["intra_compressed"] = kairo.apply_pca(st.session_state["intra_scaled"], pca, cut)
    st.session_state["intra_plot_cut"] = kairo.plot_pca_variances(pca, cut)
ui.show_plot("intra_plot_cut", "Pick the cut in the sidebar and apply PCA.")

compressed = st.session_state.get("intra_compressed")
if compressed is not None:
    st.caption(f"{len(compressed):,} events × {st.session_state['intra_cut']} components, the first 30 rows.")
    st.dataframe(compressed.head(30), width="stretch")
