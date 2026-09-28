"""PCA: scale the features, fit the components and look at them, then cut, or skip PCA."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import kairo
import ui
from controls import seed_widget
from kairo.analysis import META

SCALINGS = {"zscore": "z-score", "minmax": "min-max", "skip": "skip"}


def scale(features: pd.DataFrame, method: str) -> pd.DataFrame:
    # skipping hands the features on as they are
    return features if method == "skip" else kairo.standardize(features, method)


def show(p: str) -> None:
    ui.keep_widgets()
    ui.perspective_log(p)
    features = ui.require(f"{p}_features", "Compute the features on the **Features** page first.",
                          f"views/{p}/features.py", "Features")

    columns = features.drop(columns=META, errors="ignore").shape[1]
    pca = st.session_state.get(f"{p}_pca")

    seed_widget(f"{p}_sel_scaling", "zscore")
    seed_widget(f"{p}_sel_components", min(20, columns))
    seed_widget(f"{p}_sel_cut", 5)

    # the bounds come from the features and the fitted pca, so a value left from an
    # earlier run can lie outside of them. pca fits at most one component per row
    most = min(columns, len(features))
    st.session_state[f"{p}_sel_components"] = min(st.session_state[f"{p}_sel_components"], most)
    max_cut = pca.n_components_ if pca is not None else st.session_state[f"{p}_sel_components"]
    st.session_state[f"{p}_sel_cut"] = min(st.session_state[f"{p}_sel_cut"], max_cut)

    with st.sidebar:
        st.header("PCA")
        st.radio("Scaling", list(SCALINGS), format_func=SCALINGS.get, key=f"{p}_sel_scaling", horizontal=True,
                 help="Skip hands the features on as they are.")
        st.number_input("Components to fit", min_value=1, max_value=most, step=1, key=f"{p}_sel_components")
        st.number_input("Cut after component", min_value=1, max_value=max_cut, step=1, key=f"{p}_sel_cut")

    st.title("PCA")
    st.caption("Scale the features and fit the components to see their variances, then cut. "
               "Or skip PCA and compute the states on the scaled features.")

    st.subheader("1 · Explained variance")
    if st.button("Compute PCA", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Scaling the features and fitting the components…"):
            scaled = scale(features, st.session_state[f"{p}_sel_scaling"])
            pca = kairo.compute_pca(scaled, int(st.session_state[f"{p}_sel_components"]))
        ui.clear_from(p, "pca")
        st.session_state[f"{p}_scaled"] = scaled
        st.session_state[f"{p}_pca"] = pca
        st.session_state[f"{p}_plot_variance"] = kairo.plot_pca_variances(pca)
        # the cut in the sidebar was drawn before the fit, its maximum changed
        st.rerun()
    ui.show_plot(f"{p}_plot_variance", "Fit the components to see how much variance each one carries.")

    st.subheader("2 · Cut")
    apply_column, skip_column, _ = st.columns([1, 1, 3])
    if apply_column.button("Apply PCA", type="primary", icon=":material/content_cut:", disabled=pca is None,
                           width="stretch"):
        cut = int(st.session_state[f"{p}_sel_cut"])
        ui.clear_from(p, "cut")
        st.session_state[f"{p}_cut"] = cut
        st.session_state[f"{p}_compressed"] = kairo.apply_pca(st.session_state[f"{p}_scaled"], pca, cut)
        st.session_state[f"{p}_plot_cut"] = kairo.plot_pca_variances(pca, cut)
    if skip_column.button("Skip PCA", icon=":material/skip_next:", width="stretch"):
        # the scaled features go on to the states as they are
        scaled = scale(features, st.session_state[f"{p}_sel_scaling"])
        ui.clear_from(p, "pca")
        st.session_state[f"{p}_scaled"] = scaled
        st.session_state[f"{p}_compressed"] = scaled
        st.session_state[f"{p}_pca_skipped"] = True
        st.rerun()

    compressed = st.session_state.get(f"{p}_compressed")
    if st.session_state.get(f"{p}_pca_skipped"):
        st.info(f"PCA skipped, the states are computed on the {columns:,} feature columns.", icon=":material/skip_next:")
    else:
        ui.show_plot(f"{p}_plot_cut", "Pick the cut in the sidebar and apply PCA, or skip it.")

    if compressed is not None:
        width = compressed.drop(columns=META, errors="ignore").shape[1]
        st.caption(f"{len(compressed):,} {ui.ROWS[p]} × {width:,} columns, the first 30 rows.")
        st.dataframe(compressed.head(30), width="stretch")
