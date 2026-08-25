"""Streamlit memoization for the kairo calls that run on every rerun.

kairo itself is pure and uncached; this is the one place the dashboard
re-attaches caching. Only functions whose arguments Streamlit can hash
(DataFrames, arrays, primitives) belong here — results objects stay in
session_state instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import kairo


@st.cache_data(show_spinner=False)
def read_log(name: str, raw: bytes) -> pd.DataFrame:
    return kairo.read_log(raw, name=name)


@st.cache_data(show_spinner=False)
def map_columns(df: pd.DataFrame, picked: dict) -> pd.DataFrame:
    return kairo.map_columns(df, picked)


@st.cache_data(show_spinner=False)
def build_features(log: pd.DataFrame, perspective: str, **kwargs) -> kairo.features.FeatureSet:
    return kairo.features.build_features(log, perspective, **kwargs)


@st.cache_data(show_spinner=False)
def reduce_matrix(matrix: np.ndarray, skip_pca: bool, n_components: int | None, scaling: str):
    return kairo.analysis.reduce(matrix, skip_pca=skip_pca, n_components=n_components, scaling=scaling)


@st.cache_data(show_spinner=False)
def cluster(matrix: np.ndarray, method: str, annotations: tuple[str, ...] | None, params: dict):
    return kairo.analysis.cluster(matrix, method=method, annotations=annotations, **params)


@st.cache_data(show_spinner=False)
def state_distribution(timestamps: pd.Series, state_ids: np.ndarray, n_states: int, window_minutes: int):
    return kairo.analysis.state_distribution(timestamps, state_ids, n_states, window_minutes)


@st.cache_data(show_spinner=False)
def drift_signal(distribution: pd.DataFrame, divergence: str, reference: str, lookback: int):
    return kairo.analysis.drift_signal(distribution, divergence, reference, lookback)


@st.cache_data(show_spinner=False)
def window_vector_shift(window_starts: pd.Series, vectors: np.ndarray, metric: str,
                        reference: str = "previous", lookback: int = 5):
    return kairo.analysis.window_vector_shift(window_starts, vectors, metric, reference, lookback)


@st.cache_data(show_spinner=False)
def k_distances(matrix: np.ndarray, k: int, metric: str) -> np.ndarray:
    return kairo.analysis.k_distances(matrix, k=k, metric=metric)


@st.cache_data(show_spinner=False)
def find_transitions(timestamps: pd.Series, state_ids: np.ndarray, labels: tuple[str, ...],
                     features: pd.DataFrame) -> pd.DataFrame:
    return kairo.analysis.find_transitions(timestamps, state_ids, list(labels), features)
