"""Session-state helpers for the sidebar pipeline controls.

Sidebar widgets on the pipeline pages must carry an explicit ``key``: Streamlit
identifies an unkeyed widget by hashing its construction parameters, so a
widget whose default is read from session state that the run handler rewrites
gets a *new identity* on the rerun after a successful run, and the sidebar
snaps back to stale values. Keyed widgets are seeded once through these helpers
and never receive value/index/default arguments, so their identity stays stable
across runs — including across the reruns that show or hide the parameters of
the selected clustering method.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

INTRA_SCHEMA_VERSION = "intra_prefix_v1"


def log_signature(df: pd.DataFrame, schema_version: str) -> tuple[object, ...]:
    """Cheap identity of the loaded log; a mismatch invalidates stored results."""
    return (
        schema_version,
        len(df),
        tuple(df.columns),
        df["case:concept:name"].nunique(),
        df["concept:name"].nunique(),
        str(df["time:timestamp"].min()),
        str(df["time:timestamp"].max()),
    )


def seed_widget(key: str, default: object) -> None:
    """Initialise a widget's state slot once; later runs keep the user's value."""
    if key not in st.session_state:
        st.session_state[key] = default


def seed_choice(key: str, default: object, options: Sequence) -> None:
    """Seed a selectbox slot, resetting it if its value is no longer an option."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default


def seed_multi(key: str, defaults: Iterable, options: Sequence) -> None:
    """Seed a multiselect slot, dropping stored entries that left the option set."""
    current = st.session_state.get(key)
    if current is not None:
        kept = [v for v in current if v in options]
        if kept:
            if len(kept) != len(current):
                st.session_state[key] = kept
            return
    st.session_state[key] = [v for v in defaults if v in options]
