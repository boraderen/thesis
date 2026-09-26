"""Streamlit memoization for reading the uploaded log, which runs on every rerun of the
upload page. The pipeline steps run on their buttons and keep their results in session
state instead."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import pm4py
import streamlit as st


@st.cache_data(show_spinner=False)
def read_log(name: str, raw: bytes) -> pd.DataFrame:
    # the uploaded file read once, as it is. pandas reads a csv from its bytes, pm4py reads
    # an xes only from a path, so it goes into a temporary file first
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(name).name
        path.write_bytes(raw)
        return pm4py.read_xes(str(path))
