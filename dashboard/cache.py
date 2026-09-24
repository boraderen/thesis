"""Streamlit memoization for reading the uploaded log, which runs on every rerun of the
upload page. The pipeline steps run on their buttons and keep their results in session
state instead."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pm4py
import streamlit as st

import kairo


def _with_file(name: str, raw: bytes, read):
    # pm4py and kairo read from a path, so the upload goes into a temporary file first
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(name).name
        path.write_bytes(raw)
        return read(str(path))


@st.cache_data(show_spinner=False)
def read_raw(name: str, raw: bytes) -> pd.DataFrame:
    # the file as it is, so its columns can be mapped to their roles
    if name.lower().endswith(".xes"):
        return _with_file(name, raw, pm4py.read_xes)
    return _with_file(name, raw, pd.read_csv)


@st.cache_data(show_spinner=False)
def load_log(name: str, raw: bytes, picked: dict) -> pd.DataFrame:
    # the log with its columns renamed to their roles. a csv keeps its timestamps as
    # text, so they are parsed here
    columns = {role: column for role, column in picked.items() if column}
    log = _with_file(name, raw, lambda path: kairo.read_log(path, **columns))
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    return log
