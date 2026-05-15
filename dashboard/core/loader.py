"""Event log ingestion: XES (via pm4py if present) and CSV. Synthetic fallback included."""
from __future__ import annotations

import io
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

REQUIRED = ("case_id", "activity", "timestamp")
_XES_RENAME = {
    "case:concept:name": "case_id",
    "concept:name": "activity",
    "time:timestamp": "timestamp",
    "org:resource": "resource",
    "org:group": "org_group",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to the canonical schema and sort by case + time."""
    df = df.rename(columns=_XES_RENAME).copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Log is missing required columns: {missing}")
    df["case_id"] = df["case_id"].astype(str)
    df["activity"] = df["activity"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["case_id", "timestamp"])
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    """Read a CSV buffer and normalize it."""
    return _normalize(pd.read_csv(io.BytesIO(raw)))


@st.cache_data(show_spinner=False)
def load_xes(raw: bytes) -> pd.DataFrame:
    """Read an XES buffer via pm4py and normalize it."""
    try:
        import pm4py
    except Exception as exc:
        raise RuntimeError("pm4py is required to load XES files") from exc
    tmp = io.BytesIO(raw)
    with open("/tmp/_dash_log.xes", "wb") as fh:
        fh.write(tmp.getvalue())
    log = pm4py.read_xes("/tmp/_dash_log.xes")
    df = pd.DataFrame(log) if not isinstance(log, pd.DataFrame) else log
    return _normalize(df)


@st.cache_data(show_spinner=False)
def synthetic_log(n_cases: int = 50, seed: int = 7) -> pd.DataFrame:
    """Build a small A→B→C log for offline demos."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 4, 1)
    rows = []
    resources = ["Alice", "Bob", "Carol"]
    priorities = ["low", "high"]
    for c in range(n_cases):
        t = start + timedelta(hours=float(rng.uniform(0, 24 * 5)))
        amount = float(rng.uniform(100, 1000))
        prio = priorities[int(rng.integers(0, 2))]
        for act in ("A", "B", "C"):
            t = t + timedelta(minutes=float(rng.uniform(5, 180)))
            rows.append({
                "case_id": f"c{c:03d}",
                "activity": act,
                "timestamp": t,
                "resource": resources[int(rng.integers(0, len(resources)))],
                "amount": amount,
                "priority": prio,
            })
    df = pd.DataFrame(rows)
    return _normalize(df)


def summary_stats(df: pd.DataFrame) -> dict[str, object]:
    """Return dashboard summary metrics for a log."""
    return {
        "cases": df["case_id"].nunique(),
        "events": len(df),
        "activities": df["activity"].nunique(),
        "resources": df["resource"].nunique() if "resource" in df.columns else 0,
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
    }


def case_attributes(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical case-level attributes (constant per case_id)."""
    skip = {"case_id", "activity", "timestamp", "resource", "org_group"}
    candidates = [c for c in df.columns if c not in skip]
    nunique_per_case = df.groupby("case_id")[candidates].nunique() if candidates else None
    keep = [c for c in candidates if nunique_per_case is not None and (nunique_per_case[c] <= 1).all()]
    numeric = [c for c in keep if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in keep if c not in numeric]
    return numeric, categorical


def load_uploaded(name: str, raw: bytes) -> Optional[pd.DataFrame]:
    """Dispatch on file extension."""
    lower = name.lower()
    if lower.endswith(".csv"):
        return load_csv(raw)
    if lower.endswith(".xes"):
        return load_xes(raw)
    return None
