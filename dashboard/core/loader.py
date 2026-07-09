"""Event log ingestion: XES (via pm4py if present) and CSV."""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import streamlit as st

REQUIRED = ("case:concept:name", "concept:name", "time:timestamp")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to the canonical schema and sort by case + time.

    No automatic renaming: the log must already carry the XES-style column
    names (`case:concept:name`, `concept:name`, `time:timestamp`, optionally
    `org:resource`, …).
    """
    df = df.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Log is missing required columns: {missing}")
    df["case:concept:name"] = df["case:concept:name"].astype(str)
    df["concept:name"] = df["concept:name"].astype(str)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["time:timestamp"]).sort_values(["case:concept:name", "time:timestamp"])
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


def summary_stats(df: pd.DataFrame) -> dict[str, object]:
    """Return dashboard summary metrics for a log."""
    return {
        "cases": df["case:concept:name"].nunique(),
        "events": len(df),
        "activities": df["concept:name"].nunique(),
        "resources": df["org:resource"].nunique() if "org:resource" in df.columns else 0,
        "start": df["time:timestamp"].min(),
        "end": df["time:timestamp"].max(),
    }


def span_label(df: pd.DataFrame) -> str:
    """Human-readable time-span caption: '5.2 days · 2026-04-01 04:10 → 2026-04-06 09:18'."""
    start, end = df["time:timestamp"].min(), df["time:timestamp"].max()
    span_s = (end - start).total_seconds()
    if span_s < 3600:
        rough = f"{span_s / 60:.1f} min"
    elif span_s < 86400:
        rough = f"{span_s / 3600:.1f} hours"
    elif span_s < 30 * 86400:
        rough = f"{span_s / 86400:.1f} days"
    elif span_s < 365 * 86400:
        rough = f"{span_s / (30.4 * 86400):.1f} months"
    else:
        rough = f"{span_s / (365 * 86400):.1f} years"
    return f"Log spans **{rough}** · {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M}"


def case_attributes(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical case-level attributes (constant per case_id)."""
    skip = {"case:concept:name", "concept:name", "time:timestamp", "org:resource"}
    candidates = [c for c in df.columns if c not in skip]
    nunique_per_case = df.groupby("case:concept:name")[candidates].nunique() if candidates else None
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
