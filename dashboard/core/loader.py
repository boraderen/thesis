"""Event log ingestion: XES (via pm4py if present) and CSV."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from core.schema import ROLES


@st.cache_data(show_spinner=False)
def read_log(name: str, raw: bytes) -> pd.DataFrame:
    """Read an uploaded CSV or XES buffer as-is, without touching column names."""
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    try:
        import pm4py
    except Exception as exc:
        raise RuntimeError("pm4py is required to load XES files") from exc
    with open("/tmp/_dash_log.xes", "wb") as fh:
        fh.write(raw)
    log = pm4py.read_xes("/tmp/_dash_log.xes")
    return pd.DataFrame(log) if not isinstance(log, pd.DataFrame) else log


@st.cache_data(show_spinner=False)
def apply_mapping(df: pd.DataFrame, picked: dict[str, str]) -> pd.DataFrame:
    """Rename the picked columns ({role: column}) to canonical names, sort by case + time.

    Every other column rides along unchanged, except one that carries a canonical
    name without being picked for that role: the mapping is authoritative, so a
    skipped role leaves no column behind for the features that need it.
    """
    rename = {col: ROLES[role] for role, col in picked.items() if col}
    stale = set(ROLES.values()) - set(rename.values())
    out = df.drop(columns=[c for c in df.columns if c in stale and c not in rename])
    out = out.rename(columns=rename)
    out["case:concept:name"] = out["case:concept:name"].astype(str)
    out["concept:name"] = out["concept:name"].astype(str)
    out["time:timestamp"] = pd.to_datetime(out["time:timestamp"], utc=True, errors="coerce")
    if out["time:timestamp"].isna().all():
        raise ValueError(f"No value in '{picked['timestamp']}' reads as a timestamp.")
    if picked.get("event duration"):
        out["event:duration_min"] = pd.to_numeric(out["event:duration_min"], errors="coerce")
        if out["event:duration_min"].isna().all():
            raise ValueError(f"No value in '{picked['event duration']}' reads as a number.")
    out = out.dropna(subset=["time:timestamp"])
    return out.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)


def summary_stats(df: pd.DataFrame) -> dict[str, object]:
    """Return dashboard summary metrics for a log, including per-case TPT and length."""
    by_case = df.groupby("case:concept:name")["time:timestamp"]
    tpt_d = (by_case.max() - by_case.min()).dt.total_seconds() / 86400
    length = by_case.size()
    return {
        "cases": df["case:concept:name"].nunique(),
        "events": len(df),
        "activities": df["concept:name"].nunique(),
        "resources": df["org:resource"].nunique() if "org:resource" in df.columns else 0,
        "start": df["time:timestamp"].min(),
        "end": df["time:timestamp"].max(),
        "tpt_min": float(tpt_d.min()),
        "tpt_mean": float(tpt_d.mean()),
        "tpt_max": float(tpt_d.max()),
        "len_min": int(length.min()),
        "len_mean": float(length.mean()),
        "len_max": int(length.max()),
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
