"""Event log ingestion and statistics: read XES / CSV, map columns, summarise."""
from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import ACTIVITY, CASE, DURATION, RESOURCE, ROLES, TIMESTAMP


def read_log(source: str | Path | bytes, name: str | None = None) -> pd.DataFrame:
    """Read an event log from a path or an in-memory buffer, without renaming columns.

    `source` is a path to a ``.xes`` or ``.csv`` file, or raw bytes together
    with a `name` whose extension decides the format. XES goes through pm4py.
    """
    if isinstance(source, (str, Path)):
        name = str(source)
        if name.lower().endswith(".csv"):
            return pd.read_csv(source)
        return _read_xes(str(source))
    if name is None:
        raise ValueError("Reading from bytes needs a `name` to decide the format")
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(source))
    with tempfile.NamedTemporaryFile(suffix=".xes", delete=False) as fh:
        fh.write(source)
        path = fh.name
    try:
        return _read_xes(path)
    finally:
        Path(path).unlink(missing_ok=True)


def _read_xes(path: str) -> pd.DataFrame:
    try:
        import pm4py
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pm4py is required to load XES files") from exc
    log = pm4py.read_xes(path)
    return log if isinstance(log, pd.DataFrame) else pd.DataFrame(log)


def map_columns(df: pd.DataFrame, picked: dict[str, str | None]) -> pd.DataFrame:
    """Rename the picked columns ({role: column}) to canonical names, sort by case + time.

    Every other column rides along unchanged, except one that carries a canonical
    name without being picked for that role: the mapping is authoritative, so a
    skipped role leaves no column behind for the features that need it.
    """
    rename = {col: ROLES[role] for role, col in picked.items() if col}
    stale = set(ROLES.values()) - set(rename.values())
    out = df.drop(columns=[c for c in df.columns if c in stale and c not in rename])
    out = out.rename(columns=rename)
    out[CASE] = out[CASE].astype(str)
    out[ACTIVITY] = out[ACTIVITY].astype(str)
    out[TIMESTAMP] = pd.to_datetime(out[TIMESTAMP], utc=True, errors="coerce")
    if out[TIMESTAMP].isna().all():
        raise ValueError(f"No value in '{picked['timestamp']}' reads as a timestamp.")
    if picked.get("event duration"):
        out[DURATION] = pd.to_numeric(out[DURATION], errors="coerce")
        if out[DURATION].isna().all():
            raise ValueError(f"No value in '{picked['event duration']}' reads as a number.")
    out = out.dropna(subset=[TIMESTAMP])
    return out.sort_values([CASE, TIMESTAMP]).reset_index(drop=True)


def classify_attributes(log: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    """Split candidate case-attribute columns into "numeric" or "categorical".

    A column is numeric when at least 80% of its non-null values parse as numbers.
    """
    kinds: dict[str, str] = {}
    for col in columns:
        values = log[col].dropna()
        if len(values) == 0:
            kinds[col] = "categorical"
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        kinds[col] = "numeric" if numeric.notna().mean() >= 0.8 else "categorical"
    return kinds


@dataclass(frozen=True)
class LogStatistics:
    """Summary metrics of a mapped event log."""

    cases: int
    events: int
    activities: int
    resources: int
    start: pd.Timestamp
    end: pd.Timestamp
    span_minutes: float
    tpt_days_min: float
    tpt_days_mean: float
    tpt_days_median: float
    tpt_days_max: float
    length_min: int
    length_mean: float
    length_median: float
    length_max: int
    activity_counts: pd.Series
    resource_counts: pd.Series | None
    case_attributes: dict[str, str]


def log_statistics(
    log: pd.DataFrame, case_attributes: dict[str, str] | None = None
) -> LogStatistics:
    """Compute the basic statistics of a mapped log (per-case TPT and length included)."""
    by_case = log.groupby(CASE)[TIMESTAMP]
    tpt_d = (by_case.max() - by_case.min()).dt.total_seconds() / 86400
    length = log.groupby(CASE).size()
    start, end = log[TIMESTAMP].min(), log[TIMESTAMP].max()
    return LogStatistics(
        cases=log[CASE].nunique(),
        events=len(log),
        activities=log[ACTIVITY].nunique(),
        resources=log[RESOURCE].nunique() if RESOURCE in log.columns else 0,
        start=start,
        end=end,
        span_minutes=float((end - start).total_seconds() / 60.0),
        tpt_days_min=float(tpt_d.min()),
        tpt_days_mean=float(tpt_d.mean()),
        tpt_days_median=float(tpt_d.median()),
        tpt_days_max=float(tpt_d.max()),
        length_min=int(length.min()),
        length_mean=float(length.mean()),
        length_median=float(length.median()),
        length_max=int(length.max()),
        activity_counts=log[ACTIVITY].value_counts(),
        resource_counts=log[RESOURCE].value_counts() if RESOURCE in log.columns else None,
        case_attributes=dict(case_attributes or {}),
    )


def span_label(log: pd.DataFrame) -> str:
    """Human-readable time-span caption: '5.2 days · 2026-04-01 04:10 → 2026-04-06 09:18'."""
    start, end = log[TIMESTAMP].min(), log[TIMESTAMP].max()
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
