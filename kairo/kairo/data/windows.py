"""Calendar-window helpers shared by the windowed perspectives."""
from __future__ import annotations

import pandas as pd

from .schema import TIMESTAMP

WINDOW_MINUTE_OPTIONS: list[tuple[int, str]] = [
    (5, "5 min"),
    (15, "15 min"),
    (30, "30 min"),
    (60, "1 hour"),
    (120, "2 hours"),
    (240, "4 hours"),
    (480, "8 hours"),
    (720, "12 hours"),
    (1440, "1 day"),
    (4320, "3 days"),
    (10080, "1 week"),
    (20160, "2 weeks"),
    (30240, "3 weeks"),
    (43800, "1 month"),
    (87600, "2 months"),
    (131400, "3 months"),
    (262800, "6 months"),
    (525960, "1 year"),
]


def window_minute_choices(extra: object | None = None) -> list[int]:
    """The sorted minute values, optionally including a hand-typed custom value."""
    values = [m for m, _ in WINDOW_MINUTE_OPTIONS]
    if extra is None:
        return values
    minutes = as_window_minutes(extra, fallback=0)
    return values if minutes in (0, *values) else sorted({*values, minutes})


def window_minute_label(minutes: int) -> str:
    """The friendly label for a minute value, or a fallback like '37 min'."""
    for m, label in WINDOW_MINUTE_OPTIONS:
        if m == minutes:
            return label
    return f"{minutes} min"


def default_window_minutes(span_minutes: float, cap_windows: int = 200) -> int:
    """The smallest option that keeps the window count at or below `cap_windows`."""
    if span_minutes <= 0:
        return WINDOW_MINUTE_OPTIONS[0][0]
    for minutes, _ in WINDOW_MINUTE_OPTIONS:
        if span_minutes / minutes <= cap_windows:
            return minutes
    return WINDOW_MINUTE_OPTIONS[-1][0]


def as_window_minutes(value: object, fallback: int) -> int:
    """Coerce a picked or hand-typed window size to a positive whole number of minutes."""
    try:
        minutes = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return fallback
    return minutes if minutes > 0 else fallback


def log_span_minutes(log: pd.DataFrame) -> float:
    """Time span of a log in minutes (0 if it has fewer than two timestamps)."""
    if TIMESTAMP not in log.columns or len(log) < 2:
        return 0.0
    return (log[TIMESTAMP].max() - log[TIMESTAMP].min()).total_seconds() / 60.0


def floor_to_window(ts: pd.Series, origin: pd.Timestamp, minutes: int) -> pd.Series:
    """Floor timestamps to window starts anchored at `origin` (window width = minutes)."""
    freq = pd.Timedelta(minutes=minutes)
    return origin + ((ts - origin) // freq) * freq


def window_index(origin: pd.Timestamp, last_ts: pd.Timestamp, minutes: int) -> pd.DatetimeIndex:
    """The inclusive window-start index from origin through the window containing last_ts."""
    freq = pd.Timedelta(minutes=minutes)
    last_win = origin + ((last_ts - origin) // freq) * freq
    return pd.date_range(start=origin, end=last_win, freq=freq)


def assign_windows(log: pd.DataFrame, minutes: int) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Return a copy of the log with a ``__win__`` column plus the full window index."""
    df = log.copy()
    origin = df[TIMESTAMP].min()
    df["__win__"] = floor_to_window(df[TIMESTAMP], origin, minutes)
    return df, window_index(origin, df[TIMESTAMP].max(), minutes)
