"""Shared options for calendar-window controls."""
from __future__ import annotations

import pandas as pd

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
    (262800, "6 month"),
    (525960, "1 year")
]


def window_minute_choices(extra: int | None = None) -> list[int]:
    """Return the sorted list of minute values, optionally including a custom value."""
    values = [m for m, _ in WINDOW_MINUTE_OPTIONS]
    if extra is not None and extra not in values:
        values = sorted({*values, int(extra)})
    return values


def window_minute_label(minutes: int) -> str:
    """Return the friendly label for a minute value, or a fallback like '37 min'."""
    for m, label in WINDOW_MINUTE_OPTIONS:
        if m == minutes:
            return label
    return f"{minutes} min"


def default_window_minutes(span_minutes: float, cap_windows: int = 200) -> int:
    """Pick the smallest WINDOW_MINUTE_OPTIONS value that keeps window count <= cap_windows."""
    if span_minutes <= 0:
        return WINDOW_MINUTE_OPTIONS[0][0]
    for minutes, _ in WINDOW_MINUTE_OPTIONS:
        if span_minutes / minutes <= cap_windows:
            return minutes
    return WINDOW_MINUTE_OPTIONS[-1][0]


def log_span_minutes(log: pd.DataFrame) -> float:
    """Return the time span of a log in minutes (0 if it has fewer than two timestamps)."""
    if "time:timestamp" not in log.columns or len(log) < 2:
        return 0.0
    return (log["time:timestamp"].max() - log["time:timestamp"].min()).total_seconds() / 60.0


def floor_to_window(ts: pd.Series, origin: pd.Timestamp, minutes: int) -> pd.Series:
    """Floor timestamps to window starts anchored at `origin` (window width = minutes)."""
    freq = pd.Timedelta(minutes=minutes)
    return origin + ((ts - origin) // freq) * freq


def window_index(origin: pd.Timestamp, last_ts: pd.Timestamp, minutes: int) -> pd.DatetimeIndex:
    """Build the inclusive window-start index from origin through the window containing last_ts."""
    freq = pd.Timedelta(minutes=minutes)
    last_win = origin + ((last_ts - origin) // freq) * freq
    return pd.date_range(start=origin, end=last_win, freq=freq)
