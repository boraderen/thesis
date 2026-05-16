"""Shared options for calendar-window controls."""
from __future__ import annotations

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
