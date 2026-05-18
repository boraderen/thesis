"""Shared color palette for SOM states, kept consistent across all pages."""
from __future__ import annotations

STATE_COLORS: list[tuple[str, str]] = [
    ("#EEEDFE", "#3C3489"),  # purple
    ("#E1F5EE", "#085041"),  # teal
    ("#FAEEDA", "#633806"),  # amber
    ("#FAECE7", "#712B13"),  # coral
    ("#E7F0FA", "#1C3F66"),  # blue
    ("#F4EAF6", "#5B2A6B"),  # plum
    ("#EFF4E4", "#3C5417"),  # olive
    ("#FBE9F0", "#7A1F45"),  # rose
    ("#E6F1F1", "#1F4F4F"),  # pine
    ("#F2EFE3", "#5C4A1E"),  # sand
]

GROUP_TINTS: dict[str, str] = {
    "activity": "#EEEDFE",
    "delta": "#E1F5EE",
    "elapsed": "#EEEEEE",
    "case_attr": "#EEEEEE",
    "resource_a": "#FAEEDA",
    "resource_b": "#E1F5EE",
    "resource_c": "#EEEDFE",
    "activity_resource": "#E7F0FA",
    "handover": "#FFF6CC",
    "counts": "#E1F5EE",
    "rates": "#EEEDFE",
    "gaps": "#FFF6CC",
    "stall": "#FAEEDA",
}


def state_color(idx: int) -> tuple[str, str]:
    """Return (background, foreground) for a state index."""
    return STATE_COLORS[idx % len(STATE_COLORS)]


def state_bg(idx: int) -> str:
    """Background color for a state index."""
    return state_color(idx)[0]


def state_fg(idx: int) -> str:
    """Foreground color for a state index."""
    return state_color(idx)[1]
