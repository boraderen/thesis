"""State colours, shared so a state looks the same in every figure."""
from __future__ import annotations


# 25 visually distinct colors (Green-Armytage's "colour alphabet", light gray
# dropped) — enough for the largest grids; grid cells and trajectory segments
# share these so a state looks the same in every plot.
DISTINCT_COLORS = [
    "#AA0DFE", "#3283FE", "#85660D", "#782AB6", "#565656",
    "#1C8356", "#16FF32", "#F7E1A0", "#1CBE4F", "#C4451C",
    "#DEA0FD", "#FE00FA", "#325A9B", "#FEAF16", "#F8A19F",
    "#90AD1C", "#F6222E", "#1CFFCE", "#2ED9FF", "#B10DA1",
    "#C075A6", "#FC1CBF", "#B00068", "#FBE426", "#FA0087",
]

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

ACCENT = "#2B5FE3"


def blend(light: str, dark: str, w: float) -> str:
    """Linear blend between two hex colors: w=0 → light, w=1 → dark."""
    a = [int(light[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(dark[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * w):02X}" for x, y in zip(a, b))


def state_color(idx: int) -> tuple[str, str]:
    """(background, foreground) for a state index."""
    return STATE_COLORS[idx % len(STATE_COLORS)]


def state_bg(idx: int) -> str:
    return state_color(idx)[0]


def state_fg(idx: int) -> str:
    return state_color(idx)[1]


def _is_dark(color: str) -> bool:
    """Perceived-luminance check to pick a readable label color."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return (0.299 * r + 0.587 * g + 0.114 * b) < 140
