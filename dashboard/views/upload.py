"""Upload page: read an event log (XES / CSV) and map its columns by clicking them."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import cache
import kairo

# the column roles of kairo.read_log, the last four are optional
ROLES = {
    "case_id": "case id",
    "activity": "activity",
    "timestamp": "timestamp",
    "event_id": "event id",
    "start_timestamp": "start timestamp",
    "resource": "resource",
    "event_duration": "event duration",
}
OPTIONAL_ROLES = ("event_id", "start_timestamp", "resource", "event_duration")

st.title("Upload event log")
st.caption("Load an XES or CSV file, then map its columns to their roles by clicking them.")


def reset() -> None:
    """Forget the mapping, keeping the uploaded file."""
    for key in ("picked", "log"):
        st.session_state.pop(key, None)


def remove_log() -> None:
    """Forget the mapping and the file, and hand the uploader a fresh key."""
    reset()
    st.session_state.pop("file", None)
    st.session_state.pop("file_bytes", None)
    st.session_state["uploader"] = st.session_state.get("uploader", 0) + 1


def mapping_table() -> None:
    """The mapping so far, one row per decided role."""
    rows = [{"role": ROLES[role], "column": column or "— skipped"} for role, column in picked.items()]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def pick(prompt: str, key: str) -> str | None:
    """Show the unmapped columns and return the one whose header was clicked."""
    st.markdown(prompt)
    free = [c for c in raw.columns if c not in set(picked.values())]
    if not free:
        st.error("Every column is already mapped.")
        st.stop()
    event = st.dataframe(
        raw[free].head(10),
        width="stretch",
        hide_index=True,
        key=key,
        on_select="rerun",
        selection_mode="single-column",
    )
    chosen = event.selection["columns"]
    return chosen[0] if chosen else None


uploaded = st.file_uploader(
    "Event log", type=["xes", "csv"], key=f"upload_{st.session_state.get('uploader', 0)}"
)
if uploaded is not None and st.session_state.get("file") != uploaded.name:
    reset()
    st.session_state["file"] = uploaded.name
    st.session_state["file_bytes"] = uploaded.getvalue()

if "file" not in st.session_state:
    st.stop()

raw = cache.read_raw(st.session_state["file"], st.session_state["file_bytes"])
picked: dict[str, str | None] = st.session_state.setdefault("picked", {})

left, right, _ = st.columns([1, 1, 6])
left.button("Reset mapping", on_click=reset, width="stretch")
right.button("Remove log", on_click=remove_log, width="stretch")
mapping_table()

# --- one role at a time ----------------------------------------------------
for role, label in ROLES.items():
    if role in picked:
        continue
    column = pick(f"Which column holds the **{label}**?", f"pick_{role}")
    if role in OPTIONAL_ROLES and st.button("Skip"):
        column = None
    elif column is None:
        st.stop()
    picked[role] = column
    st.rerun()

# --- mapping complete: load the log ---------------------------------------
try:
    log = cache.load_log(st.session_state["file"], st.session_state["file_bytes"], picked)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.session_state["log"] = log
stats = kairo.compute_log_stats(log)

st.success(
    f"Log mapped — {stats['events']:,} events of {stats['cases']:,} cases, "
    f"{stats['start']} to {stats['end']}",
    icon=":material/check_circle:",
)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cases", f"{stats['cases']:,}")
m2.metric("Events", f"{stats['events']:,}")
m3.metric("Activities", f"{stats['activities']:,}")
m4.metric("Resources", f"{stats['resources']:,}" if "org:resource" in log.columns else "—")
m5.metric("Span (h)", f"{stats['span_minutes'] / 60:,.1f}")

t1, t2, t3, l1, l2, l3 = st.columns(6)
t1.metric("Min TPT (d)", f"{stats['tpt_days_min']:,.2f}")
t2.metric("Avg TPT (d)", f"{stats['tpt_days_mean']:,.2f}")
t3.metric("Max TPT (d)", f"{stats['tpt_days_max']:,.2f}")
l1.metric("Min trace length", f"{stats['length_min']:,}")
l2.metric("Avg trace length", f"{stats['length_mean']:,.1f}")
l3.metric("Max trace length", f"{stats['length_max']:,}")

tab_preview, tab_activities = st.tabs(["Preview", "Activity counts"])
with tab_preview:
    preview = log.head(20).copy()
    cases = preview["case:concept:name"].unique().tolist()
    color_map = {c: f"hsl({(i * 53) % 360}, 60%, 92%)" for i, c in enumerate(cases)}
    styled = preview.style.apply(
        lambda row: [f"background-color: {color_map[row['case:concept:name']]}"] * len(row), axis=1
    )
    st.dataframe(styled, width="stretch")
with tab_activities:
    st.plotly_chart(kairo.plot_activity_counts(stats), width="stretch")
