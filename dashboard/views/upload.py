"""Upload page: read an event log (XES / CSV) once, then rename its columns by clicking them."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pm4py
import streamlit as st

import kairo

# the name every role's column gets, as kairo expects it, the last four are optional
ROLES = {
    "case:concept:name": "case id",
    "concept:name": "activity",
    "time:timestamp": "timestamp",
    "event_id": "event id",
    "start_timestamp": "start timestamp",
    "org:resource": "resource",
    "event_duration": "event duration",
}
OPTIONAL_ROLES = ("event_id", "start_timestamp", "org:resource", "event_duration")
# the kinds a case attribute can be for the inter-case features
KINDS = ["numerical", "categorical"]

st.title("Upload event log")
st.caption("Load an XES or CSV file, map its columns to their roles by clicking them, then pick the case attributes.")


def reset() -> None:
    """Forget the mapping, keeping the uploaded file."""
    for key in ("picked", "case_attributes", "log"):
        st.session_state.pop(key, None)


def remove_log() -> None:
    """Forget the mapping and the file, and hand the uploader a fresh key."""
    reset()
    st.session_state.pop("file", None)
    st.session_state.pop("raw", None)
    st.session_state["uploader"] = st.session_state.get("uploader", 0) + 1


def read_log(uploaded) -> pd.DataFrame:
    # the uploaded file read once, as it is. pandas reads a csv straight from the upload,
    # pm4py reads an xes only from a path, so it goes into a temporary file first
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(uploaded.name).name
        path.write_bytes(uploaded.getbuffer())
        return pm4py.read_xes(str(path))


def mapping_table() -> None:
    """The mapping so far, one row per decided role."""
    rows = [{"role": ROLES[role], "column": column or "— skipped"} for role, column in picked.items()]
    rows += [{"role": f"case attribute, {kind}", "column": column}
             for column, kind in st.session_state.get("case_attributes", {}).items()]
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
# a new file is read once and only the table is kept, so a log of up to 1 GB is not
# read, hashed or copied again on every click
if uploaded is not None and st.session_state.get("file") != uploaded.name:
    reset()
    with st.spinner(f"Reading {uploaded.name}…"):
        try:
            st.session_state["raw"] = read_log(uploaded)
        except Exception as exc:
            st.error(f"The file cannot be read: {exc}")
            st.stop()
    st.session_state["file"] = uploaded.name

if "raw" not in st.session_state:
    st.stop()

raw = st.session_state["raw"]
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

# --- then the case attributes, for the inter-case features ---------------------------
if "case_attributes" not in st.session_state:
    st.markdown("Which columns are **case attributes** for the inter-case features? Pick them and say "
                "whether each one is numerical or categorical, or skip them.")
    free = [c for c in raw.columns if c not in set(picked.values()) and c not in ROLES]
    columns = st.multiselect("Case attributes", free, key="attribute_columns")
    kinds = {}
    for column in columns:
        # a guess from the column, numbers stored as text are categorical until switched
        numerical = pd.api.types.is_numeric_dtype(raw[column])
        kinds[column] = st.radio(column, KINDS, index=0 if numerical else 1, horizontal=True,
                                 key=f"attribute_kind_{column}")
    if st.button("Done" if columns else "Skip"):
        st.session_state["case_attributes"] = kinds
        st.rerun()
    st.stop()

# --- mapping complete: rename the columns to their roles ----------------------
if "log" not in st.session_state:
    mapping = {column: role for role, column in picked.items() if column}
    # a column already named like a role another column was picked for would appear twice
    log = raw.drop(columns=[role for role in mapping.values() if role in raw.columns and role not in mapping])
    log = log.rename(columns=mapping)
    # a csv keeps its timestamps as text
    try:
        log["time:timestamp"] = pd.to_datetime(log["time:timestamp"])
    except ValueError as exc:
        st.error(f"The timestamp column cannot be read as dates: {exc}")
        st.stop()
    st.session_state["log"] = log

log = st.session_state["log"]
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
