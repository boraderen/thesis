"""Upload page: read an event log (XES / CSV) and map its columns by clicking them."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.loader import apply_mapping, read_log, summary_stats
from core.schema import MAX_CASE_ATTRS, OPTIONAL_ROLES, ROLES, render_feature_requirements

st.set_page_config(page_title="Upload", layout="wide")
st.title("1 — Upload event log")

STATE_KEYS = (
    "picked", "attrs", "attr_types", "n_attrs",
    "log", "case_numeric_attrs", "case_categorical_attrs",
)


def reset() -> None:
    """Forget the mapping, keeping the uploaded file."""
    for key in STATE_KEYS:
        st.session_state.pop(key, None)


def remove_log() -> None:
    """Forget the mapping and the file, and hand the uploader a fresh key."""
    reset()
    st.session_state.pop("file", None)
    st.session_state.pop("raw", None)
    st.session_state["uploader"] = st.session_state.get("uploader", 0) + 1


def mapping_table() -> None:
    """The mapping so far — one row per decided role and per picked attribute."""
    rows = [{"role": role, "column": picked[role] or "— skipped"} for role in ROLES if role in picked]
    rows += [
        {"role": f"case attribute ({attr_types[col]})" if col in attr_types else "case attribute",
         "column": col}
        for col in attrs
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def pick(prompt: str, key: str) -> str | None:
    """Show the unmapped columns and return the one whose header the user clicked."""
    st.markdown(prompt)
    free = [c for c in raw.columns if c not in set(picked.values()) and c not in attrs]
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
if uploaded is not None:
    if st.session_state.get("file") != uploaded.name:
        reset()
        st.session_state["file"] = uploaded.name
    st.session_state["raw"] = read_log(uploaded.name, uploaded.getvalue())

raw: pd.DataFrame | None = st.session_state.get("raw")
if raw is None:
    st.stop()

picked: dict[str, str | None] = st.session_state.setdefault("picked", {})
attrs: list[str] = st.session_state.setdefault("attrs", [])
attr_types: dict[str, str] = st.session_state.setdefault("attr_types", {})

render_feature_requirements()
left, right, _ = st.columns([1, 1, 6])
left.button("Reset mapping", on_click=reset, width="stretch")
right.button("Remove log", on_click=remove_log, width="stretch")
mapping_table()

# --- one role at a time ----------------------------------------------------
for role in ROLES:
    if role in picked:
        continue
    col = pick(f"Which column holds the **{role}**?", f"pick_{role}")
    if role in OPTIONAL_ROLES and st.button("Skip"):
        col = None
    elif col is None:
        st.stop()
    picked[role] = col
    st.rerun()

# --- then the case attributes, each one picked and typed before the next ---
if "n_attrs" not in st.session_state:
    n = st.number_input("How many case attributes?", 0, MAX_CASE_ATTRS, 0, 1)
    if st.button("Confirm", type="primary"):
        st.session_state["n_attrs"] = int(n)
        st.rerun()
    st.stop()

for i in range(st.session_state["n_attrs"]):
    if i == len(attrs):
        col = pick(
            f"Which column holds case attribute {i + 1} of {st.session_state['n_attrs']}?",
            f"pick_attr_{i}",
        )
        if col is None:
            st.stop()
        attrs.append(col)
        st.rerun()
    if attrs[i] not in attr_types:
        st.markdown(f"Is **{attrs[i]}** numeric or categorical?")
        num, cat, _ = st.columns([1, 1, 6])
        for kind, box in (("numeric", num), ("categorical", cat)):
            if box.button(kind.capitalize(), key=f"type_{i}_{kind}", width="stretch"):
                attr_types[attrs[i]] = kind
                st.rerun()
        st.stop()

# --- mapping complete: load the log ---------------------------------------
try:
    log = apply_mapping(raw, picked)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

numeric_attrs = [c for c in attrs if attr_types[c] == "numeric"]
categorical_attrs = [c for c in attrs if c not in numeric_attrs]
for col in numeric_attrs:
    log[col] = pd.to_numeric(log[col], errors="coerce")
st.session_state["log"] = log
st.session_state["case_numeric_attrs"] = numeric_attrs
st.session_state["case_categorical_attrs"] = categorical_attrs

stats = summary_stats(log)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cases", f"{stats['cases']:,}")
m2.metric("Events", f"{stats['events']:,}")
m3.metric("Activities", f"{stats['activities']:,}")
m4.metric("Resources", f"{stats['resources']:,}")
m5.metric("Span (h)", f"{(stats['end'] - stats['start']).total_seconds() / 3600:,.1f}")

t1, t2, t3, l1, l2, l3 = st.columns(6)
t1.metric("Min TPT (d)", f"{stats['tpt_min']:,.2f}")
t2.metric("Avg TPT (d)", f"{stats['tpt_mean']:,.2f}")
t3.metric("Max TPT (d)", f"{stats['tpt_max']:,.2f}")
l1.metric("Min trace length", f"{stats['len_min']:,}")
l2.metric("Avg trace length", f"{stats['len_mean']:,.1f}")
l3.metric("Max trace length", f"{stats['len_max']:,}")

st.subheader("Preview (first 20 rows)")
preview = log.head(20).copy()
cases = preview["case:concept:name"].unique().tolist()
color_map = {c: f"hsl({(i * 53) % 360}, 60%, 92%)" for i, c in enumerate(cases)}
styled = preview.style.apply(
    lambda row: [f"background-color: {color_map[row['case:concept:name']]}"] * len(row), axis=1
)
st.dataframe(styled, width="stretch")

st.subheader("Activity frequency")
counts = log["concept:name"].value_counts().sort_values(ascending=True).reset_index()
counts.columns = ["activity", "count"]
fig = px.bar(counts, x="count", y="activity", orientation="h", height=max(220, 28 * len(counts)))
fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width="stretch")
