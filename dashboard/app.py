"""Kairo dashboard entry point: navigation, branding, and shared chrome."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ICON = ROOT.parent / "kairo" / "kairo.png"

st.set_page_config(
    page_title="Kairo — state-based process monitoring",
    page_icon=str(ICON) if ICON.exists() else "🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

if ICON.exists():
    st.logo(str(ICON), icon_image=str(ICON), size="large")

# Light global polish: calmer headers, bordered metric cards, roomier sidebar.
st.markdown(
    """
    <style>
      h1 { font-size: 1.9rem !important; letter-spacing: -0.01em; }
      h2 { font-size: 1.35rem !important; }
      h3 { font-size: 1.05rem !important; }
      [data-testid="stMetric"] {
        background: #F5F8FF;
        border: 1px solid #DDE5F6;
        border-radius: 10px;
        padding: 10px 14px;
      }
      [data-testid="stMetric"] label { color: #5E6470; }
      [data-testid="stSidebar"] [data-testid="stExpander"] details {
        border-radius: 8px;
      }
      div[data-testid="stDataFrame"] { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = st.navigation(
    {
        "": [st.Page("views/home.py", title="Overview", icon=":material/home:", default=True)],
        "Pipeline": [
            st.Page("views/upload.py", title="Upload log", icon=":material/upload_file:"),
            st.Page("views/intra.py", title="Intra-case states", icon=":material/route:"),
            st.Page("views/resource.py", title="Resource states", icon=":material/group:"),
            st.Page("views/inter.py", title="Inter-case states", icon=":material/hub:"),
        ],
    }
)
pages.run()
