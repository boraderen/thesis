from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Kairo Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        ],
        "Intra-case states": [
            st.Page("views/intra/features.py", title="Features", icon=":material/table_chart:"),
            st.Page("views/intra/pca.py", title="PCA", icon=":material/compress:"),
            st.Page("views/intra/states.py", title="States & Trajectories", icon=":material/route:"),
            st.Page("views/intra/drift.py", title="Drift Signal", icon=":material/monitoring:"),
            st.Page("views/intra/copilot.py", title="Copilot", icon=":material/smart_toy:"),
        ],
        "Other perspectives": [
            st.Page("views/resource.py", title="Resource states", icon=":material/group:"),
            st.Page("views/inter.py", title="Inter-case states", icon=":material/hub:"),
        ],
    }
)
pages.run()
