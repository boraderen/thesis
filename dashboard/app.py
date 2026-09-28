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


st.logo(str(Path(__file__).parent.parent / "kairo" / "kairo.png"), size="large")


def pipeline(p: str) -> list:
    # the same five pages for every perspective, each in the perspective's folder. the url
    # paths carry the perspective, the file names alone would clash
    return [
        st.Page(f"views/{p}/features.py", title="Features", icon=":material/table_chart:", url_path=f"{p}_features"),
        st.Page(f"views/{p}/pca.py", title="PCA", icon=":material/compress:", url_path=f"{p}_pca"),
        st.Page(f"views/{p}/states.py", title="States & Trajectories", icon=":material/route:", url_path=f"{p}_states"),
        st.Page(f"views/{p}/drift.py", title="Drift Signal", icon=":material/monitoring:", url_path=f"{p}_drift"),
        st.Page(f"views/{p}/copilot.py", title="Copilot", icon=":material/smart_toy:", url_path=f"{p}_copilot"),
    ]


pages = st.navigation(
    {
        "Kairo Dashboard": [st.Page("views/home.py", title="Overview", icon=":material/home:", default=True)],
        "Pipeline": [
            st.Page("views/upload.py", title="Upload log", icon=":material/upload_file:"),
        ],
        "Intra-case states": pipeline("intra"),
        "Resource states": pipeline("resource"),
        "Inter-case states": pipeline("inter"),
    }
)
pages.run()
