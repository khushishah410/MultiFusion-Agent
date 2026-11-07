"""Streamlit entry point: streamlit run app/frontend/frontend_ui.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # streamlit only puts this file's folder on sys.path
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.frontend.components.ui import DISCLAIMER, client, inject_css  # noqa: E402
from app.frontend.views import about, clinical, evaluation, explorer, manager  # noqa: E402

st.set_page_config(page_title="MultiFusionAgent", page_icon="🩻", layout="wide")
inject_css()

PAGES = {
    "Clinical Analysis": clinical.render,
    "Dataset Explorer": explorer.render,
    "Evaluation": evaluation.render,
    "Dataset Manager": manager.render,
    "About & System Status": about.render,
}

with st.sidebar:
    st.markdown("### MultiFusionAgent")
    st.caption("Multimodal research prototype")
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    api = client()
    if api.healthy():
        st.markdown(f"Backend: **online**  \n`{api.base_url}`")
    else:
        st.markdown(f"Backend: **offline**  \n`{api.base_url}`")
        st.caption("Start it with .\\run_backend.ps1, then reload this page.")
    st.divider()
    st.caption(DISCLAIMER)

if not api.healthy():
    st.title(choice)
    st.error(f"The backend is not reachable at {api.base_url}. Start it in another terminal with "
             "`.\\run_backend.ps1` (or set MULTIFUSION_API_URL) and reload.")
    st.stop()

PAGES[choice](api)
