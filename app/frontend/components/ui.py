from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from app.frontend.api_client import ApiClient, BackendError

CSS = Path(__file__).resolve().parents[1] / "styles" / "theme.css"
PALETTE = ["#1f5fa8", "#138a8a", "#0f2742", "#5a8fc7", "#6cc3bd", "#9aa9b8"]
DISCLAIMER = ("Educational research prototype. Not a medical device and not clinically validated. "
              "Suggested diagnoses are rule-based and must be confirmed by a qualified doctor.")


def inject_css() -> None:
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def card(label: str, value, sub: str | None = None) -> None:
    sub_html = f"<div class='mfa-sub'>{html.escape(str(sub))}</div>" if sub else ""
    st.markdown(f"<div class='mfa-card'><div class='mfa-label'>{html.escape(label)}</div>"
                f"<div class='mfa-value'>{html.escape(str(value))}</div>{sub_html}</div>", unsafe_allow_html=True)


def chip(text: str, kind: str) -> str:
    return f"<span class='mfa-chip chip-{kind}'>{html.escape(text)}</span>"


def banner(text: str) -> None:
    st.markdown(f"<div class='mfa-banner'>{html.escape(text)}</div>", unsafe_allow_html=True)


@st.cache_resource
def client() -> ApiClient:
    return ApiClient()


def safe_call(fn, *args, **kwargs):
    """Run an API call; show the error inline and return None on failure."""
    try:
        return fn(*args, **kwargs)
    except BackendError as exc:
        st.error(str(exc))
        return None


def plotly_layout(fig, height: int = 320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="white",
                      plot_bgcolor="white", font=dict(color="#1c2733", size=12),
                      colorway=PALETTE, legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#e9eef3"); fig.update_yaxes(gridcolor="#e9eef3")
    return fig
