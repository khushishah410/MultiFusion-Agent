from __future__ import annotations

import pandas as pd
import streamlit as st

from app.frontend.components.ui import DISCLAIMER, card, chip, safe_call

LIMITATIONS = [
    "Symptom extraction is rule-based (lexicon, negation and uncertainty cues). It misses paraphrases and can misread negation.",
    "The image model, when installed, is TorchXRayVision DenseNet-121. Its scores are uncalibrated and were produced on "
    "public datasets that may differ from your images.",
    "The suggested diagnosis is a transparent point score over hand-set rules (config/diagnosis_rules.yaml); "
    "it is not a probability and not a clinical diagnosis. The image model and the rules have not been validated together.",
    "Dataset labels are weak (report-derived). Retrospective metrics are not evidence of clinical benefit.",
    "Uploaded images are processed in memory and not stored unless retain_uploads is enabled.",
    "The API has no authentication; keep it bound to 127.0.0.1.",
]


def _state_chip(ok: bool | None, yes="ready", no="unavailable") -> str:
    return chip(yes if ok else no, "ok" if ok else ("warn" if ok is None else "bad"))


def render(api) -> None:
    st.title("About & System Status")
    s = safe_call(api.get, "/status")
    if s is None:
        return
    m = s["image_model"]
    c = st.columns(4)
    with c[0]: card("Backend", "online", f"Python {s['backend']['python']}")
    with c[1]: card("Active manifest", s["dataset_manifest"]["manifest"] or "none",
                    f"{s['dataset_manifest']['case_count']} cases" + (" · synthetic" if s["dataset_manifest"]["synthetic"] else ""))
    with c[2]: card("Image model", m["status"], m["name"])
    with c[3]: card("Evaluation reports", s["evaluation_reports"]["count"], s["evaluation_reports"]["latest"] or "none yet")

    st.subheader("Components")
    r = s["retrieval_index"]
    rows = [
        ("Backend API", True, "FastAPI, /docs for the schema"),
        ("Dataset manifest", s["dataset_manifest"]["error"] is None,
         s["dataset_manifest"]["error"] or f"{s['dataset_manifest']['manifest']} (selected: {s['dataset_manifest']['selected_by']})"),
        ("Image model", m["status"] == "loaded", m.get("reason") or m.get("version") or ""),
        ("Retrieval index", r["status"] == "active",
         f"{r.get('doc_count')} documents, built {r.get('built_at')}" if r["status"] == "active" else r.get("detail", "")),
        ("Symptom agent", True, f"{s['symptom_agent']['method']}, lexicon {s['symptom_agent']['lexicon_version']}"),
    ]
    for name, ok, detail in rows:
        st.markdown(f"{_state_chip(ok)} **{name}** — {detail}", unsafe_allow_html=True)

    st.subheader("Downloaded / prepared datasets")
    st.dataframe(pd.DataFrame(s["datasets"])[["id", "download_state", "imported_from_local_folder", "prepared",
                                              "validation_ok", "active"]], width="stretch", hide_index=True)

    st.subheader("Dependencies and configuration")
    cols = st.columns(2)
    cols[0].markdown("\n".join(f"- {_state_chip(v, 'installed', 'not installed')} `{k}`" for k, v in s["dependencies"].items()),
                     unsafe_allow_html=True)
    cols[1].json(s["config"])

    if m["status"] == "loaded":
        st.subheader("Model provenance")
        st.json({"provenance": m["provenance"], "published_training_data": m["training_datasets"], "labels": m["labels"]})

    st.subheader("Limitations")
    for lim in LIMITATIONS:
        st.markdown(f"- {lim}")
    st.warning(DISCLAIMER)
