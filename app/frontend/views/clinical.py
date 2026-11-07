from __future__ import annotations

import html
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.frontend.components.ui import banner, card, chip, plotly_layout, safe_call

MODE_TEXT = {
    "demo": ("Demo", "Synthetic demo data contributed to this result."),
    "text_only": ("Text analysis only", "Only the symptom text was analysed."),
    "image_validation_only": ("Image validation only", "The image was checked, but no model output was available."),
    "image_model": ("Image model", "Only the image model produced output."),
    "multimodal": ("Multimodal", "Symptom text and image-model output are both present."),
}


def _highlight(sym: dict) -> str:
    text = sym["original_text"]
    spans = [(m["start"], m["end"], "sym") for m in sym["symptom_mentions"]]
    spans += [(m["start"], m["end"], "neg") for m in sym["negated_mentions"]]
    spans += [(u["start"], u["end"], "unk") for u in sym["unrecognized_terms"]]
    spans += [(d["start"], d["end"], "dur") for d in sym["duration_mentions"]]
    out, pos = [], 0
    for s, e, cls in sorted(spans):
        if s < pos:
            continue
        out.append(html.escape(text[pos:s]))
        out.append(f"<mark class='{cls}'>{html.escape(text[s:e])}</mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def _case_options(api) -> list[str]:
    data = safe_call(api.get, "/dataset/cases", page_size=200)
    return [c["case_id"] for c in data["items"]] if data else []


def render(api) -> None:
    st.title("Clinical Analysis")
    banner("Enter symptoms in your own words and optionally a chest X-ray. The app suggests the most likely "
           "diagnosis and alternatives, shows the evidence behind each, and lists the tests a doctor would use to confirm.")
    st.write("")

    with st.form("analysis"):
        left, right = st.columns([3, 2], gap="large")
        with left:
            text = st.text_area("Symptoms", height=140, max_chars=5000,
                                placeholder="e.g. Dry cough and fever for 3 days, no chest pain, a strange tingling in my fingers")
            c1, c2 = st.columns(2)
            context = c1.text_input("Context (optional)", placeholder="e.g. smoker, recent travel")
            duration = c2.text_input("Duration (optional)", placeholder="e.g. since last week")
            retrieve = st.toggle("Search the retrieval index", value=True)
        with right:
            upload = st.file_uploader("Chest X-ray (PNG or JPEG, optional)", type=["png", "jpg", "jpeg"])
            if upload is not None:
                st.image(upload, caption=upload.name, width="stretch")
            with st.expander("Attach a dataset case (optional)"):
                options = [""] + _case_options(api)
                case_id = st.selectbox("Reference case", options)
                use_case_image = st.checkbox("Analyse the reference case's first image instead of an upload",
                                             help="Lets you compare model output with that case's labels.")
        submitted = st.form_submit_button("Run analysis", type="primary", width="stretch")

    if submitted:
        if not (text.strip() or upload or (case_id and use_case_image)):
            st.warning("Enter symptom text, upload an image, or choose a dataset case image.")
            return
        form = {"text": text, "context": context, "duration": duration, "retrieve": str(retrieve).lower(),
                "case_id": case_id or None,
                "use_case_image": 0 if (case_id and use_case_image and upload is None) else None}
        image = (upload.name, upload.getvalue(), upload.type or "application/octet-stream") if upload else None
        with st.spinner("Analysing..."):
            st.session_state["last_result"] = safe_call(api.analyze, form, image)
    result = st.session_state.get("last_result")
    if result:
        show_result(result)


def show_result(r: dict) -> None:
    st.divider()
    title, desc = MODE_TEXT.get(r["mode"], (r["mode"], ""))
    lat = r["processing_metadata"]["latency_ms"]
    total = sum(v for v in lat.values() if isinstance(v, (int, float)))
    c = st.columns(4)
    with c[0]: card("Mode", title, desc)
    with c[1]: card("Symptoms found", len((r.get("symptom_analysis") or {}).get("symptom_mentions", [])),
                    f"{len((r.get('symptom_analysis') or {}).get('unrecognized_terms', []))} unrecognized term(s)")
    with c[2]:
        img = r.get("image_analysis")
        card("Image model", img["model"]["status"] if img else "not used", img["model"]["name"] if img else None)
    with c[3]: card("Latency", f"{total:.0f} ms", ", ".join(f"{k} {v:.0f}" for k, v in lat.items()))

    sym = r.get("symptom_analysis") or {}
    if sym.get("safety_notice"):
        st.error(sym["safety_notice"])
    for w in r["warnings"]:
        st.warning(w)
    if r["missing_components"]:
        st.info("Missing components: " + ", ".join(r["missing_components"]))

    if r.get("diagnosis"):
        _diagnosis_panel(r["diagnosis"])

    tabs = st.tabs(["Summary", "Symptoms", "Image", "Retrieved evidence", "Evidence table", "JSON"])
    with tabs[0]:
        for item in r["fusion_summary"]:
            ev = f"<div class='ev'>evidence: {', '.join(item['supported_by'])}</div>" if item["supported_by"] else ""
            st.markdown(f"<div class='mfa-item'>{chip(item['category'].replace('_', ' '), item['category'])}"
                        f"{html.escape(item['statement'])}{ev}</div>", unsafe_allow_html=True)
        if r["uncertainties"]:
            st.markdown("**Uncertainties**")
            for u in dict.fromkeys(r["uncertainties"]):
                st.markdown(f"- {u}")
        st.caption(r["disclaimer"])
    with tabs[1]:
        _symptom_tab(sym)
    with tabs[2]:
        _image_tab(r.get("image_analysis"))
    with tabs[3]:
        _retrieval_tab(r.get("retrieval_results"))
    with tabs[4]:
        if r["evidence"]:
            st.dataframe(pd.DataFrame(r["evidence"]), width="stretch", hide_index=True)
        else:
            st.write("No evidence items.")
    with tabs[5]:
        st.download_button("Download JSON", json.dumps(r, indent=2), file_name=f"{r['case_id']}.json",
                           mime="application/json")
        st.json(r, expanded=False)


STRENGTH_KIND = {"strong": "ok", "moderate": "warn", "weak": "missing"}


def _diagnosis_panel(d: dict) -> None:
    st.subheader("Suggested diagnosis")
    p = d.get("primary")
    if p is None:
        st.info(d["headline"])
    else:
        with st.container(border=True):
            left, right = st.columns([3, 2], gap="large")
            with left:
                st.markdown(f"### {html.escape(p['condition'])}", unsafe_allow_html=True)
                st.markdown(chip(f"{p['strength']} evidence", STRENGTH_KIND[p["strength"]])
                            + chip("X-ray supports" if p["image_support"] else "no X-ray support",
                                   "ok" if p["image_support"] else "missing")
                            + chip("symptoms support" if p["symptom_support"] else "no symptom support",
                                   "ok" if p["symptom_support"] else "missing")
                            + f" {p['points']:g} points · {p['match_percent']:g}% of available",
                            unsafe_allow_html=True)
                st.markdown("**Why**\n" + "\n".join(f"- {html.escape(x)}" for x in p["supporting"]))
                if p["against"]:
                    st.markdown("**Against**\n" + "\n".join(f"- {html.escape(x)}" for x in p["against"]))
            with right:
                st.markdown("**Tests a doctor would use to confirm**\n"
                            + "\n".join(f"- {html.escape(t)}" for t in p["confirm_tests"]))
                st.markdown(f"Source: [{p['source_name']}]({p['source_url']})")
    if d["differentials"]:
        rows = [{"condition": c["condition"], "points": c["points"], "match %": c["match_percent"],
                 "evidence": c["strength"], "X-ray": c["image_support"], "symptoms": c["symptom_support"],
                 "why": "; ".join(c["supporting"])} for c in ([p] if p else []) + d["differentials"]]
        df = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(x=df["points"], y=df["condition"], orientation="h",
                               marker_color=["#1f5fa8" if i == 0 and p else "#9aa9b8" for i in range(len(df))]))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(plotly_layout(fig, 60 + 40 * len(df)).update_layout(title="Differential diagnosis (points)"),
                        width="stretch")
        with st.expander("Differential table and scoring"):
            st.dataframe(df, width="stretch", hide_index=True)
            st.caption(d["scoring_note"])
    for m in d["missing_information"]:
        st.caption(f"Not considered: {m}")
    st.caption(d["disclaimer"])


def _symptom_tab(sym: dict) -> None:
    if not sym:
        st.write("No symptom text was analysed.")
        return
    st.markdown(f"Method: `{sym['method']}` · lexicon {sym['lexicon_version']}")
    st.markdown(_highlight(sym), unsafe_allow_html=True)
    st.caption("Blue: recognised symptom · grey/struck: stated as absent · amber: not in lexicon · teal: timing")
    rows = [{"text": m["text"], "concept": m["canonical"], "match": m["match_type"],
             "spelling similarity": m["match_similarity"], "status": "denied" if m["negated"] else "reported",
             "cue": m["negation_cue"] or m["uncertainty_cue"], "uncertain": m["uncertain"], "red flag": m["red_flag"]}
            for m in sym["symptom_mentions"] + sym["negated_mentions"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if sym["unrecognized_terms"]:
        st.markdown("**Unrecognized terms** (kept, not interpreted)")
        st.dataframe(pd.DataFrame([{"term": u["text"], "in lexicon": u["in_lexicon"],
                                    "appears in dataset reports": {True: "yes", False: "no", None: "not checked"}[u["in_dataset_text"]]}
                                   for u in sym["unrecognized_terms"]]), width="stretch", hide_index=True)
    cols = st.columns(2)
    cols[0].markdown("**Timing**  \n" + ("\n".join(f"- {d['text']}" for d in sym["duration_mentions"]) or "none found"))
    cols[1].markdown("**Context**  \n" + ("\n".join(f"- {c['kind']}: {c['text']}" for c in sym["context_mentions"]) or "none found"))
    for c in sym["contradictions"]:
        st.warning(c)
    with st.expander("Limitations of this extractor"):
        for lim in sym["limitations"]:
            st.markdown(f"- {lim}")


def _image_tab(img: dict | None) -> None:
    if not img:
        st.write("No image was analysed.")
        return
    v = img["validation"]
    st.markdown(chip("valid" if v["valid"] else "invalid", "ok" if v["valid"] else "bad")
                + f" {v.get('detected_format') or '?'} · {v.get('width')}x{v.get('height')} · mode {v.get('mode')} · "
                  f"{v['size_bytes'] / 1e3:.0f} kB", unsafe_allow_html=True)
    for e in v["errors"]:
        st.error(e)
    for w in v["warnings"]:
        st.warning(w)
    m = img["model"]
    st.markdown(f"**Model:** {m['name']} — status `{m['status']}`" + (f"  \n{m['reason']}" if m.get("reason") else ""))
    if img.get("error"):
        st.info(img["error"])
    if img["scores"]:
        df = pd.DataFrame(img["scores"]).sort_values("score")
        colors = ["#138a8a" if a else "#9aa9b8" for a in df["above_operating_point"].fillna(False)]
        fig = go.Figure(go.Bar(x=df["score"], y=df["label"], orientation="h", marker_color=colors))
        fig.add_vline(x=0.5, line_dash="dot", line_color="#b7791f", annotation_text="operating point")
        st.plotly_chart(plotly_layout(fig, 480).update_layout(title="Raw model scores (not probabilities)"),
                        width="stretch")
        st.caption(img["score_type"])
        for lim in img["score_limitations"]:
            st.markdown(f"- {lim}")
        with st.expander("Preprocessing and provenance"):
            for p in img["preprocessing"]:
                st.markdown(f"- {p}")
            st.json(m["provenance"])
            st.markdown("Published training data: " + ", ".join(m.get("training_datasets") or ["unknown"]))


def _retrieval_tab(ret: dict | None) -> None:
    if not ret:
        st.write("Retrieval was not run.")
        return
    if ret["status"] != "active":
        st.info(ret.get("note") or "Retrieval is not configured.")
        return
    st.caption(f"Query: “{ret['query']}” · method {ret['method']} · similarity is lexical overlap, not a probability")
    if not ret["passages"]:
        st.write(ret.get("note") or "No passages.")
    for p in ret["passages"]:
        tag = chip("reference case" if p["kind"] == "reference_case" else "document", "reference")
        if p["synthetic"]:
            tag += chip("synthetic", "warn")
        src = f"<a href='{html.escape(p['source_url'])}'>{html.escape(p['source'])}</a>" if p.get("source_url") else html.escape(p["source"])
        st.markdown(f"<div class='mfa-item'>{tag}<b>{html.escape(p['title'])}</b> · {src} · score {p['similarity']}"
                    f"<br>{html.escape(p['text'])}</div>", unsafe_allow_html=True)
