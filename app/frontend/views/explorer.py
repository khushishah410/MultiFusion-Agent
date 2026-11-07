from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.frontend.components.ui import banner, card, plotly_layout, safe_call

PAGE_SIZE = 25


def render(api) -> None:
    st.title("Dataset Explorer")
    summary = safe_call(api.get, "/dataset/summary")
    if summary is None:
        return
    if summary.get("error"):
        st.warning(summary["error"])
        st.info("Generate the demo data (python -m scripts.generate_demo_data) or prepare a dataset in Dataset Manager.")
        return

    top = st.columns([4, 1])
    top[0].markdown(f"Active manifest **{summary['manifest']}** ({summary['selected_by'].replace('_', ' ')}) · "
                    f"sources: {', '.join(summary['dataset_sources']) or '—'} · generated {summary.get('generated_at') or '—'}")
    if top[1].button("Refresh manifest", width="stretch"):
        safe_call(api.post, "/admin/reload")
        st.rerun()
    if summary["synthetic"]:
        banner("This manifest contains synthetic demo records. They have no clinical content and no labels.")

    v = summary.get("validation") or {}
    c = st.columns(5)
    with c[0]: card("Cases", f"{summary['case_count']:,}")
    with c[1]: card("Images listed", f"{summary['image_count']:,}",
                    f"{v.get('missing_image_files', '?')} missing on disk" if v else "not validated")
    with c[2]: card("Duplicates", summary.get("duplicate_case_ids") or 0, "case ids dropped at build")
    with c[3]: card("Malformed", summary.get("malformed_records") or 0, "records rejected at build")
    with c[4]: card("Splits", " / ".join(f"{k} {n}" for k, n in sorted(summary["split_counts"].items())))
    if v:
        if v.get("missing_image_files"):
            st.warning(f"{v['missing_image_files']} image file(s) listed in the manifest are missing "
                       f"({v['records_with_missing_images']} case(s)). Re-run download/prepare, or check the dataset folder.")
        if v.get("unreadable_images"):
            st.error(f"{v['unreadable_images']} image file(s) could not be decoded.")
        if v.get("leakage") and not v["leakage"].get("ok"):
            st.error("Leakage check failed: some patients/images appear in more than one split.")
    else:
        st.info("This manifest has not been validated yet (Dataset Manager → Validate).")

    if summary["label_counts"]:
        lc = pd.DataFrame(list(summary["label_counts"].items())[:25], columns=["label", "positive cases"])
        fig = px.bar(lc, x="positive cases", y="label", orientation="h", title="Positive labels (top 25)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(plotly_layout(fig, 420), width="stretch")
        if summary.get("uncertain_label_counts"):
            st.caption("Uncertain labels kept as uncertain: " +
                       ", ".join(f"{k} {n}" for k, n in summary["uncertain_label_counts"].items()))

    st.subheader("Cases")
    f = st.columns([3, 1, 2, 1])
    q = f[0].text_input("Search text or case id", key="ex_q")
    split = f[1].selectbox("Split", [""] + sorted(summary["split_counts"]), key="ex_split")
    label = f[2].selectbox("Label (positive)", [""] + list(summary["label_counts"]), key="ex_label")
    imgs = f[3].selectbox("Images", ["any", "with", "without"], key="ex_imgs")
    has_images = {"any": None, "with": True, "without": False}[imgs]

    page = st.session_state.get("ex_page", 1)
    data = safe_call(api.get, "/dataset/cases", q=q or None, split=split or None, label=label or None,
                     has_images=has_images, page=page, page_size=PAGE_SIZE)
    if data is None:
        return
    pages = max(1, -(-data["total"] // PAGE_SIZE))
    if page > pages:
        st.session_state["ex_page"] = 1
        st.rerun()
    if not data["items"]:
        st.write("No cases match these filters.")
        return
    rows = [{"case_id": c["case_id"], "split": c["split"], "images": len(c["image_paths"]),
             "missing images": c["image_exists"].count(False),
             "labels": ", ".join(k for k, v in c["labels"].items() if v == 1),
             "uncertain": ", ".join(c["label_uncertainty"]),
             "indication": (c["clinical_indication"] or "")[:80], "synthetic": c["synthetic"]} for c in data["items"]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    p = st.columns([1, 2, 1, 2])
    if p[0].button("◀ Previous", disabled=page <= 1):
        st.session_state["ex_page"] = page - 1; st.rerun()
    p[1].markdown(f"Page {page} of {pages} · {data['total']:,} matching cases")
    if p[2].button("Next ▶", disabled=page >= pages):
        st.session_state["ex_page"] = page + 1; st.rerun()
    with p[3]:
        e = st.columns(2)
        for i, fmt in enumerate(("csv", "json")):
            try:
                payload = api._request("GET", "/dataset/export", params={k: v for k, v in
                                       {"fmt": fmt, "q": q or None, "split": split or None, "label": label or None}.items() if v}).content
                e[i].download_button(fmt.upper(), payload, file_name=f"cases.{fmt}", width="stretch")
            except Exception as exc:  # export is optional; show why it is missing
                e[i].caption(f"{fmt} export failed: {exc}")

    st.subheader("Case details")
    cid = st.selectbox("Case", [c["case_id"] for c in data["items"]], key="ex_case")
    case = next(c for c in data["items"] if c["case_id"] == cid)
    left, right = st.columns([3, 2], gap="large")
    with left:
        for name, key in (("Clinical indication", "clinical_indication"), ("Findings", "radiology_findings"),
                          ("Impression", "radiology_impression")):
            st.markdown(f"**{name}**  \n{case.get(key) or '_not provided_'}")
        st.markdown(f"**MeSH major:** {', '.join(case['mesh_major']) or '—'}")
        st.markdown(f"**Labels:** {case['labels'] or '—'}  \n**Uncertain:** {case['label_uncertainty'] or '—'}  \n"
                    f"**Label source:** {case.get('label_source') or '—'}")
        st.markdown(f"**Image ids:** {', '.join(case['image_ids']) or '—'}  \n"
                    f"**Image paths (relative to dataset root):** {', '.join(case['image_paths']) or '—'}")
        with st.expander("Metadata"):
            st.json(case["metadata"])
    with right:
        if not case["image_paths"]:
            st.write("No images listed for this case.")
        for i, (path, exists) in enumerate(zip(case["image_paths"], case["image_exists"])):
            if exists:
                img = safe_call(api.get_bytes, f"/dataset/cases/{cid}/images/{i}")
                if img:
                    st.image(img, caption=path, width="stretch")
            else:
                st.warning(f"Missing on disk: {path}")
