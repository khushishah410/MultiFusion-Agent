from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.frontend.components.ui import banner, safe_call

POLL_SECONDS = 2
POLL_LIMIT = 900


def _watch_job(api, job_id: str) -> None:
    box = st.empty()
    bar = st.progress(0.0)
    for _ in range(POLL_LIMIT):
        job = safe_call(api.get, f"/datasets/jobs/{job_id}")
        if job is None:
            return
        p = job.get("progress") or {}
        if p.get("total"):
            bar.progress(min(1.0, p["bytes"] / p["total"]), text=f"{p['file']}: {p['bytes'] / 1e6:.1f} / {p['total'] / 1e6:.1f} MB")
        elif p:
            box.write(p)
        if job["state"] != "running":
            if job["state"] == "succeeded":
                st.success(f"{job['kind'].title()} finished.")
                st.json(job["result"], expanded=False)
            else:
                st.error(f"{job['kind'].title()} failed: {job['error']}")
            return
        time.sleep(POLL_SECONDS)
    st.info("Still running; reopen this page to check again.")


def render(api) -> None:
    st.title("Dataset Manager")
    banner("Downloads only run for datasets whose terms permit it. Gated datasets (CheXpert, MIMIC-CXR, VinDr-CXR, "
           "PadChest, NIH via Box) must be obtained by you under their agreements, then imported from a local folder.")
    specs = safe_call(api.get, "/datasets")
    status = safe_call(api.get, "/datasets/status")
    if specs is None or status is None:
        return
    st_by_id = {s["id"]: s for s in status}
    table = [{"dataset": s["display_name"], "access": "automatic" if s["automated_download_allowed"] else s["strategy"],
              "documented size (MB)": s["documented_total_mb"], "download": st_by_id[s["id"]]["download_state"],
              "prepared": st_by_id[s["id"]]["prepared"], "validated": st_by_id[s["id"]]["validation_ok"],
              "active": st_by_id[s["id"]]["active"]} for s in specs]
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

    for spec in specs:
        ds, stt = spec["id"], st_by_id[spec["id"]]
        with st.expander(f"{spec['display_name']}  ·  {stt['download_state']}", expanded=False):
            st.markdown(f"**Official source:** [{spec['official_source']}]({spec['official_source']})" if
                        spec["official_source"].startswith("http") else f"**Source:** {spec['official_source']}")
            st.markdown(f"**License / access:** {spec['license']}")
            if spec.get("citation"):
                st.markdown(f"**Cite:** {spec['citation']}")
            st.markdown(f"**Modalities:** {', '.join(spec['modalities'])}  \n"
                        f"**Documented counts:** {spec.get('documented_counts') or '—'}  \n"
                        f"**Labels:** {spec.get('label_source') or '—'}")
            if spec["files"]:
                st.dataframe(pd.DataFrame(spec["files"])[["name", "url", "documented_size_mb", "metadata_only"]],
                             width="stretch", hide_index=True)
                st.caption(f"Size source: {spec.get('size_provenance') or 'not documented'}. No official checksums are published.")
            if stt.get("last_error"):
                st.error(f"Last error: {stt['last_error']}")

            cols = st.columns(4)
            if spec["automated_download_allowed"]:
                if cols[0].button("Download reports only", key=f"meta_{ds}"):
                    res = safe_call(api.post, f"/datasets/{ds}/download", {"metadata_only": True})
                    if res:
                        _watch_job(api, res["job_id"])
                big = (spec["documented_total_mb"] or 0) > 200
                confirm = st.checkbox(f"I confirm a download of about {spec['documented_total_mb']:,.0f} MB",
                                      key=f"conf_{ds}") if big else True
                if cols[1].button("Download everything", key=f"full_{ds}", disabled=not confirm):
                    res = safe_call(api.post, f"/datasets/{ds}/download", {"confirm_large": True})
                    if res:
                        _watch_job(api, res["job_id"])
            else:
                st.info(spec.get("instructions") or "Manual access required.")
                with st.form(f"import_{ds}"):
                    folder = st.text_input("Folder containing the files you downloaded",
                                           placeholder=r"D:\datasets\{}".format(ds))
                    if st.form_submit_button("Import folder") and folder:
                        res = safe_call(api.post, f"/datasets/{ds}/import", {"source_dir": folder})
                        if res:
                            st.json(res)
            if cols[2].button("Prepare + validate", key=f"prep_{ds}"):
                res = safe_call(api.post, f"/datasets/{ds}/prepare")
                if res:
                    _watch_job(api, res["job_id"])
            if cols[3].button("Activate", key=f"act_{ds}", disabled=not stt["prepared"]):
                res = safe_call(api.post, f"/datasets/{ds}/activate")
                if res:
                    st.success(f"{ds} is now the active dataset.")
            if stt["prepared"] and st.button("Re-validate (existence only)", key=f"val_{ds}"):
                res = safe_call(api.post, "/datasets/validate", {"dataset_id": ds, "check_images": False}, timeout=600)
                if res:
                    st.json({k: v for k, v in res.items() if k != "issues"}, expanded=False)
                    if res["issues"]:
                        st.dataframe(pd.DataFrame(res["issues"]), width="stretch", hide_index=True)
            if stt["download_state"] == "failed":
                st.caption("Retry: press the same download button. Partial files resume where the server allows it. "
                           "If the host is unreachable, download the archives in a browser from the official source, "
                           f"place them in data\\raw\\{ds}\\ and run Prepare.")
