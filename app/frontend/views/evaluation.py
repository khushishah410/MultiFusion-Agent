from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.frontend.components.ui import card, plotly_layout, safe_call

EXPLANATIONS = {
    "Weak labels": "IU X-Ray labels come from MeSH terms assigned to reports; NIH and CheXpert labels were mined "
                   "from reports by NLP. They contain errors, so metrics measure agreement with noisy labels.",
    "Label uncertainty": "CheXpert-style -1 labels are kept as uncertain. The report states the policy used: exclude "
                         "(default), treat as positive, or treat as negative. Results change with the policy.",
    "Data leakage": "Splits are grouped by patient id where the dataset provides one. If the model was trained on the "
                    "same dataset (TorchXRayVision 'all' weights include NIH, CheXpert, MIMIC, PadChest and Open-i), "
                    "the evaluation is not held out and the report says so.",
    "Class imbalance": "Rare labels make precision and F1 unstable; read AUROC with its confidence interval and the "
                       "number of positives.",
    "Retrospective vs clinical validity": "These numbers describe agreement with archived labels on one split. They do "
                                          "not show that the tool helps patients or clinicians in practice.",
}


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


def render(api) -> None:
    st.title("Evaluation")
    reports = safe_call(api.get, "/evaluation/reports")
    if reports is None:
        return
    with st.expander("Run an evaluation from a predictions file"):
        preds = safe_call(api.get, "/evaluation/predictions") or []
        if not preds:
            st.write("No predictions files found in reports/evaluation/. Create one with:")
            st.code("python -m scripts.predict_dataset --split test", language="powershell")
        else:
            c = st.columns(3)
            pf = c[0].selectbox("Predictions file", preds)
            split = c[1].selectbox("Split", ["test", "val", "train", "all"])
            policy = c[2].selectbox("Uncertain labels", ["exclude", "positive", "negative"])
            if st.button("Run evaluation", type="primary"):
                with st.spinner("Computing metrics (with bootstrap CIs)..."):
                    res = safe_call(api.post, "/evaluation/run", {"predictions_file": pf,
                                    "split": None if split == "all" else split, "uncertainty_policy": policy},
                                    timeout=600)
                if res:
                    st.success(f"Saved {res['file']}")
                    st.rerun()

    if not reports:
        st.info("No evaluation reports yet. Metrics are only shown when a report computed from real predictions exists.")
        st.code("python -m scripts.download_model\n"
                "python -m scripts.predict_dataset --split test\n"
                "python -m scripts.evaluate --input reports\\evaluation\\predictions_<dataset>_test.jsonl",
                language="powershell")
        _explain()
        return

    names = [f"{r['file']} · {r['generated_at']} · {r['model'].get('name')}" for r in reports]
    idx = st.selectbox("Report", range(len(reports)), format_func=lambda i: names[i])
    r = reports[idx]
    for w in r["warnings"]:
        st.warning(w)
    c = st.columns(5)
    with c[0]: card("Cases evaluated", r["cases_evaluated"], f"of {r['cases_in_split']} in split '{r['split']}'")
    with c[1]: card("Failed / missing", f"{r['failed']} / {r['missing_predictions']}")
    with c[2]: card("Avg latency", f"{r['avg_latency_ms']} ms" if r["avg_latency_ms"] is not None else "—")
    with c[3]: card("Macro AUROC", _fmt(r["macro"]["auroc"]), "labels with both classes")
    with c[4]: card("Macro sens / spec", f"{_fmt(r['macro']['sensitivity'])} / {_fmt(r['macro']['specificity'])}",
                    r["decision_rule"][:60])
    st.caption(f"Model {r['model'].get('name')} ({r['model'].get('version')}) · dataset {r['dataset']} · "
               f"uncertainty policy {r['uncertainty_policy']} · aggregation: {r['case_aggregation']}")

    if not r["per_label"]:
        st.info("The report has no evaluable labels (check the label map).")
        _explain()
        return
    df = pd.DataFrame(r["per_label"])
    show = df[["label", "n", "positives", "prevalence", "auroc", "sensitivity", "specificity", "precision", "f1", "accuracy"]].copy()
    show["AUROC 95% CI"] = df["ci95"].apply(lambda d: f"{d['auroc'][0]:.3f}–{d['auroc'][1]:.3f}" if d.get("auroc") else "—")
    st.dataframe(show, width="stretch", hide_index=True)

    long = df.melt(id_vars="label", value_vars=["auroc", "sensitivity", "specificity"]).dropna()
    fig = px.bar(long, x="label", y="value", color="variable", barmode="group", title="Per-label metrics")
    st.plotly_chart(plotly_layout(fig, 380).update_yaxes(range=[0, 1]), width="stretch")

    cols = st.columns(2)
    with cols[0]:
        lab = st.selectbox("Confusion matrix for", df["label"])
        e = df[df["label"] == lab].iloc[0]
        z = [[e["tn"], e["fp"]], [e["fn"], e["tp"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["pred negative", "pred positive"], y=["true negative", "true positive"],
                                   text=z, texttemplate="%{text}", colorscale=[[0, "#f4f7fa"], [1, "#1f5fa8"]],
                                   showscale=False))
        st.plotly_chart(plotly_layout(fig, 300).update_layout(title=f"{lab}"), width="stretch")
    with cols[1]:
        dist = pd.DataFrame([{"label": k, **v} for k, v in r["class_distribution"].items()])
        fig = px.bar(dist, x="label", y=["positive", "negative"], title="Class distribution", barmode="stack")
        st.plotly_chart(plotly_layout(fig, 300), width="stretch")

    compatible = [x for x in reports if x["dataset"] == r["dataset"] and x["split"] == r["split"]]
    if len(compatible) > 1:
        st.subheader("Compatible runs (same dataset and split)")
        comp = pd.DataFrame([{"run": f"{x['model'].get('name')} · {x['generated_at']}", **x["macro"]} for x in compatible])
        fig = px.bar(comp.melt(id_vars="run", value_vars=["auroc", "sensitivity", "specificity"]).dropna(),
                     x="variable", y="value", color="run", barmode="group")
        st.plotly_chart(plotly_layout(fig, 340), width="stretch")
    st.caption(r["interpretation"])
    _explain()


def _explain() -> None:
    with st.expander("How to read these results"):
        for k, v in EXPLANATIONS.items():
            st.markdown(f"**{k}.** {v}")
