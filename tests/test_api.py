from app.agents.retrieval_agent import RetrievalAgent
from tests.conftest import FakeAdapter, png_bytes, write_jsonl


def test_health_and_status(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}
    s = api_client.get("/status").json()
    assert s["image_model"]["status"] == "unavailable"
    assert s["dataset_manifest"]["manifest"] == "iu_xray.jsonl"
    assert s["retrieval_index"]["status"] == "not_configured"
    assert s["evaluation_reports"]["count"] == 0


def test_symptom_endpoint_and_validation(api_client):
    r = api_client.post("/analyze/symptoms", json={"text": "cough, no fever"})
    assert r.status_code == 200 and r.json()["negated_mentions"][0]["canonical"] == "fever"
    assert api_client.post("/analyze/symptoms", json={"text": "   "}).status_code == 422
    assert api_client.post("/analyze/symptoms", json={}).status_code == 422
    assert api_client.post("/analyze/symptoms", json={"text": "x" * 5001}).status_code == 422


def test_unrecognized_term_checked_against_active_dataset(api_client):
    r = api_client.post("/analyze/symptoms", json={"text": "fixture indication and glittery vision"}).json()
    found = {u["text"]: u["in_dataset_text"] for u in r["unrecognized_terms"]}
    assert found == {"fixture indication": True, "glittery vision": False}


def test_analyze_requires_some_input(api_client):
    assert api_client.post("/analyze", data={}).status_code == 422


def test_analyze_text_and_image_without_model(api_client):
    r = api_client.post("/analyze", data={"text": "dry cough"}, files={"image": ("x.png", png_bytes(), "image/png")})
    j = r.json()
    assert r.status_code == 200 and j["mode"] == "text_only"
    assert "image_model_output" in j["missing_components"] and j["image_analysis"]["scores"] == []


def test_analyze_with_fake_model_is_multimodal(api_client):
    state = api_client.state_obj
    state.adapter = FakeAdapter()
    state.reload()
    j = api_client.post("/analyze", data={"text": "cough"}, files={"image": ("x.png", png_bytes(), "image/png")}).json()
    assert j["mode"] == "multimodal" and j["image_analysis"]["model"]["name"].startswith("FakeAdapter")


def test_reference_case_image_analysis(api_client):
    state = api_client.state_obj
    state.adapter = FakeAdapter()
    state.reload()
    j = api_client.post("/analyze", data={"case_id": "CXR1", "use_case_image": 0}).json()
    assert j["mode"] == "image_model" and j["reference_case"]["case_id"] == "CXR1"
    assert "patient_id" not in j["reference_case"]
    missing = api_client.post("/analyze", data={"case_id": "CXR3", "use_case_image": 1})
    assert missing.status_code == 404
    assert api_client.post("/analyze", data={"text": "x", "case_id": "nope"}).status_code == 404


def test_upload_limits_and_bad_image(api_client):
    big = b"0" * (api_client.state_obj.settings.max_upload_bytes + 1)
    assert api_client.post("/analyze/image", files={"image": ("b.png", big, "image/png")}).status_code == 413
    r = api_client.post("/analyze/image", files={"image": ("fake.png", b"GIF89a....", "image/png")}).json()
    assert not r["validation"]["valid"] and r["scores"] == []


def test_dataset_routes(api_client):
    s = api_client.get("/dataset/summary").json()
    assert s["case_count"] == 6 and s["label_counts"] == {"Cardiomegaly": 3}
    assert s["validation"]["missing_image_files"] == 1
    page = api_client.get("/dataset/cases", params={"q": "indication 1", "page_size": 10}).json()
    assert page["total"] == 1 and page["items"][0]["case_id"] == "CXR1"
    assert api_client.get("/dataset/cases", params={"label": "Cardiomegaly"}).json()["total"] == 3
    case = api_client.get("/dataset/cases/CXR3").json()
    assert case["image_exists"] == [True, False]
    assert api_client.get("/dataset/cases/CXR1/images/0").headers["content-type"] == "image/png"
    assert api_client.get("/dataset/cases/CXR3/images/1").status_code == 404
    assert api_client.get("/dataset/cases/CXR1/images/9").status_code == 404
    assert api_client.get("/dataset/cases/missing").status_code == 404
    csv_text = api_client.get("/dataset/export", params={"fmt": "csv"}).text
    assert csv_text.splitlines()[0].startswith("case_id") and len(csv_text.splitlines()) == 7


def test_datasets_registry_routes_do_not_conflict(api_client):
    assert len(api_client.get("/datasets").json()) >= 7
    status = api_client.get("/datasets/status").json()
    assert isinstance(status, list) and {d["id"] for d in status} >= {"iu_xray", "chexpert"}
    assert api_client.get("/datasets/jobs").json() == []
    assert api_client.get("/datasets/iu_xray").json()["id"] == "iu_xray"
    assert api_client.get("/datasets/unknown").status_code == 400
    assert api_client.post("/datasets/chexpert/download", json={}).status_code == 400
    assert api_client.post("/datasets/iu_xray/download", json={}).status_code == 409  # large, unconfirmed
    v = api_client.post("/datasets/validate", json={"dataset_id": "iu_xray", "check_images": False}).json()
    assert v["missing_image_files"] == 1


def test_evaluation_routes(api_client):
    assert api_client.get("/evaluation/summary").json()["count"] == 0
    assert api_client.get("/evaluation/reports").json() == []
    bad = api_client.post("/evaluation/run", json={"predictions_file": "../../etc/passwd"})
    assert bad.status_code == 400
    assert api_client.post("/evaluation/run", json={"predictions_file": "nope.jsonl"}).status_code == 404
    eval_dir = api_client.state_obj.settings.evaluation_dir_abs
    model = {"name": "fake", "labels": ["Cardiomegaly"], "training_datasets": []}
    write_jsonl(eval_dir / "p.jsonl", [{"case_id": f"CXR{i}", "dataset_source": "iu_xray", "model": model,
                                        "scores": {"Cardiomegaly": 0.9 if i % 2 else 0.1}, "error": None}
                                       for i in range(1, 7)])
    rep = api_client.post("/evaluation/run", json={"predictions_file": "p.jsonl", "split": None}).json()
    assert rep["cases_evaluated"] == 6 and rep["per_label"][0]["auroc"] == 1.0
    assert api_client.get("/evaluation/summary").json()["count"] == 1


def test_retrieval_inactive_without_index(tmp_path):
    agent = RetrievalAgent(tmp_path / "none.json")
    res = agent.search("cough")
    assert res.status == "not_configured" and res.passages == [] and "build_retrieval_index" in res.note
