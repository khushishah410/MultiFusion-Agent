import yaml

from app.agents.diagnosis_agent import DiagnosisAgent
from app.agents.image_agent import ImageAgent
from app.agents.model_adapters import UnavailableAdapter
from app.agents.symptom_agent import RuleBasedSymptomAgent
from app.core.config import REPO_ROOT
from tests.conftest import FakeAdapter, png_bytes

RULES = REPO_ROOT / "config" / "diagnosis_rules.yaml"


def run(lexicon_path, text, scores=None, context=None):
    sym = RuleBasedSymptomAgent(lexicon_path).analyze(text, context) if text is not None else None
    img = None
    if scores is not None:
        img = ImageAgent(FakeAdapter(scores), 10**7, 64, 4000).analyze(png_bytes())
    return DiagnosisAgent(RULES).suggest(sym, img)


def test_heart_failure_from_text_and_image(lexicon_path):
    d = run(lexicon_path, "Shortness of breath when climbing stairs for 2 weeks, swollen ankles, no chest pain, "
                          "maybe some cough at night", {"Cardiomegaly": 0.62, "Pneumonia": 0.2}, "ex-smoker")
    assert d.status == "suggested" and d.primary.key == "heart_failure"
    assert d.primary.strength == "strong" and d.primary.image_support and d.primary.symptom_support
    assert any("Cardiomegaly" in s for s in d.primary.supporting)
    assert not any("cough" in a for a in d.primary.against)  # hedged cough is not a denial
    assert d.primary.source_url.startswith("https://medlineplus.gov/")


def test_text_only_pneumonia_and_tb(lexicon_path):
    assert run(lexicon_path, "High fever with chills, cough with phlegm, sharp pain when breathing").primary.key == "pneumonia"
    assert run(lexicon_path, "Cough for a month, coughing up blood, night sweats and weight loss").primary.key == "tuberculosis"


def test_image_only_flag(lexicon_path):
    d = run(lexicon_path, None, {"Pneumothorax": 0.8})
    assert d.primary.key == "pneumothorax" and not d.primary.symptom_support
    assert any("No symptom text" in m for m in d.missing_information)


def test_denied_symptoms_lower_score(lexicon_path):
    plain = run(lexicon_path, "fever, chills and cough with phlegm").primary
    denied = run(lexicon_path, "fever, chills and cough with phlegm, no vomiting").primary
    assert plain.key == denied.key == "pneumonia"
    assert denied.points == plain.points - 0.5 and denied.against == ["Denies vomiting -0.5"]
    # denying a key symptom can change the ranking
    assert run(lexicon_path, "fever and cough with phlegm, no chills, no sharp pain when breathing").primary.key != "pneumonia"


def test_insufficient_evidence_and_no_image_model(lexicon_path):
    d = run(lexicon_path, "I feel a bit off today")
    assert d.status == "insufficient_evidence" and d.primary is None
    img = ImageAgent(UnavailableAdapter("x", "x", "no weights"), 10**7, 64, 4000).analyze(png_bytes())
    d = DiagnosisAgent(RULES).suggest(None, img)
    assert d.primary is None and any("No X-ray model output" in m for m in d.missing_information)


def test_rules_reference_known_symptoms_and_labels(lexicon_path):
    rules = yaml.safe_load(RULES.read_text())["conditions"]
    lexicon = set(yaml.safe_load(lexicon_path.read_text())["symptoms"])
    from app.agents.model_adapters import TXRV_TRAINING_DATA  # noqa: F401  (adapter module imports fine)
    xrv_labels = {"Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                  "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
                  "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"}
    for key, r in rules.items():
        assert set(r["symptoms"]) <= lexicon, key
        assert set(r["image"]) <= xrv_labels, key
        assert r["source"]["url"].startswith("https://medlineplus.gov/") and r["confirm_tests"], key


def test_api_returns_diagnosis(api_client):
    j = api_client.post("/analyze", data={"text": "fever, chills and cough with phlegm"}).json()
    assert j["diagnosis"]["primary"]["key"] == "pneumonia"
    assert j["fusion_summary"][0]["category"] == "diagnosis"


def test_negation_scope_stops_at_hedge(lexicon_path):
    r = RuleBasedSymptomAgent(lexicon_path).analyze("no chest pain, maybe some cough")
    assert [m.canonical for m in r.negated_mentions] == ["chest_pain"]
    assert r.symptom_mentions[0].canonical == "cough" and r.symptom_mentions[0].uncertain
    r = RuleBasedSymptomAgent(lexicon_path).analyze("No fever, chills or cough")
    assert [m.canonical for m in r.negated_mentions] == ["fever", "chills", "cough"]
