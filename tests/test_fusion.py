from app.agents.fusion_agent import FusionAgent
from app.agents.image_agent import ImageAgent
from app.agents.model_adapters import UnavailableAdapter
from app.agents.retrieval_agent import RetrievalAgent, build_index
from app.agents.symptom_agent import RuleBasedSymptomAgent
from app.schemas.analysis import AnalysisMode
from tests.conftest import FakeAdapter, png_bytes

MAPS = {"iu_xray": {"Cardiomegaly": "Cardiomegaly", "Pleural Effusion": "Effusion"}}


def components(lexicon_path, tmp_path, adapter=None, synthetic=False):
    sym = RuleBasedSymptomAgent(lexicon_path).analyze("cough and fever, no chest pain, odd buzzing in ears")
    img = ImageAgent(adapter or FakeAdapter(), 10**7, 64, 4000).analyze(png_bytes())
    build_index([{"doc_id": "d1", "title": "Report 1", "text": "Findings: cough fever pneumonia", "source": "iu_xray",
                  "kind": "reference_case", "synthetic": synthetic}], [], tmp_path / "i.json")
    ret = RetrievalAgent(tmp_path / "i.json").search("cough fever")
    return sym, img, ret


def test_all_components_multimodal_with_provenance(lexicon_path, tmp_path):
    sym, img, ret = components(lexicon_path, tmp_path)
    r = FusionAgent(MAPS).fuse(sym, img, ret)
    assert r.mode == AnalysisMode.multimodal and r.missing_components == []
    ids = {e.id for e in r.evidence}
    for item in r.fusion_summary:
        assert set(item.supported_by) <= ids  # every cited evidence id exists
    retrieved = [e for e in r.evidence if e.component == "retrieval_agent"]
    assert retrieved and all(e.derived_from for e in retrieved)  # marked as dependent on the text
    assert r.warnings == []
    assert "diagnos" in r.disclaimer.lower()


def test_missing_components_reported(lexicon_path):
    sym = RuleBasedSymptomAgent(lexicon_path).analyze("cough")
    r = FusionAgent().fuse(sym, None, None)
    assert r.mode == AnalysisMode.text_only
    assert set(r.missing_components) == {"image", "retrieval"}


def test_image_without_model_is_validation_only(lexicon_path):
    img = ImageAgent(UnavailableAdapter("x", "x", "no weights"), 10**7, 64, 4000).analyze(png_bytes())
    r = FusionAgent().fuse(None, img, None)
    assert r.mode == AnalysisMode.image_validation_only
    assert "image_model_output" in r.missing_components
    assert not [e for e in r.evidence if e.component == "image_agent"]  # nothing invented


def test_reference_labels_compared_only_for_own_image(lexicon_path, tmp_path):
    sym, img, ret = components(lexicon_path, tmp_path)
    case = {"case_id": "CXR1", "dataset_source": "iu_xray", "labels": {"Pleural Effusion": 1},
            "label_source": "MeSH", "synthetic": False}
    new_upload = FusionAgent(MAPS).fuse(sym, img, ret, reference_case=case)
    assert not [i for i in new_upload.fusion_summary if i.category == "disagreement"]
    own = FusionAgent(MAPS).fuse(sym, img, ret, reference_case=case, image_is_reference_image=True)
    dis = [i for i in own.fusion_summary if i.category == "disagreement"]
    assert dis and "Cardiomegaly" in dis[0].statement and "Effusion" in dis[0].statement
    assert any(x.startswith("I") for x in dis[0].supported_by)


def test_synthetic_inputs_force_demo_mode(lexicon_path, tmp_path):
    sym, img, ret = components(lexicon_path, tmp_path, synthetic=True)
    r = FusionAgent().fuse(sym, img, ret)
    assert r.mode == AnalysisMode.demo and r.processing_metadata["synthetic_data_used"]


def test_safety_item_and_text_disagreement(lexicon_path):
    sym = RuleBasedSymptomAgent(lexicon_path).analyze("chest pain. no chest pain.")
    r = FusionAgent().fuse(sym, None, None)
    cats = [i.category for i in r.fusion_summary]
    assert "safety" in cats and "disagreement" in cats