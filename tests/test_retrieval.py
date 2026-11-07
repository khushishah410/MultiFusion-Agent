import json

import pytest

from app.agents.retrieval_agent import RetrievalAgent, build_index
from scripts.build_retrieval_index import knowledge_docs, manifest_docs
from tests.conftest import write_jsonl


def test_bm25_ranking_and_provenance(tmp_path):
    docs = [{"doc_id": "a", "title": "A", "text": "cough fever cough", "source": "s1", "kind": "document",
             "source_url": "https://example.org/a", "source_date": "2024-01-01"},
            {"doc_id": "b", "title": "B", "text": "fracture of rib", "source": "s2", "kind": "document"},
            {"doc_id": "c", "title": "C", "text": "fever only", "source": "s3", "kind": "reference_case", "synthetic": True}]
    build_index(docs, [{"type": "test"}], tmp_path / "i.json")
    res = RetrievalAgent(tmp_path / "i.json").search("cough fever")
    assert [p.doc_id for p in res.passages] == ["a", "c"]
    assert res.passages[0].source_url == "https://example.org/a" and res.passages[1].synthetic
    assert RetrievalAgent(tmp_path / "i.json").search("zzz").passages == []


def test_corrupt_index_reports_error(tmp_path):
    (tmp_path / "i.json").write_text("{not json")
    agent = RetrievalAgent(tmp_path / "i.json")
    assert agent.status()["status"] == "error" and agent.search("x").status == "error"


def test_build_rejects_incomplete_docs(tmp_path):
    with pytest.raises(ValueError):
        build_index([{"doc_id": "x", "text": "t"}], [], tmp_path / "i.json")


def test_index_sources_exclude_test_split_and_read_knowledge(tmp_path):
    m = write_jsonl(tmp_path / "m.jsonl", [
        {"case_id": "1", "dataset_source": "d", "split": "train", "radiology_findings": "clear lungs"},
        {"case_id": "2", "dataset_source": "d", "split": "test", "radiology_findings": "effusion"}])
    docs = [d for d in manifest_docs(m) if d["split"] != "test"]
    assert [d["doc_id"] for d in docs] == ["d:1"]
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "notes.md").write_text("my permitted note")
    (kb / "src.jsonl").write_text(json.dumps({"id": "x1", "title": "T", "text": "body", "source": "Org"}) + "\n")
    kd = knowledge_docs(kb)
    assert {d["doc_id"] for d in kd} == {"kb:notes", "kb:x1"}
