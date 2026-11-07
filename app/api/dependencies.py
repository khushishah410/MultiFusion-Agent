"""Application state: agents, active manifest and services, built once and reloadable."""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path

from app.agents.diagnosis_agent import DiagnosisAgent
from app.agents.fusion_agent import FusionAgent
from app.agents.image_agent import ImageAgent
from app.agents.model_adapters import build_adapter
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.symptom_agent import DatasetTextIndex, RuleBasedSymptomAgent
from app.core.config import Settings, get_settings
from app.services.dataset_ops import DatasetOps
from app.services.evaluation_service import load_label_map
from app.services.jobs import JobManager
from app.services.manifest_service import ManifestStore

log = logging.getLogger(__name__)


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs = JobManager()
        self._lock = threading.Lock()
        self.adapter = build_adapter(settings)
        self.reload()

    def reload(self) -> None:
        with self._lock:
            s = self.settings
            manifest, how = s.active_manifest()
            self.store = ManifestStore(manifest, None, how)
            self.dataset_ops = DatasetOps(s)
            index = DatasetTextIndex(self.store.report_texts())
            self.symptom_agent = RuleBasedSymptomAgent(s.resolve(s.symptom_lexicon), index)
            self.image_agent = ImageAgent(self.adapter, s.max_upload_bytes, s.min_image_side, s.max_image_side)
            self.retrieval_agent = RetrievalAgent(s.resolve(s.retrieval_index))
            maps_dir = s.resolve("config/label_maps")
            self.label_maps = {p.stem: load_label_map(p) for p in Path(maps_dir).glob("*.yaml")}
            self.fusion_agent = FusionAgent(self.label_maps)
            self.diagnosis_agent = DiagnosisAgent(s.resolve(s.diagnosis_rules))
            log.info("State loaded: manifest=%s (%s), %d cases, retrieval=%s",
                     manifest.name if manifest else None, how, len(self.store.records),
                     self.retrieval_agent.status()["status"])


@lru_cache
def get_state() -> AppState:
    return AppState(get_settings())
