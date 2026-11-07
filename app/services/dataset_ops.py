"""Dataset operations shared by the CLI (scripts/dataset_manager.py) and the API."""
from __future__ import annotations

import glob
import json
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import REPO_ROOT, Settings
from app.services import downloader
from app.services.archive import safe_extract
from app.services.manifest_service import build_manifest, validate_manifest, write_validation
from app.services.registry import DatasetSpec, load_registry
from app.services.torrent import TorrentError, read_torrent, verify_file


class DatasetOpError(RuntimeError):
    pass


class ConfirmationRequired(DatasetOpError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DatasetOps:
    def __init__(self, settings: Settings, root_override: Path | None = None):
        self.settings = settings
        self.registry = load_registry(settings.resolve(settings.datasets_config))
        self.data_root = root_override.resolve() if root_override else settings.data_root_abs
        self.manifest_dir = settings.manifest_dir_abs

    # ------------------------------------------------------------ lookup
    def spec(self, dataset_id: str) -> DatasetSpec:
        if dataset_id not in self.registry:
            raise DatasetOpError(f"Unknown dataset {dataset_id!r}. Known: {', '.join(self.registry)}")
        return self.registry[dataset_id]

    def download_dir(self, dataset_id: str) -> Path:
        return self.data_root / dataset_id

    def dataset_root(self, dataset_id: str) -> Path:
        """Folder holding the dataset files: an imported folder, or data_root/<id>."""
        status = downloader.read_status(self.download_dir(dataset_id))
        local = status.get("local_root")
        return Path(local) if local else self.download_dir(dataset_id)

    def status(self, dataset_id: str) -> dict:
        spec = self.spec(dataset_id)
        st = downloader.read_status(self.download_dir(dataset_id))
        manifest = self.manifest_dir / f"{dataset_id}.jsonl"
        validation = self.manifest_dir / f"{dataset_id}.validation.json"
        val = json.loads(validation.read_text(encoding="utf-8")) if validation.exists() else None
        return {
            "id": dataset_id,
            "display_name": spec.display_name,
            "strategy": spec.strategy,
            "downloaded_files": st.get("files", {}),
            "download_state": st.get("download_state", "not_started"),
            "imported_from_local_folder": bool(st.get("local_root")),
            "prepared": manifest.exists(),
            "manifest": manifest.name if manifest.exists() else None,
            "prepared_at": st.get("prepared_at"),
            "validation_ok": val["ok"] if val else None,
            "validated_at": val["generated_at"] if val else None,
            "active": self.active_dataset() == dataset_id,
            "last_error": st.get("last_error"),
        }

    def active_dataset(self) -> str | None:
        manifest, _ = self.settings.active_manifest()
        return manifest.stem if manifest else None

    # ------------------------------------------------------------ inspect
    def inspect(self, dataset_id: str, check_remote: bool = False) -> dict:
        spec = self.spec(dataset_id)
        info = spec.public_dict()
        target = self.download_dir(dataset_id)
        probe = target if target.exists() else self.data_root
        probe.mkdir(parents=True, exist_ok=True)
        info["storage_dir"] = str(target)
        info["disk_free_gb"] = round(shutil.disk_usage(probe).free / 1e9, 2)
        info["status"] = self.status(dataset_id)
        root = self.dataset_root(dataset_id)
        info["expected_files_present"] = {
            pattern: bool(glob.glob(str(root / pattern), recursive=True)) for pattern in spec.expected_files
        } if root.exists() else {}
        if spec.strategy == "manual":
            info["prerequisites"] = [spec.instructions or "See the official source.",
                                     f"Then run: python -m scripts.dataset_manager import --dataset {dataset_id} "
                                     "--source-dir <folder>"]
        if check_remote and spec.strategy == "http":
            sizes = {}
            for f in spec.files:
                try:
                    sizes[f.name] = downloader.remote_size(f.url)
                except downloader.DownloadError as exc:
                    sizes[f.name] = f"unreachable: {exc}"
            info["remote_sizes_bytes"] = sizes
        return info

    # ------------------------------------------------------------ download
    def plan_download(self, dataset_id: str, metadata_only: bool = False) -> list:
        spec = self.spec(dataset_id)
        if spec.strategy != "http" or not spec.automated_download_allowed:
            raise DatasetOpError(
                f"{spec.display_name} cannot be downloaded automatically ({spec.strategy}). "
                f"{spec.instructions or ''} Official source: {spec.official_source}")
        return [f for f in spec.files if f.metadata_only or not metadata_only]

    def download(self, dataset_id: str, metadata_only: bool = False, confirm_large: bool = False,
                 dry_run: bool = False, progress: Callable[[str, int, int | None], None] | None = None) -> dict:
        files = self.plan_download(dataset_id, metadata_only)
        total_mb = sum(f.documented_size_mb or 0 for f in files)
        target = self.download_dir(dataset_id)
        plan = {"dataset": dataset_id, "target": str(target),
                "files": [{"name": f.name, "url": f.url, "documented_size_mb": f.documented_size_mb} for f in files],
                "documented_total_mb": total_mb}
        if dry_run:
            return plan | {"dry_run": True}
        if total_mb > self.settings.large_download_mb and not confirm_large:
            raise ConfirmationRequired(
                f"This download is about {total_mb:,.0f} MB (documented size). Confirm explicitly "
                f"(--yes on the CLI) or use --metadata-only for the {sum(f.documented_size_mb or 0 for f in files if f.metadata_only):.1f} MB reports only.")
        downloader.write_status(target, {"download_state": "in_progress", "last_error": None})
        results = {}
        try:
            for f in files:
                cb = (lambda done, total, name=f.name: progress(name, done, total)) if progress else None
                res, served_from, failures = None, None, []
                for url in f.urls:
                    try:
                        res = downloader.download_file(url, target / f.name, expected_size=f.size_bytes,
                                                       expected_sha256=f.sha256, progress=cb)
                        served_from = url
                        break
                    except downloader.DownloadError as exc:
                        failures.append(f"{url}: {exc}")
                if res is None:
                    raise downloader.DownloadError(f"{f.name}: every source failed:\n  " + "\n  ".join(failures))
                check = self.check_archive(dataset_id, f.name)
                results[f.name] = {"bytes": res.bytes, "sha256": res.sha256, "url": served_from,
                                   "from_mirror": served_from != f.url, "failed_sources": failures,
                                   "resumed": res.resumed, "skipped_existing": res.skipped_existing,
                                   "downloaded_at": _now(), "integrity": check,
                                   "checksum_verified_against_official": bool(f.sha256)}
                if not check["ok"]:
                    raise downloader.DownloadError(f"{f.name}: integrity check failed ({check['detail']}). "
                                                   "The file was kept for inspection and not extracted.")
                if f.extract:
                    results[f.name]["extracted_files"] = safe_extract(target / f.name, target)
                st = downloader.read_status(target)
                downloader.write_status(target, {"files": st.get("files", {}) | {f.name: results[f.name]}})
        except Exception as exc:
            downloader.write_status(target, {"download_state": "failed", "last_error": str(exc)})
            raise
        state = "metadata_only" if metadata_only and len(files) < len(self.spec(dataset_id).files) else "complete"
        downloader.write_status(target, {"download_state": state, "source_version": "as served on download date",
                                         "official_source": self.spec(dataset_id).official_source})
        return plan | {"results": results, "download_state": state}

    # ------------------------------------------------------------ integrity of archives
    def check_archive(self, dataset_id: str, name: str, torrent_path: Path | None = None) -> dict:
        """Check an archive in the download folder: exact size, then torrent piece hashes when available."""
        f = next((x for x in self.spec(dataset_id).files if x.name == name), None)
        if f is None:
            raise DatasetOpError(f"{name} is not a file of {dataset_id}")
        path = self.download_dir(dataset_id) / name
        if not path.is_file():
            return {"ok": False, "present": False, "detail": f"{name} not found in {path.parent}"}
        size = path.stat().st_size
        out = {"present": True, "bytes": size, "expected_bytes": f.size_bytes,
               "size_ok": f.size_bytes is None or size == f.size_bytes, "torrent": None}
        if not out["size_ok"]:
            return out | {"ok": False, "detail": f"size {size:,} bytes, expected {f.size_bytes:,} "
                                                 "(incomplete or different file)"}
        tpath = torrent_path or (REPO_ROOT / f.torrent if f.torrent else None)
        if tpath is None:
            return out | {"ok": True, "detail": "exact size matches; no piece hashes available"}
        try:
            t = read_torrent(tpath)
        except (OSError, TorrentError) as exc:
            return out | {"ok": False, "detail": f"torrent unusable: {exc}"}
        if t.name != name or t.length != size or (f.torrent_infohash and t.infohash != f.torrent_infohash):
            return out | {"ok": False, "detail": f"torrent {tpath.name} describes {t.name} ({t.length} bytes, "
                                                 f"infohash {t.infohash}), not this file"}
        res = verify_file(path, t)
        return out | {"ok": res["ok"], "torrent": {"infohash": t.infohash, **res},
                      "detail": f"torrent piece hashes: {res['detail']}"}

    def ingest_placed_archives(self, dataset_id: str) -> dict:
        """Verify and extract archives the user placed in the download folder by hand (browser, torrent)."""
        spec = self.spec(dataset_id)
        target = self.download_dir(dataset_id)
        st = downloader.read_status(target)
        done = {}
        for f in spec.files:
            if not f.extract or not (target / f.name).is_file() or st.get("files", {}).get(f.name, {}).get("extracted_files"):
                continue
            check = self.check_archive(dataset_id, f.name)
            if not check["ok"]:
                raise DatasetOpError(f"{f.name}: {check['detail']}. Not extracted.")
            done[f.name] = {"bytes": check["bytes"], "url": None, "placed_manually": True, "integrity": check,
                            "verified_at": _now(), "extracted_files": safe_extract(target / f.name, target)}
        if done:
            st = downloader.read_status(target)
            downloader.write_status(target, {"files": st.get("files", {}) | done, "last_error": None,
                                             "download_state": "placed_manually"})
        return done

    # ------------------------------------------------------------ import (manual datasets)
    def import_local(self, dataset_id: str, source_dir: Path) -> dict:
        spec = self.spec(dataset_id)
        source_dir = source_dir.expanduser().resolve()
        if not source_dir.is_dir():
            raise DatasetOpError(f"Folder not found: {source_dir}")
        present = {p: bool(glob.glob(str(source_dir / p), recursive=True)) for p in spec.expected_files}
        missing = [p for p, ok in present.items() if not ok]
        if missing and spec.expected_files and all(not v for v in present.values()):
            raise DatasetOpError(f"None of the expected files {spec.expected_files} were found in {source_dir}.")
        downloader.write_status(self.download_dir(dataset_id), {
            "local_root": str(source_dir), "download_state": "imported_local", "imported_at": _now(),
            "expected_files_present": present, "last_error": None})
        return {"dataset": dataset_id, "local_root": str(source_dir), "expected_files_present": present,
                "warning": f"Missing expected files: {missing}" if missing else None,
                "note": "Files were not copied; the manifest will reference them in place (read-only)."}

    # ------------------------------------------------------------ prepare / validate / activate
    def prepare(self, dataset_id: str, source_dir: Path | None = None, options: dict | None = None) -> dict:
        spec = self.spec(dataset_id)
        if source_dir:
            self.import_local(dataset_id, source_dir)
        if spec.strategy == "http" and not downloader.read_status(self.download_dir(dataset_id)).get("local_root"):
            self.ingest_placed_archives(dataset_id)
        root = self.dataset_root(dataset_id)
        if not root.exists():
            raise DatasetOpError(f"No files for {dataset_id} at {root}. Download or import them first.")
        st = downloader.read_status(self.download_dir(dataset_id))
        source_info = {"official_source": spec.official_source, "license": spec.license,
                       "download_state": st.get("download_state"),
                       "files": {k: {"url": v.get("url"), "downloaded_at": v.get("downloaded_at"), "bytes": v.get("bytes"),
                                     "sha256": v.get("sha256")} for k, v in st.get("files", {}).items()}}
        meta = build_manifest(dataset_id, spec.manifest_builder, root, self.manifest_dir, options, source_info)
        downloader.write_status(self.download_dir(dataset_id), {"prepared_at": meta["generated_at"]})
        return meta

    def validate(self, dataset_id: str, check_images: bool = True, max_images: int | None = None) -> dict:
        manifest = self.manifest_dir / f"{dataset_id}.jsonl"
        if not manifest.exists():
            raise DatasetOpError(f"No manifest for {dataset_id}; run prepare first.")
        report = validate_manifest(manifest, self.dataset_root(dataset_id), check_images, max_images)
        json_path, md_path = write_validation(report, self.manifest_dir)
        return report.model_dump() | {"report_files": [json_path.name, md_path.name]}

    def activate(self, dataset_id: str) -> dict:
        manifest = self.manifest_dir / f"{dataset_id}.jsonl"
        if not manifest.exists():
            raise DatasetOpError(f"No manifest for {dataset_id}; run prepare first.")
        rel = manifest.relative_to(REPO_ROOT).as_posix() if manifest.is_relative_to(REPO_ROOT) else str(manifest)
        (self.manifest_dir / "active.json").write_text(
            json.dumps({"manifest": rel, "activated_at": _now()}, indent=2), encoding="utf-8")
        return {"active": dataset_id, "manifest": rel,
                "note": "Restart the backend (or call POST /admin/reload) to load it."}

    def cleanup_instructions(self, dataset_id: str) -> list[str]:
        return [
            "Nothing is deleted automatically. To free space, delete these yourself if you no longer need them:",
            f"  downloaded archives: {self.download_dir(dataset_id)}\\*.tgz (after extraction)",
            f"  extracted files:     {self.download_dir(dataset_id)}",
            f"  manifest + reports:  {self.manifest_dir / dataset_id}.*",
            "Imported (manual) datasets are referenced in place and are never modified by this project.",
        ]
