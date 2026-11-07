"""Download, extraction and dataset-manager logic with a fake HTTP session (no network)."""
import hashlib
import io
import tarfile

import pytest

from app.core.paths import UnsafePathError
from app.services import downloader
from app.services.archive import safe_extract
from app.services.dataset_ops import ConfirmationRequired, DatasetOpError, DatasetOps
from app.services.registry import load_registry

PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes


class FakeResp:
    def __init__(self, status, body=b"", headers=None):
        self.status_code, self._body = status, body
        self.headers = {"Content-Type": "application/octet-stream", **(headers or {})}

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    def __init__(self, honour_range=True, body=PAYLOAD, content_type=None, truncate=False):
        self.honour_range, self.body, self.ct, self.truncate = honour_range, body, content_type, truncate
        self.requests = []

    def head(self, url, **kw):
        return FakeResp(200, headers={"Content-Length": str(len(self.body))})

    def get(self, url, headers=None, **kw):
        self.requests.append(headers or {})
        extra = {"Content-Type": self.ct} if self.ct else {}
        rng = (headers or {}).get("Range")
        if rng and self.honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            return FakeResp(206, self.body[start:], {"Content-Length": str(len(self.body) - start), **extra})
        body = self.body[: len(self.body) // 2] if self.truncate else self.body
        return FakeResp(200, body, {"Content-Length": str(len(self.body)), **extra})


def test_full_download_and_skip_existing(tmp_path):
    dest = tmp_path / "f.bin"
    res = downloader.download_file("https://x/f.bin", dest, session=FakeSession())
    assert dest.read_bytes() == PAYLOAD and res.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    again = downloader.download_file("https://x/f.bin", dest, expected_size=len(PAYLOAD), session=FakeSession())
    assert again.skipped_existing


def test_resume_from_partial(tmp_path):
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(PAYLOAD[:1000])
    s = FakeSession()
    res = downloader.download_file("https://x/f.bin", dest, session=s)
    assert s.requests[0] == {"Range": "bytes=1000-"} and res.resumed and dest.read_bytes() == PAYLOAD


def test_server_ignoring_range_restarts(tmp_path):
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(b"garbage")
    downloader.download_file("https://x/f.bin", dest, session=FakeSession(honour_range=False))
    assert dest.read_bytes() == PAYLOAD


def test_incomplete_download_detected(tmp_path):
    with pytest.raises(downloader.DownloadError, match="incomplete"):
        downloader.download_file("https://x/f.bin", tmp_path / "f.bin", session=FakeSession(truncate=True))
    assert (tmp_path / "f.bin.part").exists() and not (tmp_path / "f.bin").exists()


def test_html_login_page_rejected(tmp_path):
    with pytest.raises(downloader.DownloadError, match="HTML"):
        downloader.download_file("https://x/f", tmp_path / "f", session=FakeSession(content_type="text/html"))


def test_checksum_mismatch_removes_partial(tmp_path):
    with pytest.raises(downloader.DownloadError, match="checksum"):
        downloader.download_file("https://x/f", tmp_path / "f", expected_sha256="0" * 64, session=FakeSession())
    assert not (tmp_path / "f.part").exists()


def _tar(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "sym":
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return path


def test_safe_extract_ok_and_idempotent(tmp_path):
    arc = _tar(tmp_path / "a.tgz", [("ecgen-radiology/1.xml", b"<x/>", "f"), ("img.png", b"123", "f")])
    assert safe_extract(arc, tmp_path / "out") == 2
    assert safe_extract(arc, tmp_path / "out") == 2
    assert (tmp_path / "out" / "ecgen-radiology" / "1.xml").read_bytes() == b"<x/>"


@pytest.mark.parametrize("members", [[("../evil.txt", b"x", "f")], [("/abs.txt", b"x", "f")],
                                     [("link", b"", "sym")]])
def test_safe_extract_rejects_unsafe(tmp_path, members):
    arc = _tar(tmp_path / "bad.tgz", members)
    with pytest.raises(UnsafePathError):
        safe_extract(arc, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_corrupt_archive_detected(tmp_path):
    (tmp_path / "x.tgz").write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="not a tar or zip"):
        safe_extract(tmp_path / "x.tgz", tmp_path / "out")


def test_registry_entries_are_consistent(settings):
    reg = load_registry(settings.resolve(settings.datasets_config))
    assert {"iu_xray", "nih_cxr14", "chexpert", "mimic_cxr_jpg", "vindr_cxr", "padchest"} <= set(reg)
    auto = [d for d in reg.values() if d.automated_download_allowed]
    assert [d.id for d in auto] == ["iu_xray"]
    assert all(f.url.startswith("https://") for f in reg["iu_xray"].files)
    for gated in ("chexpert", "mimic_cxr_jpg", "vindr_cxr", "padchest", "nih_cxr14"):
        assert reg[gated].strategy == "manual" and not reg[gated].files


def test_manager_refuses_gated_and_requires_confirmation(settings):
    ops = DatasetOps(settings)
    with pytest.raises(DatasetOpError, match="cannot be downloaded automatically"):
        ops.download("mimic_cxr_jpg")
    with pytest.raises(ConfirmationRequired):
        ops.download("iu_xray")
    plan = ops.download("iu_xray", dry_run=True)
    assert plan["dry_run"] and len(plan["files"]) == 2
    assert not (settings.data_root_abs / "iu_xray" / ".dataset_status.json").exists()  # dry run writes nothing
    meta_plan = ops.download("iu_xray", metadata_only=True, dry_run=True)
    assert [f["name"] for f in meta_plan["files"]] == ["NLMCXR_reports.tgz"]


def test_failed_download_recorded_not_marked_complete(settings, monkeypatch):
    def boom(*a, **k):
        raise downloader.DownloadError("host unreachable")
    monkeypatch.setattr(downloader, "download_file", boom)
    ops = DatasetOps(settings)
    with pytest.raises(downloader.DownloadError):
        ops.download("iu_xray", metadata_only=True)
    st = ops.status("iu_xray")
    assert st["download_state"] == "failed" and "unreachable" in st["last_error"] and not st["prepared"]


def test_import_local_folder_for_gated_dataset(settings, tmp_path):
    ops = DatasetOps(settings)
    with pytest.raises(DatasetOpError, match="None of the expected files"):
        ops.import_local("chexpert", tmp_path)
    (tmp_path / "train.csv").write_text("Path,Cardiomegaly\n")
    res = ops.import_local("chexpert", tmp_path)
    assert res["expected_files_present"]["train.csv"] and ops.dataset_root("chexpert") == tmp_path.resolve()
