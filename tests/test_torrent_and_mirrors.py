"""Torrent-based integrity checks, mirror fallback and hand-placed archives (no network)."""
import hashlib
import io
import tarfile

import pytest
import yaml

from app.core.config import REPO_ROOT
from app.services import downloader
from app.services.dataset_ops import DatasetOpError, DatasetOps
from app.services.torrent import TorrentError, read_torrent, verify_file
from tests.conftest import IU_XML, png_bytes

PIECE = 16384


def bencode(v) -> bytes:
    if isinstance(v, int):
        return b"i%de" % v
    if isinstance(v, str):
        v = v.encode()
    if isinstance(v, bytes):
        return b"%d:%s" % (len(v), v)
    if isinstance(v, list):
        return b"l" + b"".join(bencode(x) for x in v) + b"e"
    return b"d" + b"".join(bencode(k) + bencode(v[k]) for k in sorted(v)) + b"e"


def make_torrent(path, data: bytes, name: str):
    pieces = b"".join(hashlib.sha1(data[i:i + PIECE]).digest() for i in range(0, len(data), PIECE))
    info = {"length": len(data), "name": name, "piece length": PIECE, "pieces": pieces}
    path.write_bytes(bencode({"announce": "https://example.org/a", "info": info, "url-list": ["https://m/x"]}))
    return hashlib.sha1(bencode(info)).hexdigest()


def iu_archive() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        xml = IU_XML.format(n=1, indication="cough", findings="f", impression="i",
                            mesh="<major>Cardiomegaly</major>",
                            images='<parentImage id="CXR1_1_IM-0001-1001"/>').encode()
        for name, data in (("ecgen-radiology/1.xml", xml), ("CXR1_1_IM-0001-1001.png", png_bytes())):
            info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_bundled_png_torrent_matches_academic_torrents_id():
    t = read_torrent(REPO_ROOT / "config/torrents/NLMCXR_png.tgz.torrent")
    assert t.infohash == "5a3a439df24931f410fac269b87b050203d9467d"
    assert t.name == "NLMCXR_png.tgz" and t.length == 1_360_814_128 and len(t.pieces) == 325


def test_verify_detects_good_and_corrupted_files(tmp_path):
    data = bytes(range(256)) * 300
    make_torrent(tmp_path / "f.torrent", data, "f.bin")
    t = read_torrent(tmp_path / "f.torrent")
    (tmp_path / "f.bin").write_bytes(data)
    assert verify_file(tmp_path / "f.bin", t)["ok"]
    bad = bytearray(data); bad[PIECE + 5] ^= 0xFF
    (tmp_path / "f.bin").write_bytes(bytes(bad))
    res = verify_file(tmp_path / "f.bin", t)
    assert not res["ok"] and res["bad_pieces"] == [1]
    (tmp_path / "f.bin").write_bytes(data[:-1])
    assert not verify_file(tmp_path / "f.bin", t)["size_ok"]


def test_text_converted_torrent_is_reported(tmp_path):
    make_torrent(tmp_path / "f.torrent", b"x" * 100, "f.bin")
    raw = (tmp_path / "f.torrent").read_bytes()
    start = raw.index(b"6:pieces20:") + len(b"6:pieces20:")
    damaged = raw[:start] + b"\xef\xbf\xbd" * 8 + raw[start + 20:]
    (tmp_path / "f.torrent").write_bytes(damaged)
    with pytest.raises(TorrentError, match="converted to text"):
        read_torrent(tmp_path / "f.torrent")


@pytest.fixture
def toy_ops(settings, tmp_path):
    """Registry with one auto-download archive described by a real torrent built for it."""
    archive = iu_archive()
    torrent = tmp_path / "toy.torrent"
    infohash = make_torrent(torrent, archive, "toy.tgz")
    reg = {"datasets": {"toy": {
        "display_name": "Toy", "official_source": "https://official/toy", "license": "test",
        "automated_download_allowed": True, "manifest_builder": "iu_xray_xml",
        "download": {"strategy": "http", "files": [{
            "name": "toy.tgz", "url": "https://official/toy.tgz", "mirrors": ["https://mirror/toy.tgz"],
            "size_bytes": len(archive), "documented_size_mb": 0.01, "torrent": str(torrent),
            "torrent_infohash": infohash, "extract": True}]}}}}
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml.safe_dump(reg))
    return DatasetOps(settings.model_copy(update={"datasets_config": cfg})), archive


class RoutingSession:
    """Fake HTTP: official host down, mirror serves `body`."""

    def __init__(self, body):
        self.body, self.urls = body, []

    def get(self, url, headers=None, **kw):
        from tests.test_downloads import FakeResp
        self.urls.append(url)
        if "official" in url:
            return FakeResp(503)
        return FakeResp(200, self.body, {"Content-Length": str(len(self.body))})


def _patch_session(monkeypatch, session):
    real = downloader.download_file
    monkeypatch.setattr(downloader, "download_file", lambda url, dest, **kw: real(url, dest, session=session, **kw))


def test_mirror_fallback_verifies_and_extracts(toy_ops, monkeypatch):
    ops, archive = toy_ops
    s = RoutingSession(archive)
    _patch_session(monkeypatch, s)
    res = ops.download("toy")
    r = res["results"]["toy.tgz"]
    assert s.urls == ["https://official/toy.tgz", "https://mirror/toy.tgz"]
    assert r["from_mirror"] and r["integrity"]["ok"] and r["integrity"]["torrent"]["detail"] == "all pieces match"
    assert r["extracted_files"] == 2 and "503" in r["failed_sources"][0]


def test_corrupted_mirror_file_is_not_extracted(toy_ops, monkeypatch):
    ops, archive = toy_ops
    bad = bytearray(archive); bad[10] ^= 0xFF
    _patch_session(monkeypatch, RoutingSession(bytes(bad)))
    with pytest.raises(downloader.DownloadError, match="integrity check failed"):
        ops.download("toy")
    assert not (ops.download_dir("toy") / "ecgen-radiology").exists()
    assert ops.status("toy")["download_state"] == "failed"


def test_hand_placed_archive_verified_then_prepared(toy_ops):
    ops, archive = toy_ops
    folder = ops.download_dir("toy"); folder.mkdir(parents=True)
    (folder / "toy.tgz").write_bytes(archive[:-10])
    with pytest.raises(DatasetOpError, match="expected"):
        ops.prepare("toy")
    (folder / "toy.tgz").write_bytes(archive)
    assert ops.check_archive("toy", "toy.tgz")["ok"]
    meta = ops.prepare("toy")
    assert meta["records_written"] == 1
    st = ops.status("toy")
    assert st["download_state"] == "placed_manually" and st["downloaded_files"]["toy.tgz"]["placed_manually"]


def test_registry_iu_sizes_and_mirrors(settings):
    ops = DatasetOps(settings)
    files = {f.name: f for f in ops.spec("iu_xray").files}
    assert files["NLMCXR_reports.tgz"].size_bytes == 1_112_632
    assert files["NLMCXR_png.tgz"].size_bytes == 1_360_814_128
    assert all(len(f.urls) >= 3 and f.urls[0].startswith("https://openi.nlm.nih.gov") for f in files.values())
