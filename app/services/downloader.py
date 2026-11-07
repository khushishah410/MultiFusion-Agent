"""Resumable, streaming HTTP downloads with integrity checks and status records."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)
CHUNK = 1024 * 1024
TIMEOUT = (15, 60)
DISK_HEADROOM = 1.10  # need 10% more free space than the file


class DownloadError(RuntimeError):
    pass


@dataclass
class DownloadResult:
    name: str
    path: Path
    bytes: int
    sha256: str
    resumed: bool
    skipped_existing: bool
    url: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def remote_size(url: str, session: requests.Session | None = None) -> int | None:
    s = session or requests.Session()
    try:
        r = s.head(url, allow_redirects=True, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise DownloadError(f"Could not reach {url}: {exc}") from exc
    if r.status_code >= 400:
        raise DownloadError(f"HEAD {url} returned HTTP {r.status_code}")
    length = r.headers.get("Content-Length")
    return int(length) if length and length.isdigit() else None


def check_disk_space(target_dir: Path, needed_bytes: int) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target_dir).free
    if free < needed_bytes * DISK_HEADROOM:
        raise DownloadError(
            f"Not enough disk space in {target_dir}: {free / 1e9:.2f} GB free, "
            f"about {needed_bytes * DISK_HEADROOM / 1e9:.2f} GB needed. Free space or pass a different --root."
        )


def download_file(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    session: requests.Session | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> DownloadResult:
    """Download `url` to `dest`, resuming from `dest.part` when the server supports ranges.

    A completed file whose size (and checksum, if published) matches is not downloaded again.
    """
    s = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        digest = sha256_file(dest)
        if expected_sha256 and digest != expected_sha256:
            raise DownloadError(f"{dest.name}: existing file checksum mismatch; delete it and retry.")
        return DownloadResult(dest.name, dest, dest.stat().st_size, digest, False, True, url)

    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        resp = s.get(url, headers=headers, stream=True, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException as exc:
        raise DownloadError(f"Could not reach {url}: {exc}") from exc

    with resp:
        if resp.status_code == 416 and expected_size and offset == expected_size:
            pass  # .part already complete
        elif resp.status_code == 206 and offset:
            log.info("Resuming %s at %d bytes", dest.name, offset)
        elif resp.status_code == 200:
            if offset:
                log.info("Server ignored the range request; restarting %s", dest.name)
            offset = 0
        else:
            raise DownloadError(f"GET {url} returned HTTP {resp.status_code}")

        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype:
            raise DownloadError(
                f"{url} returned an HTML page instead of a file (Content-Type {ctype}). "
                "The link may require a login or may have moved; check the official source."
            )
        total = expected_size
        if total is None and resp.headers.get("Content-Length", "").isdigit():
            total = int(resp.headers["Content-Length"]) + (offset if resp.status_code == 206 else 0)
        if total:
            check_disk_space(dest.parent, total - offset)

        if resp.status_code != 416:
            mode = "ab" if resp.status_code == 206 else "wb"
            done = offset if mode == "ab" else 0
            with part.open(mode) as fh:
                for chunk in resp.iter_content(CHUNK):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)

    size = part.stat().st_size
    if total is not None and size != total:
        raise DownloadError(f"{dest.name}: incomplete download ({size} of {total} bytes). Re-run to resume.")
    digest = sha256_file(part)
    if expected_sha256 and digest != expected_sha256:
        part.unlink()
        raise DownloadError(f"{dest.name}: checksum mismatch (got {digest}); the partial file was removed.")
    part.replace(dest)
    return DownloadResult(dest.name, dest, size, digest, bool(offset), False, url)


def write_status(dataset_dir: Path, update: dict) -> dict:
    path = dataset_dir / ".dataset_status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    status.update(update)
    status["updated_at"] = _now()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def read_status(dataset_dir: Path) -> dict:
    path = dataset_dir / ".dataset_status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
