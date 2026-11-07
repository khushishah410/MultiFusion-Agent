"""Archive extraction that refuses path traversal, links and device files."""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from app.core.paths import UnsafePathError, is_unsafe_relative


def _check_member(name: str, target: Path) -> Path:
    if is_unsafe_relative(name):
        raise UnsafePathError(f"Archive member has an unsafe path: {name!r}")
    out = (target / name).resolve()
    if not out.is_relative_to(target.resolve()):
        raise UnsafePathError(f"Archive member escapes the target folder: {name!r}")
    return out


def safe_extract(archive: Path, target: Path) -> int:
    """Extract `archive` into `target`. Returns the number of regular files written.

    Existing files with the same size are skipped, so re-running is cheap.
    """
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            for member in tar:
                out = _check_member(member.name, target)
                if member.isdir():
                    out.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise UnsafePathError(f"Archive contains a link or special file: {member.name!r}")
                if out.exists() and out.stat().st_size == member.size:
                    count += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                with src, out.open("wb") as fh:
                    while block := src.read(1024 * 1024):
                        fh.write(block)
                count += 1
        return count
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                out = _check_member(info.filename, target)
                if info.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise UnsafePathError(f"Archive contains a symlink: {info.filename!r}")
                if out.exists() and out.stat().st_size == info.file_size:
                    count += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, out.open("wb") as fh:
                    while block := src.read(1024 * 1024):
                        fh.write(block)
                count += 1
        return count
    raise ValueError(f"{archive.name} is not a tar or zip archive (corrupt or incomplete download?)")
