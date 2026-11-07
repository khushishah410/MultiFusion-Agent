"""Path safety helpers. Every dataset path goes through safe_join."""
from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


class UnsafePathError(ValueError):
    pass


def is_unsafe_relative(relative: str) -> bool:
    """True for absolute paths, drive letters, or any '..' component."""
    if not relative:
        return True
    for flavour in (PurePosixPath(relative), PureWindowsPath(relative)):
        if flavour.is_absolute() or flavour.drive or flavour.root:
            return True
        if ".." in flavour.parts:
            return True
    return False


def safe_join(root: Path, relative: str) -> Path:
    """Join `relative` to `root` and guarantee the result stays inside root."""
    if is_unsafe_relative(relative):
        raise UnsafePathError(f"Rejected unsafe path: {relative!r}")
    root = root.resolve()
    candidate = (root / relative.replace("\\", "/")).resolve()
    if not candidate.is_relative_to(root):
        raise UnsafePathError(f"Path escapes dataset root: {relative!r}")
    return candidate
