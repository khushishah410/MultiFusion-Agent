"""Read single-file .torrent metadata and verify a downloaded file against its piece hashes.

Used only for integrity checks: no peer-to-peer transfer happens here. The torrent's
SHA-1 piece hashes prove a file matches what the torrent author published; they do
not prove anything about the original publisher.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


class TorrentError(ValueError):
    pass


def _decode(data: bytes, i: int = 0):
    """Return (value, next_index). Dict values keep the raw bytes span of 'info' for the infohash."""
    c = data[i:i + 1]
    if c == b"i":
        end = data.index(b"e", i)
        return int(data[i + 1:end]), end + 1
    if c == b"l":
        i += 1
        out = []
        while data[i:i + 1] != b"e":
            v, i = _decode(data, i)
            out.append(v)
        return out, i + 1
    if c == b"d":
        i += 1
        out = {}
        while data[i:i + 1] != b"e":
            k, i = _decode(data, i)
            start = i
            v, i = _decode(data, i)
            out[k] = v
            if k == b"info":
                out[b"__info_raw__"] = data[start:i]
        return out, i + 1
    if c.isdigit():
        colon = data.index(b":", i)
        n = int(data[i:colon])
        return data[colon + 1:colon + 1 + n], colon + 1 + n
    raise TorrentError(f"Invalid bencode at byte {i}")


@dataclass
class TorrentInfo:
    name: str
    length: int
    piece_length: int
    pieces: list[bytes]
    infohash: str
    web_seeds: list[str] = field(default_factory=list)


def read_torrent(path: Path) -> TorrentInfo:
    data = Path(path).read_bytes()
    try:
        meta, _ = _decode(data)
        info = meta[b"info"]
        if b"files" in info:
            raise TorrentError("Multi-file torrents are not supported")
        raw_pieces = info[b"pieces"]
        seeds = meta.get(b"url-list", [])
        seeds = [seeds] if isinstance(seeds, bytes) else seeds
        return TorrentInfo(
            name=info[b"name"].decode("utf-8"),
            length=info[b"length"],
            piece_length=info[b"piece length"],
            pieces=[raw_pieces[k:k + 20] for k in range(0, len(raw_pieces), 20)],
            infohash=hashlib.sha1(meta[b"__info_raw__"]).hexdigest(),
            web_seeds=[s.decode("utf-8") for s in seeds],
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if b"\xef\xbf\xbd" in data:
            raise TorrentError(f"{Path(path).name} was converted to text somewhere (binary hash bytes were "
                               "replaced with U+FFFD); use the original .torrent file") from exc
        raise TorrentError(f"{Path(path).name}: not a readable single-file torrent ({exc})") from exc


def verify_file(path: Path, torrent: TorrentInfo) -> dict:
    """Check size and every piece hash. Returns {'ok', 'size_ok', 'bad_pieces', 'pieces'}."""
    size = path.stat().st_size
    if size != torrent.length:
        return {"ok": False, "size_ok": False, "bad_pieces": [], "pieces": len(torrent.pieces),
                "detail": f"size {size} bytes, torrent says {torrent.length}"}
    bad = []
    with path.open("rb") as fh:
        for idx, expected in enumerate(torrent.pieces):
            if hashlib.sha1(fh.read(torrent.piece_length)).digest() != expected:
                bad.append(idx)
    return {"ok": not bad, "size_ok": True, "bad_pieces": bad[:20], "pieces": len(torrent.pieces),
            "detail": "all pieces match" if not bad else f"{len(bad)} of {len(torrent.pieces)} pieces differ"}
