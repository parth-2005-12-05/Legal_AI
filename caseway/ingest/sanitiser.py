"""File sanitiser: the first thing that touches an unsealed upload.

* Real type by magic bytes, never by file name: allow PDF, JPEG, PNG, TIFF, HEIC
  (plus UTF-8 text when `ingest.allow_plain_text` is on, for development).
* Size cap (25 MB per file) and page cap (200 per case).
* PDFs: flag active content (/JavaScript, /JS, /Launch, /EmbeddedFile, /OpenAction).
  v1 never executes or renders such content; it only reads the text layer, and
  flags the file so the qpdf / mutool flattening step (Phase 1 remaining work)
  can strip it before any rendering.
* ClamAV: if `clamdscan` or `clamscan` is installed it is run; otherwise the
  result records that the scan was unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ACTIVE_PDF_MARKERS = [b"/JavaScript", b"/JS", b"/Launch", b"/EmbeddedFile", b"/OpenAction", b"/AA"]


class RejectedFile(ValueError):
    """The upload failed a hard sanitiser check and must not be processed."""


@dataclass
class SanitiseResult:
    mime: str
    size: int
    flags: list[str] = field(default_factory=list)
    av_scan: str = "unavailable"


def sniff_type(data: bytes, allow_text: bool) -> str | None:
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return "image/heic"
    if allow_text and b"\x00" not in data[:4096]:
        try:
            data.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            return None
    return None


def _clamav(data: bytes) -> str:
    exe = shutil.which("clamdscan") or shutil.which("clamscan")
    if not exe:
        return "unavailable"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(data)
        name = tmp.name
    try:
        proc = subprocess.run([exe, "--no-summary", name], capture_output=True, timeout=120)
        return "clean" if proc.returncode == 0 else "infected"
    finally:
        Path(name).unlink(missing_ok=True)


def sanitise(data: bytes, max_mb: int = 25, allow_text: bool = True) -> SanitiseResult:
    if len(data) > max_mb * 1024 * 1024:
        raise RejectedFile(f"file larger than {max_mb} MB")
    mime = sniff_type(data, allow_text)
    if mime is None:
        raise RejectedFile("file type not allowed (PDF, JPEG, PNG, TIFF, HEIC only)")
    res = SanitiseResult(mime=mime, size=len(data))
    if mime == "application/pdf":
        found = [m.decode() for m in ACTIVE_PDF_MARKERS if m in data]
        if found:
            res.flags.append("pdf_active_content:" + ",".join(found))
    res.av_scan = _clamav(data)
    if res.av_scan == "infected":
        raise RejectedFile("malware detected")
    return res
