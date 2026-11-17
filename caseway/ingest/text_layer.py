"""Born-digital shortcut (design: OCR step 2).

If a PDF already has a real text layer, use it directly and skip OCR: many
e-Courts printouts and certified copies are born-digital. A page counts as
born-digital when its text layer holds at least MIN_WORDS real words.
"""

from __future__ import annotations

import io
import re

MIN_WORDS = 15


def pdf_pages_text(data: bytes) -> list[str | None]:
    """Text per page; None for pages that need OCR (no usable text layer)."""
    from pypdf import PdfReader  # local import: optional at runtime

    reader = PdfReader(io.BytesIO(data))
    out: list[str | None] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # malformed page: treat as image-only
            text = ""
        words = re.findall(r"[A-Za-zऀ-ॿ]{2,}", text)
        out.append(text if len(words) >= MIN_WORDS else None)
    return out


def pdf_page_count(data: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(data)).pages)
