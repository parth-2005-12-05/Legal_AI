"""Document-type detection from header keywords (design: OCR step 8).

Rules on header keywords tell us whether a page is an FIR, court order, revenue
record, receipt, notice or affidavit. Each type has its own field template
downstream (e.g. labels in an FIR header are trusted more), and the type is
itself a Tier-1 classifier signal (an FIR implies a criminal matter).
"""

from __future__ import annotations

KEYWORDS: dict[str, list[str]] = {
    "fir": ["first information report", "प्रथम सूचना रिपोर्ट", "f.i.r.", "fir no", "प्रथम सूचना"],
    "court_order": ["in the court of", "high court", "order sheet", "न्यायालय", "judgment", "order dated"],
    "revenue_record": ["khatauni", "jamabandi", "खतौनी", "जमाबंदी", "khasra", "खसरा"],
    "receipt": ["tax invoice", "receipt", "invoice no", "रसीद", "bill no"],
    "notice": ["legal notice", "notice under", "नोटिस", "show cause"],
    "affidavit": ["affidavit", "शपथ पत्र", "हलफनामा"],
}
HEADER_CHARS = 600


def detect_doc_type(text: str) -> tuple[str, float]:
    """Return (doc_type, score). Header hits count double."""
    head, body = text[:HEADER_CHARS].lower(), text.lower()
    scores = {}
    for dtype, words in KEYWORDS.items():
        s = sum(2 for w in words if w in head) + sum(1 for w in words if w in body)
        if s:
            scores[dtype] = s
    if not scores:
        return "unknown", 0.0
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    return best, round(scores[best] / total, 3)
