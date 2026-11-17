"""OCR adapters and ROVER-style voting (design: OCR steps 3-7).

Production runs two local engines (Tesseract 5 hin+eng and PaddleOCR PP-OCRv4)
on preprocessed pages (deskew, denoise, Sauvola binarisation, stamp removal),
then aligns their word sequences with Needleman-Wunsch and keeps, per word, the
reading with the higher confidence. Bounding boxes travel with each word so the
review console can highlight the exact spot.

v1 status:
* `rover_vote` (the alignment + per-word vote) is implemented and tested.
* `TesseractEngine` works when `pytesseract` and the Tesseract binary are
  installed; `available_engines()` returns whatever is present.
* With no engine installed, image pages are routed to human review instead of
  being guessed (handwritten pages get low confidence the same way).
* Image preprocessing (OpenCV) and PaddleOCR are Phase 1 remaining work.
"""

from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass
class Word:
    text: str
    confidence: float                         # 0..1
    box: tuple[int, int, int, int] | None = None


class OCRUnavailable(RuntimeError):
    pass


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "hin+eng"):
        import pytesseract  # noqa: F401  (raises ImportError if missing)
        self.lang = lang

    def read(self, image_bytes: bytes) -> list[Word]:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        data = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)
        words = []
        for i, text in enumerate(data["text"]):
            if text.strip():
                conf = max(0.0, float(data["conf"][i])) / 100.0
                box = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
                words.append(Word(text, conf, box))
        return words


def available_engines() -> list:
    engines = []
    try:
        engines.append(TesseractEngine())
    except Exception:
        pass
    return engines


def _align(a: list[Word], b: list[Word]) -> list[tuple[Word | None, Word | None]]:
    """Needleman-Wunsch alignment of two word sequences (match 1, mismatch -1, gap -1)."""
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = -i
    for j in range(1, m + 1):
        score[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = 1 if a[i - 1].text.lower() == b[j - 1].text.lower() else -1
            score[i][j] = max(score[i - 1][j - 1] + s, score[i - 1][j] - 1, score[i][j - 1] - 1)
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = 1 if a[i - 1].text.lower() == b[j - 1].text.lower() else -1
            if score[i][j] == score[i - 1][j - 1] + s:
                out.append((a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] - 1:
            out.append((a[i - 1], None))
            i -= 1
        else:
            out.append((None, b[j - 1]))
            j -= 1
    return list(reversed(out))


def rover_vote(a: list[Word], b: list[Word]) -> list[Word]:
    """Combine two engines' readings: per aligned slot keep the higher-confidence
    word. A word only one engine saw is kept at reduced confidence."""
    out = []
    for x, y in _align(a, b):
        if x and y:
            if x.text.lower() == y.text.lower():
                out.append(Word(x.text, min(1.0, max(x.confidence, y.confidence) + 0.05), x.box or y.box))
            else:
                out.append(x if x.confidence >= y.confidence else y)
        else:
            w = x or y
            out.append(Word(w.text, w.confidence * 0.7, w.box))
    return out


def ocr_image(image_bytes: bytes, engines: list | None = None) -> tuple[str, float]:
    """Run all available engines and vote. Returns (text, mean confidence)."""
    engines = engines if engines is not None else available_engines()
    if not engines:
        raise OCRUnavailable("no OCR engine installed (install Tesseract 5 + pytesseract)")
    words = engines[0].read(image_bytes)
    for eng in engines[1:]:
        words = rover_vote(words, eng.read(image_bytes))
    if not words:
        return "", 0.0
    return " ".join(w.text for w in words), sum(w.confidence for w in words) / len(words)
