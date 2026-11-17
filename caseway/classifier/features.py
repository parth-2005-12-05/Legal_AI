"""Feature extraction shared by the BM25 and Naive Bayes models.

Text is romanised first so Hindi, English and Hinglish descriptions share one
vocabulary. Features are word unigrams + bigrams and character 3-5-grams, which
are robust to Hinglish spelling variation and OCR noise.
"""

from __future__ import annotations

import re

from ..normalise import romanise

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "my", "me", "i", "for", "on", "at",
             "has", "have", "was", "with", "by", "from", "it", "this", "that", "hai", "ka", "ki", "ke",
             "ko", "se", "ne", "mein", "aur", "hain", "kar", "ho"}


def words(text: str) -> list[str]:
    return [w for w in _WORD.findall(romanise(text).lower()) if w not in STOPWORDS]


def featurise(text: str, char_ngrams: bool = True) -> list[str]:
    ws = words(text)
    feats = ws + [f"{a}_{b}" for a, b in zip(ws, ws[1:])]
    if char_ngrams:
        for w in ws:
            padded = f"#{w}#"
            for n in (3, 4, 5):
                feats += [f"c:{padded[i:i + n]}" for i in range(len(padded) - n + 1)]
    return feats
