"""Text normalisation (design: "Normalise" step of the OCR pipeline).

Every piece of text, whether OCR output, a PDF text layer or the customer's chat,
goes through `normalise()` before extraction, so one set of grammars and one
gazetteer index serve Hindi and English input alike.

    normalise(text)    Unicode NFC, Devanagari digits -> ASCII, OCR look-alike fixes
                       inside numbers, whitespace collapse (newlines are kept,
                       because label-value anchoring stops at line ends).
    fold(text)         case-folded key used for exact dictionary lookups.
    romanise(text)     simple ITRANS-style Devanagari -> Latin transliteration with
                       Hindi schwa deletion, so "कोतवाली" -> "kotvaalii".
    phonetic_key(text) Indic Soundex-like key that merges common transliteration
                       variants, so "Ghaziabad", "Gaziabad", "Ghaziyabad" collide.
"""

from __future__ import annotations

import re
import unicodedata

# --- digits and look-alikes -------------------------------------------------

_DEVANAGARI_DIGITS = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}
# Characters OCR commonly confuses with digits (design: O->0, l/I->1, S->5, B->8).
_LOOKALIKE_TO_DIGIT = {"O": "0", "o": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"}


def _fix_numeric_lookalikes(text: str) -> str:
    """Replace a look-alike letter only when digits sit on BOTH sides of it, so
    '2O23' -> '2023' but 'IPC', '498A' and '120-B' are left alone."""
    chars = list(text)
    for _ in range(2):  # two passes handle runs such as '2OO3'
        for i in range(1, len(chars) - 1):
            c = chars[i]
            if c in _LOOKALIKE_TO_DIGIT and chars[i - 1].isdigit() and chars[i + 1].isdigit():
                chars[i] = _LOOKALIKE_TO_DIGIT[c]
    return "".join(chars)


def normalise(text: str) -> str:
    """Canonical form of any input text. All extraction spans refer to this output."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_DEVANAGARI_DIGITS)
    text = text.replace(" ", " ").replace("\t", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _fix_numeric_lookalikes(text)
    lines = [re.sub(r" {2,}", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def fold(text: str) -> str:
    """Case-folded lookup key (keeps the original script)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip(" .,")


# --- Devanagari -> Latin -----------------------------------------------------

_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri",
           "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au"}
_MATRAS = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
           "े": "e", "ै": "ai", "ो": "o", "ौ": "au"}
_CONSONANTS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
               "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
               "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
               "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
               "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h"}
_NUKTA_FORMS = {"k": "q", "kh": "kh", "g": "g", "j": "z", "d": "r", "dh": "rh", "ph": "f"}
_NUKTA, _VIRAMA = "़", "्"
_SIGNS = {"ं": "n", "ँ": "n", "ः": "h", "।": ".", "॥": "."}


def romanise(text: str) -> str:
    """Transliterate Devanagari to Latin; non-Devanagari characters pass through.

    Works on a token list so it can apply the classic Hindi schwa-deletion rule:
    an inherent 'a' is dropped at the end of a word, and in the context
    vowel-consonant-[a]-consonant-vowel (Ohala's rule)."""
    text = unicodedata.normalize("NFC", text)
    # tokens: (kind, roman) where kind in {"C", "V", "A" (inherent a), "X" (other)}
    tokens: list[list[str]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _CONSONANTS:
            roman = _CONSONANTS[ch]
            if i + 1 < len(text) and text[i + 1] == _NUKTA:
                roman = _NUKTA_FORMS.get(roman, roman)
                i += 1
            tokens.append(["C", roman])
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in _MATRAS:
                tokens.append(["V", _MATRAS[nxt]])
                i += 1
            elif nxt == _VIRAMA:
                i += 1
            else:
                tokens.append(["A", "a"])
        elif ch in _VOWELS:
            tokens.append(["V", _VOWELS[ch]])
        elif ch in _SIGNS:
            tokens.append(["V" if ch in "ंँ" else "X", _SIGNS[ch]])
        elif ch == _NUKTA:
            pass
        else:
            tokens.append(["X", ch])
        i += 1

    def kind(j: int) -> str:
        return tokens[j][0] if 0 <= j < len(tokens) else "X"

    for j, (k, _) in enumerate(tokens):
        if k != "A":
            continue
        word_final = kind(j + 1) == "X" and kind(j - 1) == "C" and kind(j - 2) in ("V", "A")
        medial = (kind(j - 1) == "C" and kind(j - 2) in ("V",) and kind(j + 1) == "C"
                  and kind(j + 2) in ("V", "A"))
        if word_final or medial:
            tokens[j][1] = ""
    return "".join(t[1] for t in tokens)


# --- phonetic key ------------------------------------------------------------

_PHONETIC_RULES = [
    (r"ph", "f"), (r"w", "v"), (r"z", "j"), (r"sh", "s"), (r"q", "k"),
    (r"(?<=[bcdfgjklmnpqrstvxz])h", ""),   # aspirates: kh->k, gh->g, bh->b, th->t ...
    (r"ee|ii|ie|ey", "i"), (r"oo|uu|ou", "u"), (r"aa", "a"),
    (r"(?<=[aeiou])y(?=[aeiou])", ""),     # Ghaziyabad -> Ghaziabad
    (r"([a-z])\1+", r"\1"),                # doubled letters
]


def phonetic_key(text: str) -> str:
    """Key that collapses transliteration variants (v/w, sh/s, z/j, ph/f, aa/a,
    ee/i, oo/u, doubled consonants, aspirates, final schwa)."""
    s = romanise(text).lower()
    s = re.sub(r"[^a-z ]", "", s)
    words = []
    for w in s.split():
        for pattern, repl in _PHONETIC_RULES:
            w = re.sub(pattern, repl, w)
        if len(w) > 3 and w.endswith("a"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)
