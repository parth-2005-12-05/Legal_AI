"""Aho-Corasick multi-pattern matcher (design: Layer 3, gazetteer scan).

One automaton holds every gazetteer alias (and the bilingual field labels), each
pattern pointing to its payloads. One linear pass over a document finds every
mention in O(n + z), independent of how many names the gazetteer holds.

After matching we enforce word boundaries and keep the longest match where
matches overlap ("Sadar Bazar" beats "Sadar"). Pure Python so it installs
anywhere offline; `pyahocorasick` can replace it later behind the same API.
"""

from __future__ import annotations

import unicodedata
from collections import deque
from dataclasses import dataclass
from typing import Any


def lower_same_length(text: str) -> str:
    """Lower-case each character but never change the string length, so match
    offsets on the lowered text are valid offsets into the original."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


def is_word_char(c: str) -> bool:
    # Devanagari vowel signs / virama are combining marks (category M*); treat
    # them as part of a word so "कोतवाल" does not match inside "कोतवाली".
    return c.isalnum() or unicodedata.category(c).startswith("M")


@dataclass
class Match:
    start: int
    end: int          # exclusive
    key: str
    payloads: list[Any]


class AhoCorasick:
    def __init__(self) -> None:
        self._goto: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._out: list[list[str]] = [[]]
        self._payloads: dict[str, list[Any]] = {}
        self._built = False

    def add(self, key: str, payload: Any) -> None:
        key = lower_same_length(key.strip())
        if not key:
            return
        self._payloads.setdefault(key, [])
        if payload not in self._payloads[key]:
            self._payloads[key].append(payload)
        node = 0
        for ch in key:
            nxt = self._goto[node].get(ch)
            if nxt is None:
                nxt = len(self._goto)
                self._goto[node][ch] = nxt
                self._goto.append({})
                self._fail.append(0)
                self._out.append([])
            node = nxt
        if key not in self._out[node]:
            self._out[node].append(key)
        self._built = False

    def build(self) -> None:
        """Breadth-first construction of failure links."""
        queue = deque()
        for node in self._goto[0].values():
            self._fail[node] = 0
            queue.append(node)
        while queue:
            r = queue.popleft()
            for ch, s in self._goto[r].items():
                queue.append(s)
                f = self._fail[r]
                while f and ch not in self._goto[f]:
                    f = self._fail[f]
                self._fail[s] = self._goto[f].get(ch, 0) if self._goto[f].get(ch, 0) != s else 0
                self._out[s] = self._out[s] + [k for k in self._out[self._fail[s]] if k not in self._out[s]]
        self._built = True

    def find_all(self, text: str, whole_words: bool = True) -> list[Match]:
        if not self._built:
            self.build()
        low = lower_same_length(text)
        node, found = 0, []
        for i, ch in enumerate(low):
            while node and ch not in self._goto[node]:
                node = self._fail[node]
            node = self._goto[node].get(ch, 0)
            for key in self._out[node]:
                start, end = i - len(key) + 1, i + 1
                if whole_words:
                    if start > 0 and is_word_char(text[start - 1]) and is_word_char(key[0]):
                        continue
                    if end < len(text) and is_word_char(text[end]) and is_word_char(key[-1]):
                        continue
                found.append(Match(start, end, key, self._payloads[key]))
        return found

    def find_longest(self, text: str) -> list[Match]:
        """Non-overlapping matches, preferring the longest (then earliest)."""
        return longest_non_overlapping(self.find_all(text))


def longest_non_overlapping(matches: list[Match]) -> list[Match]:
    """Keep the longest match wherever matches overlap ("Sadar Bazar" over "Sadar")."""
    taken: list[Match] = []
    for m in sorted(matches, key=lambda m: (-(m.end - m.start), m.start)):
        if all(m.end <= t.start or m.start >= t.end for t in taken):
            taken.append(m)
    return sorted(taken, key=lambda m: m.start)
