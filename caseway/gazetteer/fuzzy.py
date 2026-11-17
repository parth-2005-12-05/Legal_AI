"""Fuzzy matching for OCR noise and spelling variants (design: Layer 5).

What exact lookup and Aho-Corasick miss goes through two stages:

1. Candidate generation (fast, high recall), union of:
   - character trigram inverted index (keep entities sharing >= 40% of trigrams)
   - BK-tree keyed on Damerau-Levenshtein distance for short names (<= 15 chars)
   - Indic phonetic key index ("Ghaziabad" / "Gaziabad" / "Ghaziyabad")
2. Ranking (precise), a weighted blend of
   - Jaro-Winkler similarity (rewards matching prefixes, suits place names)
   - weighted edit distance with an OCR confusion matrix (rn<->m, l<->1, ...)
   - phonetic-key agreement
   - context bonus when the candidate lies inside an already-confirmed parent

Accept the top candidate only if score >= 0.92 AND it beats the runner-up by
>= 0.05; otherwise return the top 3 for human review. Duplicate names are
expected (many districts have a "Kotwali"); the resolver settles those.

All algorithms are implemented here in plain Python (no network, no models).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .model import Gazetteer
from ..normalise import fold, phonetic_key, romanise

# --- string metrics ----------------------------------------------------------


def jaro_winkler(a: str, b: str, prefix_scale: float = 0.1) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    window = max(len(a), len(b)) // 2 - 1
    a_flags, b_flags = [False] * len(a), [False] * len(b)
    matches = 0
    for i, ch in enumerate(a):
        lo, hi = max(0, i - window), min(len(b), i + window + 1)
        for j in range(lo, hi):
            if not b_flags[j] and b[j] == ch:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    a_m = [c for c, f in zip(a, a_flags) if f]
    b_m = [c for c, f in zip(b, b_flags) if f]
    transpositions = sum(x != y for x, y in zip(a_m, b_m)) / 2
    jaro = (matches / len(a) + matches / len(b) + (matches - transpositions) / matches) / 3
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * prefix_scale * (1 - jaro)


def damerau_levenshtein(a: str, b: str) -> int:
    """Optimal string alignment distance (insert, delete, substitute, transpose)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


# Substitution costs for common OCR / transliteration confusions. In production
# this matrix is LEARNED from reviewer corrections (design: "Learn from corrections").
OCR_CONFUSION: dict[frozenset, float] = {
    frozenset("l1"): 0.2, frozenset("i1"): 0.2, frozenset("il"): 0.3, frozenset("o0"): 0.2,
    frozenset("s5"): 0.3, frozenset("b8"): 0.3, frozenset("ec"): 0.4, frozenset("uv"): 0.4,
    frozenset("vw"): 0.3, frozenset("ao"): 0.6, frozenset("ae"): 0.6, frozenset("ie"): 0.5,
    frozenset("zj"): 0.4, frozenset("ड़र"): 0.3, frozenset("ठढ"): 0.3,
}
_RN_M_COST = 0.2  # the classic two-for-one OCR confusion "rn" <-> "m"


def weighted_edit_distance(a: str, b: str) -> float:
    n, m = len(a), len(b)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = float(i)
    for j in range(1, m + 1):
        d[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                sub = 0.0
            else:
                sub = OCR_CONFUSION.get(frozenset((a[i - 1], b[j - 1])), 1.0)
            best = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
            if i > 1 and a[i - 2:i] == "rn" and b[j - 1] == "m":
                best = min(best, d[i - 2][j - 1] + _RN_M_COST)
            if j > 1 and b[j - 2:j] == "rn" and a[i - 1] == "m":
                best = min(best, d[i - 1][j - 2] + _RN_M_COST)
            d[i][j] = best
    return d[n][m]


def trigrams(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


# --- BK-tree -----------------------------------------------------------------


class BKTree:
    """Metric tree over Damerau-Levenshtein: finds all keys within distance k
    while visiting only a small fraction of the tree."""

    def __init__(self) -> None:
        self.root: tuple[str, dict] | None = None

    def add(self, key: str) -> None:
        if self.root is None:
            self.root = (key, {})
            return
        node = self.root
        while True:
            dist = damerau_levenshtein(key, node[0])
            if dist == 0:
                return
            child = node[1].get(dist)
            if child is None:
                node[1][dist] = (key, {})
                return
            node = child

    def search(self, key: str, k: int) -> list[tuple[str, int]]:
        if self.root is None:
            return []
        out, stack = [], [self.root]
        while stack:
            word, children = stack.pop()
            dist = damerau_levenshtein(key, word)
            if dist <= k:
                out.append((word, dist))
            for d, child in children.items():
                if dist - k <= d <= dist + k:
                    stack.append(child)
        return out


# --- matcher -----------------------------------------------------------------


@dataclass
class FuzzyCandidate:
    entity_id: str
    alias: str
    score: float
    details: dict


@dataclass
class FuzzyResult:
    accepted: FuzzyCandidate | None
    candidates: list[FuzzyCandidate]   # top 3, best first


class FuzzyMatcher:
    def __init__(self, gazetteer: Gazetteer, accept_score: float = 0.92, accept_gap: float = 0.05,
                 trigram_min_overlap: float = 0.4, bk_max_distance: int = 2) -> None:
        self.gaz = gazetteer
        self.accept_score, self.accept_gap = accept_score, accept_gap
        self.min_overlap, self.k = trigram_min_overlap, bk_max_distance
        self._key_to_ids: dict[str, set[str]] = defaultdict(set)
        self._trigram_index: dict[str, set[str]] = defaultdict(set)
        self._phonetic_index: dict[str, set[str]] = defaultdict(set)
        self._bk = BKTree()
        for alias in gazetteer.aliases:
            for key in {alias.folded, alias.romanised}:
                self._key_to_ids[key].add(alias.entity_id)
                for tg in trigrams(key):
                    self._trigram_index[tg].add(key)
                if len(key) <= 15:
                    self._bk.add(key)
            self._phonetic_index[alias.phonetic.replace(" ", "")].add(alias.folded)

    def _generate(self, query: str) -> set[str]:
        keys: set[str] = set()
        q_tri = trigrams(query)
        counts: dict[str, int] = defaultdict(int)
        for tg in q_tri:
            for key in self._trigram_index.get(tg, ()):
                counts[key] += 1
        keys |= {k for k, c in counts.items() if c / max(len(q_tri), 1) >= self.min_overlap}
        if len(query) <= 15:
            keys |= {k for k, _ in self._bk.search(query, self.k)}
        keys |= self._phonetic_index.get(phonetic_key(query).replace(" ", ""), set())
        return keys

    def match(self, text: str, entity_type: str | None = None, within: str | None = None) -> FuzzyResult:
        """Rank gazetteer entities for a noisy name. `within` is an already
        confirmed ancestor (e.g. the district) that earns a context bonus."""
        query = fold(text)
        q_roman = fold(romanise(query))
        q_phon = phonetic_key(query).replace(" ", "")
        best: dict[str, FuzzyCandidate] = {}
        for key in self._generate(query) | (self._generate(q_roman) if q_roman != query else set()):
            for eid in self._key_to_ids.get(key, ()):
                ent = self.gaz.entities[eid]
                if entity_type and ent.type != entity_type:
                    continue
                q = q_roman if key.isascii() else query
                jw = jaro_winkler(q, key)
                wed = weighted_edit_distance(q, key)
                edit_sim = max(0.0, 1 - wed / max(len(q), len(key), 1))
                phon = 1.0 if phonetic_key(key).replace(" ", "") == q_phon else 0.0
                ctx = 1.0 if within and self.gaz.is_within(eid, within) else 0.0
                score = min(1.0, 0.45 * jw + 0.35 * edit_sim + 0.10 * phon + 0.10 * ctx)
                if q == key:
                    score = max(score, 0.90 + 0.10 * ctx)
                cand = FuzzyCandidate(eid, key, round(score, 4),
                                      {"jaro_winkler": round(jw, 3), "edit_sim": round(edit_sim, 3),
                                       "phonetic": phon, "context": ctx})
                if eid not in best or cand.score > best[eid].score:
                    best[eid] = cand
        ranked = sorted(best.values(), key=lambda c: -c.score)[:3]
        accepted = None
        if ranked:
            gap = ranked[0].score - (ranked[1].score if len(ranked) > 1 else 0.0)
            if ranked[0].score >= self.accept_score and gap >= self.accept_gap:
                accepted = ranked[0]
        return FuzzyResult(accepted, ranked)
