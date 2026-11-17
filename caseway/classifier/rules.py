"""Tier 1: deterministic case-type rules (highest precision).

    statute section -> sub-type   IPC 420 / BNS 318(4) -> cheating; NI 138 -> cheque bounce
    document type   -> top level   an FIR implies criminal; a Khatauni implies property
    case-no prefix  -> category    "Crl.A" -> criminal; "O.S." -> civil; "HMA" -> family
    keyword rules                  with NegEx-style negation ("not a criminal matter",
                                   "मामला नहीं") using a small token window
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..extraction.statutes import StatuteTable
from ..normalise import romanise

NEGATION_CUES = {"not", "no", "never", "without", "nahi", "nahin", "na", "नहीं", "isnt", "isn't", "wasnt"}
NEGATION_WINDOW = 4  # tokens before the keyword


@dataclass
class RuleVote:
    top: str
    sub: str | None
    weight: float
    evidence: str
    kind: str   # section | doc_type | prefix | keyword


class RuleClassifier:
    def __init__(self, taxonomy_path: str | Path, statutes: StatuteTable):
        with open(taxonomy_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.types: dict[str, dict[str, list[str]]] = raw["types"]
        self.prefixes = {k: tuple(v) for k, v in raw.get("case_number_prefix", {}).items()}
        self.doc_type_to_top = raw.get("doc_type_to_top", {})
        self.statutes = statutes
        self._keyword_rx = []
        for top, subs in self.types.items():
            for sub, words in subs.items():
                for w in words:
                    rx = re.compile(r"(?<![\wऀ-ॿ])" + re.escape(romanise(w).lower()) + r"(?![\wऀ-ॿ])")
                    self._keyword_rx.append((top, sub, w, rx))

    def all_subtypes(self) -> list[tuple[str, str]]:
        return [(t, s) for t, subs in self.types.items() for s in subs]

    @staticmethod
    def _negated(text: str, start: int) -> bool:
        before = re.findall(r"[\w']+", text[:start])[-NEGATION_WINDOW:]
        return any(tok in NEGATION_CUES for tok in before)

    def votes(self, sections: list[str], doc_types: list[str], case_prefixes: list[str], text: str) -> list[RuleVote]:
        out: list[RuleVote] = []
        for key in dict.fromkeys(sections):
            t = self.statutes.case_type(key)
            if t:
                out.append(RuleVote(t[0], t[1], 1.0, f"section {key}", "section"))
        for dt in dict.fromkeys(doc_types):
            if dt in self.doc_type_to_top:
                out.append(RuleVote(self.doc_type_to_top[dt], None, 0.6, f"document type {dt}", "doc_type"))
        for p in dict.fromkeys(case_prefixes):
            if p in self.prefixes:
                top, sub = self.prefixes[p]
                out.append(RuleVote(top, sub, 0.8, f"case number prefix {p}", "prefix"))
        low = romanise(text).lower()
        for top, sub, word, rx in self._keyword_rx:
            m = rx.search(low)
            if m and not self._negated(low, m.start()):
                out.append(RuleVote(top, sub, 0.5, f"keyword '{word}'", "keyword"))
        return out
