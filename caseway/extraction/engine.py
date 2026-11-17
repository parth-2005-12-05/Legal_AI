"""Extraction engine: runs every extraction layer over one Document.

Order (cheap and near-certain first):
  1. document-type detection (decides how much header labels are trusted)
  2. Layer 1 label-value anchoring, each value resolved by exact alias lookup,
     then an Aho-Corasick scan inside the value, then the fuzzy matcher
  3. Layer 2 regex grammars: CNR, case number, FIR number, sections, land ids
  4. dates with roles; IPC/BNS act filled from the occurrence date
  5. Layer 3 Aho-Corasick scan of the whole text for any remaining mentions
  6. evidence links: pincode -> district, CNR prefix -> district + court

Output is a flat list of Candidates; nothing is decided here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..gazetteer.ahocorasick import AhoCorasick, Match, longest_non_overlapping
from ..gazetteer.fuzzy import FuzzyMatcher
from ..gazetteer.model import Gazetteer
from ..models import Candidate, Document
from . import grammars
from .dates import find_dates
from .doctype import detect_doc_type
from .labels import anchor_values
from .statutes import StatuteTable

GAZ_FIELDS = {"state", "district", "tehsil", "village", "police_station", "court"}
MIN_FUZZY_SCORE = 0.6


@dataclass
class Extraction:
    doc_id: str
    doc_type: str
    doc_type_score: float
    candidates: list[Candidate] = field(default_factory=list)


class Extractor:
    def __init__(self, gazetteer: Gazetteer, fuzzy: FuzzyMatcher, statutes: StatuteTable):
        self.gaz, self.fuzzy, self.statutes = gazetteer, fuzzy, statutes
        self.ac = AhoCorasick()
        for alias in gazetteer.aliases:
            # very short Latin aliases ("UP", "PS") are too ambiguous for a free-text
            # scan; they are still used by exact lookup after a label.
            if alias.script == "deva" or len(alias.folded) > 3:
                self.ac.add(alias.folded, alias.entity_id)
        self.ac.build()

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _source(doc: Document, kind: str, in_header: bool = False) -> str:
        if doc.source == "chat":
            return "chat"
        if kind == "label":
            return "doc_header_label" if in_header and doc.doc_type in ("fir", "court_order") else "doc_label"
        if kind == "regex":
            return "born_digital" if doc.source == "born_digital" else "ocr_high_conf"
        return kind

    def _cand(self, doc: Document, field_: str, value: str, text: str, span, rule: str, score: float,
              source: str, **extra) -> Candidate:
        return Candidate(field=field_, value=value, text=text, doc_id=doc.doc_id, page=doc.page,
                         span=(int(span[0]), int(span[1])), rule=rule, score=round(score, 4),
                         source=source, ocr_confidence=doc.ocr_confidence, extra=extra)

    def _resolve_value(self, doc: Document, field_: str, value: str, span, source: str) -> list[Candidate]:
        """Resolve a label-anchored value to gazetteer entities of the right type."""
        ids = self.gaz.lookup_exact(value, field_)
        if ids:
            return [self._cand(doc, field_, i, value, span, f"label:{field_}:exact", 1.0, source,
                               group=f"{doc.doc_id}:{span[0]}") for i in ids]
        out = []
        for m in self.ac.find_longest(value):
            ids = [i for i in m.payloads if self.gaz.entities[i].type == field_]
            for i in ids:
                s = (span[0] + m.start, span[0] + m.end)
                out.append(self._cand(doc, field_, i, value[m.start:m.end], s, f"label:{field_}:scan", 1.0,
                                      source, group=f"{doc.doc_id}:{s[0]}"))
        if out:
            return out
        res = self.fuzzy.match(value, field_)
        for c in ([res.accepted] if res.accepted else res.candidates):
            if c.score >= MIN_FUZZY_SCORE:
                out.append(self._cand(doc, field_, c.entity_id, value, span,
                                      f"label:{field_}:fuzzy" + ("" if res.accepted else ":unsure"),
                                      c.score, source, fuzzy=c.details, group=f"{doc.doc_id}:{span[0]}"))
        return out

    # -- main -----------------------------------------------------------
    def extract(self, doc: Document) -> Extraction:
        text = doc.text
        if doc.source == "chat":
            doc.doc_type, dt_score = "chat", 1.0
        else:
            doc.doc_type, dt_score = detect_doc_type(text)
        out = Extraction(doc.doc_id, doc.doc_type, dt_score)
        cands = out.candidates
        covered: list[tuple[str, int, int]] = []   # (field, start, end) already explained by a label

        # Layer 1: label-value anchoring
        for av in anchor_values(text):
            src = self._source(doc, "label", av.in_header)
            if av.field in GAZ_FIELDS:
                got = self._resolve_value(doc, av.field, av.value, av.span, src)
                cands += got
                if got:
                    covered.append((av.field, *av.span))
            elif av.field == "opposing_party":
                name = re.split(r"\s+(?:s/o|d/o|w/o|son of|daughter of|wife of|पुत्र|पत्नी)\b", av.value, 1,
                                flags=re.IGNORECASE)[0].strip()
                if 2 <= len(name) <= 60:
                    cands.append(self._cand(doc, "opposing_party", name, name, av.span, "label:party", 0.9, src))
            elif av.field == "pincode":
                pin = re.match(r"\d{6}", av.value)
                if pin and pin.group(0) in self.gaz.pincode_to_district:
                    d = self.gaz.pincode_to_district[pin.group(0)]
                    cands.append(self._cand(doc, "district", d, pin.group(0), av.span, "pincode->district",
                                            1.0, "inferred"))
                    covered.append(("pincode", *av.span))

        if doc.source == "chat":
            for m in re.finditer(r"\bagainst\s+(?:my\s+\w+\s+|mr\.?\s+|mrs\.?\s+|one\s+)?"
                                 r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", text):
                cands.append(self._cand(doc, "opposing_party", m["name"], m["name"], m.span("name"),
                                        "chat:against", 0.7, "chat"))

        # Layer 2: identifiers
        rsrc = self._source(doc, "regex")
        for h in grammars.find_cnr(text):
            cands.append(self._cand(doc, "cnr", h.value, h.value, h.span, "regex:cnr", 1.0, rsrc))
            d = self.gaz.cnr_to_district.get(h.parts["district_code"])
            if d:
                cands.append(self._cand(doc, "district", d, h.value, h.span, "cnr->district", 1.0, "cnr_lookup"))
            court = self.gaz.establishment_to_court.get(h.parts["establishment"])
            if court:
                cands.append(self._cand(doc, "court", court, h.value, h.span, "cnr->court", 1.0, "cnr_lookup"))
        for h in grammars.find_case_numbers(text):
            cands.append(self._cand(doc, "case_number", h.value, text[h.span[0]:h.span[1]], h.span,
                                    "regex:case_no", 1.0, rsrc, prefix=h.parts["prefix"]))
        for h in grammars.find_fir_numbers(text):
            cands.append(self._cand(doc, "fir_number", h.value, text[h.span[0]:h.span[1]], h.span,
                                    "regex:fir_no", 1.0, rsrc))
        for h in grammars.find_land_ids(text):
            cands.append(self._cand(doc, "land_id", h.value, text[h.span[0]:h.span[1]], h.span,
                                    "regex:land_id", 1.0, rsrc))

        # dates (needed before sections: the occurrence date picks IPC vs BNS)
        dates = find_dates(text)
        for d in dates:
            cands.append(self._cand(doc, f"date_{d.role}", d.date.isoformat(), d.text, d.span,
                                    f"date:{d.role}", d.score, rsrc))
        occurrence = next((d.date for d in dates if d.role == "occurrence"), None)
        for h in grammars.find_sections(text):
            act = self.statutes.resolve_act(h.parts["act"], occurrence)
            key = f"{act} {h.parts['section']}"
            cands.append(self._cand(doc, "section", key, text[h.span[0]:h.span[1]], h.span, "regex:sections",
                                    1.0 if h.parts["act"] else 0.8, rsrc, act_inferred=h.parts["act"] is None))

        # pincodes outside a label: weaker evidence
        for h in grammars.find_pincodes(text):
            if h.value in self.gaz.pincode_to_district and not any(
                    f == "pincode" and s <= h.span[0] < e for f, s, e in covered):
                cands.append(self._cand(doc, "district", self.gaz.pincode_to_district[h.value], h.value, h.span,
                                        "pincode->district", 0.7, "inferred"))

        # Layer 3: gazetteer scan over the whole text. Longest-match is applied per
        # entity type, so "Family Court Ghaziabad" yields the court AND the district.
        scan_src = "chat" if doc.source == "chat" else "gazetteer_scan"
        by_type: dict[str, list] = {}
        for m in self.ac.find_all(text):
            for eid in m.payloads:
                ftype = self.gaz.entities[eid].type
                by_type.setdefault(ftype, []).append(Match(m.start, m.end, m.key, [eid]))
        for ftype, matches in by_type.items():
            for m in longest_non_overlapping(matches):
                if any(f == ftype and s <= m.start < e for f, s, e in covered):
                    continue
                cands.append(self._cand(doc, ftype, m.payloads[0], text[m.start:m.end], (m.start, m.end),
                                        "scan:aho_corasick", 1.0, scan_src, group=f"{doc.doc_id}:{m.start}"))
        return out
