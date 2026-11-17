"""Evidence fusion, consistency checks and confidence routing (design:
"Accuracy strategy" 1, 2 and 4).

1. Evidence fusion per field. Every source votes; a vote's weight is
   score x source weight x OCR confidence. Agreeing votes combine by noisy-OR;
   the strongest disagreeing value is subtracted as a penalty. Two kinds of
   "disagreement" are NOT penalised because they are not real conflicts:
     - homonyms from the same text span ("Kotwali" maps to three stations;
       the hierarchy already picked one), and
     - a bare gazetteer mention elsewhere in the text (an address) when the
       chosen value is backed by a labelled field or a CNR lookup.
2. Thresholds: >= 0.95 auto-accept; 0.75-0.95 provisional (one-click confirm);
   < 0.75 or two sources in conflict -> mandatory human review.
4. Consistency checks: police station inside the district, court hears this
   category for that area, dates in a sensible order, FIR year matches the
   registration date, case type agrees with the sections. A violation lowers
   confidence and is shown to the reviewer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .config import FusionSettings
from .gazetteer.model import Gazetteer
from .gazetteer.resolver import Resolution, noisy_or
from .models import Candidate, CaseTypeResult, FieldResult

JURISDICTION_FIELDS = ["state", "district", "tehsil", "village", "police_station", "court"]
SIMPLE_FIELDS = ["fir_number", "case_number", "cnr", "opposing_party", "land_id",
                 "date_occurrence", "date_registration", "date_order", "date_next_hearing", "date_invoice"]
STRONG_SOURCES = {"doc_header_label", "doc_label", "cnr_lookup", "born_digital"}
# Fields that must be trusted before matching can run without a person.
BLOCKING_FIELDS = ["state", "district", "case_type"]
CLOSE_MARGIN = 0.2
IMPLIED_DISCOUNT = 0.8   # a child place counts as slightly weaker evidence for its district


@dataclass
class FusionResult:
    fields: dict[str, FieldResult]
    sections: list[str]
    issues: list[str] = field(default_factory=list)
    needs_review: list[str] = field(default_factory=list)

    @property
    def auto(self) -> bool:
        return not self.needs_review


class Fusion:
    def __init__(self, gazetteer: Gazetteer, settings: FusionSettings):
        self.gaz, self.s = gazetteer, settings

    def weight(self, c: Candidate) -> float:
        return c.score * self.s.source_weights.get(c.source, 0.5) * c.ocr_confidence

    def status(self, conf: float, conflict: bool) -> str:
        if conflict or conf < self.s.provisional:
            return "review"
        return "auto_accept" if conf >= self.s.auto_accept else "provisional"

    def _fuse_field(self, name: str, cands: list[Candidate], chosen: str | None) -> FieldResult:
        if not cands and chosen is None:
            return FieldResult(name, None, None, 0.0, "missing")
        by_value: dict[str, list[Candidate]] = defaultdict(list)
        for c in cands:
            by_value[c.value].append(c)
        if chosen is None:
            chosen = max(by_value, key=lambda v: noisy_or(self.weight(c) for c in by_value[v]))
        agree = by_value.get(chosen, [])
        conf = noisy_or(self.weight(c) for c in agree)
        groups = {c.extra.get("group") for c in agree if c.extra.get("group")}
        strong = any(c.source in STRONG_SOURCES for c in agree)
        competitors = []
        for v, cs in by_value.items():
            if v == chosen:
                continue
            real = [c for c in cs if c.extra.get("group") not in groups or not c.extra.get("group")]
            if strong:
                real = [c for c in real if c.source != "gazetteer_scan"]
            if real:
                competitors.append((v, noisy_or(self.weight(c) for c in real)))
        competitors.sort(key=lambda x: -x[1])
        issues, conflict = [], False
        if competitors:
            top_v, top_w = competitors[0]
            conf -= self.s.disagreement_penalty * top_w
            if top_w >= 0.5 and top_w >= 0.6 * noisy_or(self.weight(c) for c in agree):
                conflict = True
                issues.append(f"sources disagree: '{self._display(name, top_v)}' also supported")
        conf = round(max(0.0, min(1.0, conf)), 4)
        return FieldResult(name, chosen, self._display(name, chosen), conf, self.status(conf, conflict),
                           evidence=agree, alternatives=[(v, round(w, 3)) for v, w in competitors[:3]],
                           issues=issues)

    def _display(self, name: str, value: str | None) -> str | None:
        if value and name in JURISDICTION_FIELDS:
            return self.gaz.name(value)
        return value

    def fuse(self, candidates: list[Candidate], resolution: Resolution, case_type: CaseTypeResult) -> FusionResult:
        by_field: dict[str, list[Candidate]] = defaultdict(list)
        for c in candidates:
            by_field[c.field].append(c)
        fields: dict[str, FieldResult] = {}
        for name in JURISDICTION_FIELDS:
            fields[name] = self._fuse_field(name, by_field.get(name, []), resolution.chosen.get(name))
        for name in SIMPLE_FIELDS:
            if by_field.get(name):
                fields[name] = self._fuse_field(name, by_field[name], None)
        sections = sorted({c.value for c in by_field.get("section", [])})
        issues = list(resolution.issues)

        # A police station, village, tehsil or court inside the chosen district is
        # an independent, agreeing source for the district ("a court name or a
        # revenue village all imply a district"), counted as inferred evidence.
        d = fields["district"]
        in_conflict = any(i.startswith("sources disagree") for i in d.issues)
        if d.value and not in_conflict:
            implied = [IMPLIED_DISCOUNT * fields[l].confidence
                       for l in ("tehsil", "village", "police_station", "court")
                       if fields[l].value and fields[l].evidence and self.gaz.is_within(fields[l].value, d.value)]
            if implied:
                d.confidence = round(noisy_or([d.confidence, *implied]), 4)
                d.status = self.status(d.confidence, False)
                d.issues.append("supported by places inside the district")

        # resolver-level checks
        if fields["district"].value and resolution.margin < CLOSE_MARGIN and len(resolution.district_totals) > 1:
            fr = fields["district"]
            fr.confidence = min(fr.confidence, self.s.provisional - 0.01)
            fr.status = "review"
            fr.issues.append(f"close call between districts (margin {resolution.margin:.2f})")
        for issue in resolution.issues:
            for name in JURISDICTION_FIELDS:
                if issue.startswith(name.replace("_", " ")):
                    fr = fields[name]
                    fr.confidence = round(max(0.0, fr.confidence - 0.2), 4)
                    fr.status = self.status(fr.confidence, False) if fr.value else fr.status
                    fr.issues.append(issue)
        if not by_field.get("state") and fields["district"].value:
            # no direct state evidence: the state is implied by the district's parent chain
            st = self.gaz.ancestor_of_type(fields["district"].value, "state")
            fields["state"] = FieldResult("state", st, self.gaz.name(st), fields["district"].confidence,
                                          fields["district"].status, issues=["implied by district"])

        # derive a court from the territorial table when no document names one
        if fields["court"].value is None and fields["district"].value and case_type.primary_top:
            options = self.gaz.courts_for(fields["district"].value, case_type.primary_top, case_type.primary_sub)
            if options:
                w = self.s.source_weights.get("inferred", 0.5)
                fields["court"] = FieldResult("court", options[0], self.gaz.name(options[0]), w, "derived",
                                              alternatives=[(o, w) for o in options[1:3]],
                                              issues=["derived from territorial rules (no document names the court)"])

        # court must hear this category of case
        court = self.gaz.get(fields["court"].value)
        if court and case_type.primary_top and case_type.primary_top not in court.attrs.get("categories", []):
            fields["court"].issues.append(f"{court.name} does not normally hear {case_type.primary_top} matters")
            fields["court"].confidence = round(max(0.0, fields["court"].confidence - 0.2), 4)
            if fields["court"].status != "derived":
                fields["court"].status = self.status(fields["court"].confidence, False)

        # date order and FIR year
        occ, reg = fields.get("date_occurrence"), fields.get("date_registration")
        if occ and reg and occ.value and reg.value and reg.value < occ.value:
            for fr in (occ, reg):
                fr.issues.append("registration date is before occurrence date")
                fr.status = "review"
        fir = fields.get("fir_number")
        if fir and fir.value and reg and reg.value and fir.value.split("/")[-1] != reg.value[:4]:
            fir.issues.append("FIR year does not match registration date")
            fir.status = "review"

        # case type as a field
        ct_conf = case_type.probability
        ct_issues = []
        top_from_sections = {t[0] for t in (self._section_type(s) for s in sections) if t}
        if top_from_sections and case_type.primary_top not in top_from_sections:
            ct_issues.append("case type disagrees with the cited sections")
            ct_conf = min(ct_conf, self.s.provisional - 0.01)
        ct_value = f"{case_type.primary_top}/{case_type.primary_sub or ''}" if case_type.primary_top else None
        fields["case_type"] = FieldResult("case_type", ct_value, ct_value, round(ct_conf, 4),
                                          self.status(ct_conf, False) if ct_value else "missing",
                                          issues=ct_issues)

        needs = [n for n in BLOCKING_FIELDS if fields[n].status in ("review", "missing")]
        # an extracted (not derived) court or police station that is in conflict also needs a person
        # (the police station only matters for criminal matters)
        extra_fields = ["court"] + (["police_station"] if case_type.primary_top == "criminal" else [])
        # A court / police station named only in the chat does not block: the chat
        # follow-up question confirms it, and matching falls back to the district.
        needs += [n for n in extra_fields if fields[n].status == "review" and fields[n].value is not None
                  and any(c.source != "chat" for c in fields[n].evidence)]
        for fr in fields.values():
            issues += [f"{fr.field}: {i}" for i in fr.issues if f"{fr.field}: {i}" not in issues]
        return FusionResult(fields, sections, issues, needs)

    _statutes = None

    def _section_type(self, key: str):
        return self._statutes.case_type(key) if self._statutes else None

    def attach_statutes(self, statutes) -> "Fusion":
        self._statutes = statutes
        return self
