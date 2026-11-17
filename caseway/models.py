"""Shared data structures passed between pipeline stages.

The design's core rule is "every extracted field and every lawyer match carries
the evidence behind it (the exact text span, page, rule and score)". These
dataclasses are how that evidence travels: extraction emits Candidates, fusion
turns them into FieldResults, and matching emits LawyerMatches with a score
breakdown and human-readable reasons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """One piece of input text: a page of a document, or the customer's chat."""
    doc_id: str
    text: str                    # normalised text; all spans refer to this string
    page: int = 1
    source: str = "born_digital"  # born_digital | ocr_high_conf | chat ...
    doc_type: str = "unknown"     # fir | court_order | revenue_record | receipt | notice | affidavit | chat
    ocr_confidence: float = 1.0   # mean word confidence for OCR pages, 1.0 otherwise


@dataclass
class Candidate:
    """A possible value for one field, with the evidence that produced it."""
    field: str                 # e.g. "district", "police_station", "fir_number"
    value: str                 # entity id for gazetteer fields, normalised string otherwise
    text: str                  # the exact matched text
    doc_id: str
    page: int
    span: tuple[int, int]      # [start, end) offsets into Document.text
    rule: str                  # id of the rule/grammar/matcher that fired
    score: float               # match score in 0..1 (1.0 for exact matches)
    source: str                # evidence source; looked up in fusion.source_weights
    ocr_confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FieldResult:
    """The fused decision for one field."""
    field: str
    value: str | None
    display: str | None                 # human-readable value (entity name)
    confidence: float
    status: str                         # auto_accept | provisional | review | missing
    evidence: list[Candidate] = field(default_factory=list)
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [c.to_dict() if isinstance(c, Candidate) else c for c in self.evidence]
        return d


@dataclass
class CaseTypeResult:
    primary_top: str | None
    primary_sub: str | None
    probability: float
    labels: list[tuple[str, str, float]]   # (top, sub, prob) above the multi-label threshold
    evidence: list[str]                    # which rules / similar cases decided it
    tier: str                              # "rules" | "statistical" | "ensemble" | "none"


@dataclass
class LawyerMatch:
    lawyer_id: str
    name: str
    score: float
    breakdown: dict[str, float]
    reasons: list[str]
    public_profile: dict[str, Any]   # neutral facts only (Bar Council advertising rules)


@dataclass
class MatchDecision:
    """Output of the matching engine for one case, including where it goes next."""
    case_id: str
    matches: list[LawyerMatch]
    route: str                 # customer | partner_queue | senior_review | no_match
    route_reasons: list[str]
    reviewers: list[str]       # roles that must act when route == senior_review
    flags: list[str] = field(default_factory=list)  # conflict | urgent | low_score | close_call
    rejected: dict[str, list[str]] = field(default_factory=dict)  # lawyer id -> why not eligible
    conflicts: list[dict] = field(default_factory=list)           # lawyer ids + similarity (no names)
