"""Step 1: eligibility filter (hard constraints).

A lawyer is dropped if any of these fail:
  * Court access: enrolled with the right State Bar Council and practises in the
    resolved court, or at least in that district at the same tier.
  * Conflict of interest: the opposing party's name fuzzy-matches a past or
    current client (Jaro-Winkler + phonetic key, threshold 0.90). A hit both
    drops that lawyer AND flags the case for a named senior lawyer / admin,
    whether or not partner review is switched on.
  * Availability: not on leave, no clash with the next hearing date, load below cap.
  * Language: speaks the customer's language or at least reads the documents'.
  * Urgency: bail / anticipatory-bail matters only go to lawyers flagged for
    urgent criminal work who are available within 24 hours.
Every rejection is recorded with its reason so reviewers can see why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..gazetteer.fuzzy import jaro_winkler
from ..gazetteer.model import Gazetteer
from ..normalise import fold, phonetic_key
from .profiles import CaseFacts, Lawyer


def name_similarity(a: str, b: str) -> float:
    fa, fb = fold(a), fold(b)
    jw = jaro_winkler(fa, fb)
    if phonetic_key(fa).replace(" ", "") == phonetic_key(fb).replace(" ", "") and jw > 0.8:
        jw = max(jw, 0.95)
    return jw


@dataclass
class EligibilityResult:
    eligible: list[Lawyer] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)  # lawyer id -> reasons
    conflicts: list[dict] = field(default_factory=list)            # ids + similarity only, no names in logs


class EligibilityFilter:
    def __init__(self, gazetteer: Gazetteer, conflict_threshold: float = 0.90,
                 urgent_subtypes: list[str] | None = None, urgent_max_hours: int = 24):
        self.gaz = gazetteer
        self.conflict_threshold = conflict_threshold
        self.urgent_subtypes = set(urgent_subtypes or ["bail", "anticipatory_bail"])
        self.urgent_max_hours = urgent_max_hours

    def is_urgent(self, case: CaseFacts) -> bool:
        return case.sub in self.urgent_subtypes

    def court_access(self, lawyer: Lawyer, case: CaseFacts) -> str | None:
        """Return a rejection reason, or None if the lawyer may appear."""
        if case.state and lawyer.bar_council != case.state:
            return f"not enrolled with the {case.state} Bar Council"
        court = self.gaz.get(case.court)
        if court:
            if case.court in lawyer.courts:
                return None
            if court.attrs.get("tier") == "high_court":
                return f"does not practise in {court.name}"
            same_district = [c for c in lawyer.courts if self.gaz.ancestor_of_type(c, "district") == case.district]
            return None if same_district else f"does not practise in {self.gaz.name(case.district)}"
        if case.district:
            if any(self.gaz.ancestor_of_type(c, "district") == case.district for c in lawyer.courts):
                return None
            return f"does not practise in {self.gaz.name(case.district)}"
        return None

    def check(self, lawyers: list[Lawyer], case: CaseFacts) -> EligibilityResult:
        res = EligibilityResult()
        urgent = self.is_urgent(case)
        for l in lawyers:
            reasons = []
            r = self.court_access(l, case)
            if r:
                reasons.append(r)
            for party in case.opposing_parties:
                for client in l.clients:
                    sim = name_similarity(party, client)
                    if sim >= self.conflict_threshold:
                        reasons.append("conflict of interest with an existing or past client")
                        res.conflicts.append({"lawyer_id": l.id, "similarity": round(sim, 3)})
            if l.on_leave:
                reasons.append("on leave")
            if case.next_hearing and case.next_hearing in l.busy_dates:
                reasons.append("busy on the next hearing date")
            if l.active_matters >= l.capacity:
                reasons.append("at full capacity")
            langs = set(l.languages.get("fluent", [])) | set(l.languages.get("read", []))
            if case.customer_language not in langs and case.document_language not in langs:
                reasons.append("no shared language")
            if urgent and not (l.urgent_criminal and l.available_within_hours <= self.urgent_max_hours):
                reasons.append(f"not available for urgent criminal work within {self.urgent_max_hours}h")
            if reasons:
                res.rejected[l.id] = reasons
            else:
                res.eligible.append(l)
        return res
