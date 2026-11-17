"""Matching engine: eligibility -> scoring -> assignment -> routing.

`match(case)` serves real-time mode (one case, priority-queue greedy).
`match_batch(cases)` serves batch mode (min-cost flow over all pending cases);
the flow picks each case's top-1 and the next best eligible lawyers become the
two alternates. Both return MatchDecision objects carrying the score breakdown,
reasons, route and any flags.
"""

from __future__ import annotations

from ..config import MatchingSettings
from ..gazetteer.model import Gazetteer
from ..models import MatchDecision
from .assignment import batch_assign, greedy_rank
from .eligibility import EligibilityFilter
from .profiles import CaseFacts, Lawyer
from .router import decide_route
from .scoring import Scorer


class MatchingEngine:
    def __init__(self, gazetteer: Gazetteer, lawyers: list[Lawyer], settings: MatchingSettings,
                 taxonomy_pairs: list[tuple[str, str]]):
        self.gaz, self.lawyers, self.s = gazetteer, lawyers, settings
        self.filter = EligibilityFilter(gazetteer, settings.conflict_threshold,
                                        settings.urgent_subtypes, settings.urgent_max_hours)
        self.scorer = Scorer(gazetteer, settings.weights, settings.rotation_bonus).set_taxonomy(taxonomy_pairs)

    def capacity(self) -> dict[str, int]:
        return {l.id: max(0, l.capacity - l.active_matters) for l in self.lawyers}

    def _scored(self, case: CaseFacts):
        elig = self.filter.check(self.lawyers, case)
        scored = sorted((self.scorer.score(case, l) for l in elig.eligible), key=lambda m: -m.score)
        # a lawyer with no practice-area fit at all is not a sensible suggestion
        for m in scored:
            if m.breakdown["spec"] == 0:
                elig.rejected.setdefault(m.lawyer_id, []).append("no practice in this area of law")
        scored = [m for m in scored if m.breakdown["spec"] > 0]
        return elig, scored

    def match(self, case: CaseFacts) -> MatchDecision:
        elig, scored = self._scored(case)
        top = greedy_rank(scored, self.capacity(), self.s.top_n)
        route, reasons, reviewers, flags = decide_route(
            top, self.s, conflict=bool(elig.conflicts), urgent=self.filter.is_urgent(case))
        return MatchDecision(case.case_id, top, route, reasons, reviewers, flags,
                             rejected=elig.rejected, conflicts=elig.conflicts)

    def match_batch(self, cases: list[CaseFacts]) -> list[MatchDecision]:
        prepared = {c.case_id: (c, *self._scored(c)) for c in cases}
        scores = {cid: {m.lawyer_id: m.score for m in scored} for cid, (_, _, scored) in prepared.items()}
        assigned = batch_assign(scores, self.capacity())
        decisions = []
        for cid, (case, elig, scored) in prepared.items():
            first = [m for m in scored if m.lawyer_id == assigned.get(cid)]
            rest = [m for m in scored if m.lawyer_id != assigned.get(cid)]
            top = (first + rest)[: self.s.top_n]
            route, reasons, reviewers, flags = decide_route(
                top, self.s, conflict=bool(elig.conflicts), urgent=self.filter.is_urgent(case))
            decisions.append(MatchDecision(cid, top, route, reasons, reviewers, flags,
                                           rejected=elig.rejected, conflicts=elig.conflicts))
        return decisions
