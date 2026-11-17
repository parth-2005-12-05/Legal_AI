"""Resolve one internally consistent jurisdiction (design: "Resolving one
consistent jurisdiction").

Extraction yields several candidates per level (three "Kotwali" police stations,
two plausible districts ...). We pick the single combination that is both most
likely and consistent with the gazetteer hierarchy:

1. Collect candidates per level with scores (noisy-OR over every piece of
   evidence for the same entity: score x source weight x OCR confidence).
2. Evidence links (pincode, CNR prefix, a court or village) already arrive as
   district candidates or imply a district through their parent chain.
3. Tree dynamic programme (Viterbi-style) over the hierarchy: for each district
   candidate, total = its own score + agreeing state + best consistent tehsil +
   best consistent village + best consistent police station + best consistent
   court, minus penalties when strong evidence points outside that district.
   Candidate sets are tiny (< 10 per level), so this is effectively instant.
4. Report the winner, the runner-up and the margin; a small margin goes to review.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..models import Candidate
from .model import Gazetteer

LOWER_LEVELS = ["tehsil", "village", "police_station", "court"]
CONFLICT_PENALTY = 0.3
PS_TEHSIL_AGREEMENT_BONUS = 0.1


def noisy_or(probabilities) -> float:
    p = 1.0
    for x in probabilities:
        p *= 1.0 - max(0.0, min(1.0, x))
    return 1.0 - p


@dataclass
class Resolution:
    chosen: dict[str, str | None]
    level_scores: dict[str, dict[str, float]]
    district_totals: list[tuple[str, float]]
    margin: float
    issues: list[str] = field(default_factory=list)


class JurisdictionResolver:
    def __init__(self, gazetteer: Gazetteer, source_weights: dict[str, float]):
        self.gaz = gazetteer
        self.source_weights = source_weights

    def evidence_weight(self, c: Candidate) -> float:
        return c.score * self.source_weights.get(c.source, 0.5) * c.ocr_confidence

    def _level_scores(self, candidates: list[Candidate]) -> dict[str, dict[str, float]]:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for c in candidates:
            if c.value in self.gaz.entities:
                grouped[c.field][c.value].append(self.evidence_weight(c))
        return {lvl: {eid: noisy_or(ws) for eid, ws in ents.items()} for lvl, ents in grouped.items()}

    def resolve(self, candidates: list[Candidate]) -> Resolution:
        ev = self._level_scores(candidates)
        for lvl in ["state", "district", *LOWER_LEVELS]:
            ev.setdefault(lvl, {})

        districts = set(ev["district"])
        for lvl in LOWER_LEVELS:
            for eid in ev[lvl]:
                d = self.gaz.ancestor_of_type(eid, "district")
                if d:
                    districts.add(d)

        totals: list[tuple[str, float, dict[str, str | None], list[str]]] = []
        for d in districts:
            total, picks, issues = ev["district"].get(d, 0.0), {"district": d}, []
            state = self.gaz.ancestor_of_type(d, "state")
            picks["state"] = state
            if ev["state"]:
                if state in ev["state"]:
                    total += ev["state"][state]
                else:
                    total -= 0.5 * max(ev["state"].values())
                    issues.append(f"state evidence disagrees with district {self.gaz.name(d)}")
            others = {e: sc for e, sc in ev["district"].items() if e != d}
            if others:
                other = max(others, key=others.get)
                total -= CONFLICT_PENALTY * others[other]
                issues.append(f"district evidence also points to {self.gaz.name(other)}")
            for lvl in LOWER_LEVELS:
                scope = d
                if lvl == "village" and picks.get("tehsil"):
                    scope = picks["tehsil"]
                inside = {e: s for e, s in ev[lvl].items() if self.gaz.is_within(e, scope)}
                if lvl == "village" and not inside:
                    inside = {e: s for e, s in ev[lvl].items() if self.gaz.is_within(e, d)}
                outside = {e: s for e, s in ev[lvl].items() if not self.gaz.is_within(e, d)}
                best = max(inside.items(), key=lambda kv: kv[1]) if inside else (None, 0.0)
                picks[lvl] = best[0]
                total += best[1]
                if outside:
                    strongest = max(outside.values())
                    if strongest > best[1]:
                        total -= CONFLICT_PENALTY * strongest
                        eid = max(outside, key=outside.get)
                        issues.append(f"{lvl.replace('_', ' ')} evidence '{self.gaz.name(eid)}' "
                                      f"lies outside district {self.gaz.name(d)}")
            ps = self.gaz.get(picks.get("police_station"))
            if ps and picks.get("tehsil") and ps.attrs.get("tehsil") == picks["tehsil"]:
                total += PS_TEHSIL_AGREEMENT_BONUS
            totals.append((d, round(total, 4), picks, issues))

        if not totals:
            chosen = {lvl: None for lvl in ["state", "district", *LOWER_LEVELS]}
            if ev["state"]:
                chosen["state"] = max(ev["state"], key=ev["state"].get)
            return Resolution(chosen, ev, [], margin=1.0 if chosen["state"] else 0.0,
                              issues=["no district evidence found"])

        totals.sort(key=lambda t: -t[1])
        best_d, best_total, picks, issues = totals[0]
        if len(totals) > 1 and best_total > 0:
            margin = (best_total - totals[1][1]) / best_total
        else:
            margin = 1.0 if best_total > 0 else 0.0
        return Resolution(chosen=picks, level_scores=ev,
                          district_totals=[(d, t) for d, t, _, _ in totals],
                          margin=round(max(0.0, margin), 4), issues=issues)
