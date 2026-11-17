"""Step 2: weighted score (design formula).

    S(c,l) = w1*spec + w2*court + w3*exp + w4*lang + w5*prox + w6*fee - w7*load

    spec   1.0 exact sub-type, 0.6 same top-level, 0 otherwise; weighted by the
           classifier's probability for multi-label cases
    court  matters in this exact court in 3 years, log-scaled; 0.5 for same district
    exp    log-scaled closed matters of this sub-type; seniority bump for complex matters
    lang   1 if fluent in the customer's language, 0.5 if documents only
    prox   inverse travel distance to the court (precomputed table)
    fee    1 if the fee band matches, decreasing per band apart (0.5 if unknown)
    load   (active / capacity)^2, a convex penalty so near-full lawyers are avoided

Every score keeps its breakdown and a sentence of reasons, e.g.
"practises in CJM Court Ghaziabad (24 matters), cheating specialist, Hindi, 6/10 load -> 0.86".
"""

from __future__ import annotations

import datetime as _dt
import math

from ..gazetteer.model import Gazetteer
from ..models import LawyerMatch
from .profiles import CaseFacts, Lawyer

COMPLEX_SUBTYPES = {"murder", "ndps", "insolvency", "writ", "land_acquisition", "company"}
LANG_NAMES = {"hi": "Hindi", "en": "English", "mr": "Marathi", "pa": "Punjabi", "bn": "Bengali",
              "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "gu": "Gujarati"}
DEFAULT_DISTANCE_KM = 30.0


class Scorer:
    def __init__(self, gazetteer: Gazetteer, weights: dict[str, float], rotation_bonus: float = 0.02,
                 today: _dt.date | None = None):
        self.gaz, self.w = gazetteer, weights
        self.rotation_bonus = rotation_bonus
        self.today = today or _dt.date.today()

    _sub_to_top: dict[str, str] = {}

    def set_taxonomy(self, subtypes: list[tuple[str, str]]) -> "Scorer":
        """Give the scorer the taxonomy (top, sub) pairs so it can tell which
        top-level types a lawyer practises from their sub-types."""
        self._sub_to_top = {s: t for t, s in subtypes}
        return self

    def _top_levels(self, lawyer: Lawyer) -> set[str]:
        return {self._sub_to_top.get(s, "") for s in lawyer.practice}

    def terms(self, case: CaseFacts, l: Lawyer) -> tuple[dict[str, float], list[str]]:
        reasons: list[str] = []
        labels = case.labels or ([(case.top, case.sub, 1.0)] if case.top else [])
        tops = self._top_levels(l)
        num = den = 0.0
        for top, sub, p in labels:
            m = 1.0 if sub and sub in l.practice else (0.6 if top in tops else 0.0)
            num, den = num + p * m, den + p
        spec = num / den if den else 0.0
        if case.sub and case.sub in l.practice:
            reasons.append(f"{case.sub.replace('_', ' ')} specialist")

        court_term = 0.0
        if case.court and case.court in l.courts:
            n = l.courts[case.court]
            court_term = min(1.0, math.log1p(n) / math.log1p(50))
            reasons.append(f"practises in {self.gaz.name(case.court)} ({n} matters)")
        elif case.district and any(self.gaz.ancestor_of_type(c, "district") == case.district for c in l.courts):
            court_term = 0.5
            reasons.append(f"practises in {self.gaz.name(case.district)} courts")

        closed = l.experience.get(case.sub or "", 0)
        exp = min(1.0, math.log1p(closed) / math.log1p(100))
        if case.sub in COMPLEX_SUBTYPES and l.seniority == "senior":
            exp = min(1.0, exp + 0.2)
        if closed:
            reasons.append(f"{closed} closed {case.sub.replace('_', ' ')} matters")

        if case.customer_language in l.languages.get("fluent", []):
            lang = 1.0
            reasons.append(LANG_NAMES.get(case.customer_language, case.customer_language))
        elif case.document_language in l.languages.get("fluent", []) + l.languages.get("read", []):
            lang = 0.5
        else:
            lang = 0.0

        km = l.distance_km.get(case.court or "", l.distance_km.get(case.district or "", DEFAULT_DISTANCE_KM))
        prox = 1.0 / (1.0 + km / 10.0)

        fee = 0.5 if case.fee_band is None else max(0.0, 1.0 - 0.34 * abs(case.fee_band - l.fee_band))
        load = (l.active_matters / l.capacity) ** 2 if l.capacity else 1.0
        reasons.append(f"{l.active_matters}/{l.capacity} load")
        return {"spec": spec, "court": court_term, "exp": exp, "lang": lang, "prox": prox,
                "fee": fee, "load": load}, reasons

    def score(self, case: CaseFacts, l: Lawyer) -> LawyerMatch:
        t, reasons = self.terms(case, l)
        s = sum(self.w.get(k, 0) * v for k, v in t.items() if k != "load") - self.w.get("load", 0) * t["load"]
        # fairness: small rotation bonus for juniors who have waited longest for this type
        if l.seniority == "junior" and case.top:
            last = l.last_assigned.get(case.top)
            days = (self.today - _dt.date.fromisoformat(last)).days if last else 90
            bonus = self.rotation_bonus * min(max(days, 0) / 30.0, 1.0)
            if bonus:
                t["rotation"] = bonus
                s += bonus
        s = round(max(0.0, s), 4)
        breakdown = {k: round(v, 3) for k, v in t.items()}
        reasons_text = ", ".join(reasons) + f" -> {s:.2f}"
        profile = {
            "name": l.name,
            "courts": [self.gaz.name(c) for c in l.courts],
            "practice_areas": [p.replace("_", " ") for p in l.practice],
            "languages": [LANG_NAMES.get(x, x) for x in l.languages.get("fluent", [])],
            "years_of_experience": l.years,
            "fee_band": l.fee_band,
            "next_available": f"within {l.available_within_hours} hours",
        }
        return LawyerMatch(l.id, l.name, s, breakdown, [reasons_text], profile)
