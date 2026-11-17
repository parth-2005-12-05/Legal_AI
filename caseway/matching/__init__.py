"""Lawyer matching (design: "Lawyer matching").

    profiles.py     Lawyer roster model + CaseFacts (the verified facts matching uses)
    eligibility.py  Step 1: hard filter (court access, conflict of interest,
                    availability, language, urgency rules)
    scoring.py      Step 2: transparent weighted score with a per-term breakdown
    assignment.py   Step 3: priority-queue greedy (real time) and min-cost flow (batch)
    router.py       where the result goes: customer / partner queue / senior review,
                    driven by the partner_review switch
    engine.py       glue: one case or a batch in, MatchDecision(s) out
"""

from .engine import MatchingEngine
from .profiles import CaseFacts, Lawyer, load_lawyers

__all__ = ["MatchingEngine", "CaseFacts", "Lawyer", "load_lawyers"]
