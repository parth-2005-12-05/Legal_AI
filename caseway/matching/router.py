"""Where a match goes next: the partner_review switch lives here.

    partner_review: false (default)
        Customers see the top 3 straight away; their pick assigns the case.
    partner_review: true
        Every match waits in a partner queue; the partner accepts or overrides
        with a reason, then the customer sees the list.

Whatever the switch says, these ALWAYS stop for a named senior lawyer or admin
(matching.conflict_reviewers) so no risky case goes out unchecked:
    conflict  - a conflict-of-interest hit on any lawyer
    urgent    - bail / anticipatory-bail matters
    low_score - the best score is below matching.min_score
    close_call- top-1 and top-2 are within matching.close_call_margin
    no_match  - nobody passed the eligibility filter

The partner role, queue and audit entries always exist in code; with the switch
off they are simply never routed to (and the API hides the partner endpoints),
so turning it on is one setting plus a restart.
"""

from __future__ import annotations

from ..config import MatchingSettings
from ..models import LawyerMatch

CUSTOMER, PARTNER_QUEUE, SENIOR_REVIEW = "customer", "partner_queue", "senior_review"


def decide_route(matches: list[LawyerMatch], settings: MatchingSettings, *, conflict: bool,
                 urgent: bool) -> tuple[str, list[str], list[str], list[str]]:
    """Return (route, reasons, reviewers, flags)."""
    flags, reasons = [], []
    if conflict:
        flags.append("conflict")
        reasons.append("conflict-of-interest hit: a named reviewer must check before anyone sees a match")
    if urgent:
        flags.append("urgent")
        reasons.append("urgent bail matter: a named reviewer confirms the lawyer can act within 24 hours")
    if not matches:
        flags.append("no_match")
        reasons.append("no lawyer passed the eligibility filter")
    else:
        # sort scores: in batch mode the flow's pick is listed first even if an
        # alternate scores slightly higher, so compare the two best scores.
        best = sorted((m.score for m in matches), reverse=True)
        if best[0] < settings.min_score:
            flags.append("low_score")
            reasons.append(f"best score {best[0]:.2f} is below {settings.min_score:.2f}")
        if len(best) > 1 and best[0] - best[1] < settings.close_call_margin:
            flags.append("close_call")
            reasons.append(f"top two lawyers are within {settings.close_call_margin:.2f} of each other")
    if flags:
        return SENIOR_REVIEW, reasons, list(settings.conflict_reviewers), flags
    if settings.partner_review:
        return PARTNER_QUEUE, ["partner review is switched on: a partner confirms before the customer sees it"], ["partner"], flags
    return CUSTOMER, ["partner review is off: the customer sees the top matches directly"], [], flags


def next_route_after_senior_review(settings: MatchingSettings) -> str:
    """Once a senior lawyer / admin has cleared a flagged case, it continues on the
    normal path: to the partner if the switch is on, otherwise to the customer."""
    return PARTNER_QUEUE if settings.partner_review else CUSTOMER
