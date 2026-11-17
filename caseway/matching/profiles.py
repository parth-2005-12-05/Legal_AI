"""Lawyer profiles and the case facts the matcher consumes.

Lawyer fields follow the design's eligibility and scoring terms: Bar Council
enrolment, courts practised in (with 3-year matter counts), practice areas,
closed matters per sub-type, languages, travel distance, fee band, capacity and
current load, leave, urgent-criminal flag, and past clients for conflict checks.
Win rate is deliberately absent: the design rules it out as noisy and gameable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Lawyer:
    id: str
    name: str
    bar_council: str
    enrolment_no: str
    years: int
    seniority: str                       # junior | mid | senior
    courts: dict[str, int]               # court id -> matters in last 3 years
    practice: list[str]                  # sub-types practised
    experience: dict[str, int]           # sub-type -> closed matters
    languages: dict[str, list[str]]      # {"fluent": [...], "read": [...]}
    distance_km: dict[str, float]        # court or district id -> km from base
    fee_band: int                        # 1 (lowest) .. 4
    capacity: int
    active_matters: int
    on_leave: bool = False
    urgent_criminal: bool = False
    available_within_hours: int = 72
    clients: list[str] = field(default_factory=list)
    busy_dates: list[str] = field(default_factory=list)
    last_assigned: dict[str, str] = field(default_factory=dict)  # top-level type -> ISO date


@dataclass
class CaseFacts:
    """Verified facts for one case (output of fusion / paralegal review)."""
    case_id: str
    state: str | None
    district: str | None
    court: str | None
    top: str | None
    sub: str | None
    labels: list[tuple[str, str, float]] = field(default_factory=list)  # multi-label (top, sub, p)
    customer_language: str = "hi"
    document_language: str = "hi"
    fee_band: int | None = None
    opposing_parties: list[str] = field(default_factory=list)
    next_hearing: str | None = None


def load_lawyers(path: str | Path) -> list[Lawyer]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [Lawyer(**item) for item in raw["lawyers"]]
