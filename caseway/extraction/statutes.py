"""Statute tables: IPC <-> BNS mapping and section -> case sub-type.

Since 1 July 2024 new FIRs cite BNS / BNSS / BSA. A curated two-way table maps
old and new sections to the same offence category; the occurrence date decides
which code applies when a section is cited without an act. The tables live in
data/classifier/statutes.json and must be signed off by the legal team.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path


class StatuteTable:
    def __init__(self, path: str | Path):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.bns_from = _dt.date.fromisoformat(raw.get("bns_effective_from", "2024-07-01"))
        self.old_to_new = {a: b for a, b in raw["ipc_bns"]}
        self.new_to_old = {b: a for a, b in raw["ipc_bns"]}
        self.section_to_type = {k: tuple(v) for k, v in raw["section_to_type"].items()}
        self.act_to_type = {k: tuple(v) for k, v in raw.get("act_to_type", {}).items()}

    def resolve_act(self, act: str | None, occurrence: _dt.date | None) -> str:
        """Fill a missing act from the occurrence date (IPC before 1 July 2024, BNS after)."""
        if act:
            return act
        if occurrence is None:
            return "IPC"  # most pending matters still cite IPC; flagged as an assumption
        return "BNS" if occurrence >= self.bns_from else "IPC"

    def canonical(self, key: str) -> str:
        """Map a BNS/BNSS key to its IPC/CrPC twin so both share one category."""
        return self.new_to_old.get(key, key)

    def case_type(self, key: str) -> tuple[str, str | None] | None:
        key = self.canonical(key)
        if key in self.section_to_type:
            return self.section_to_type[key]
        act = key.split(" ", 1)[0]
        return self.act_to_type.get(act)
