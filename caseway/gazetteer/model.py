"""Gazetteer data model and loader.

Mirrors the PostgreSQL model in the design, held in memory for v1:

    entity(id, type, lgd_code, parent_id, state_id, valid_from, valid_to)
    alias(entity_id, text, script, romanised, phonetic_key, source)
    materialised path on every entity, e.g. /UP/UP-GZB/UP-GZB-PS-KOT,
    so "is X inside Y" is a string-prefix check.

The loader also builds the indexes other stages need: pincode -> district,
CNR prefix (first 4 letters) -> district, establishment code -> court, and the
territorial decision table used to derive a court when no document names one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..normalise import fold, phonetic_key, romanise

LEVELS = ["state", "district", "tehsil", "village", "police_station", "court"]


@dataclass
class Alias:
    entity_id: str
    text: str
    script: str          # "deva" or "latn"
    folded: str          # fold(text): exact-lookup key in original script
    romanised: str       # fold(romanise(text)): lets Latin queries reach Hindi aliases
    phonetic: str        # phonetic_key(text)
    source: str = "seed"


@dataclass
class Entity:
    id: str
    type: str
    name: str
    parent: str | None = None
    lgd_code: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    path: str = ""
    state_id: str | None = None


def _script(text: str) -> str:
    return "deva" if any("ऀ" <= c <= "ॿ" for c in text) else "latn"


class Gazetteer:
    """In-memory gazetteer with the lookups every stage needs."""

    def __init__(self, entities: Iterable[Entity], aliases: Iterable[Alias],
                 territorial_rules: dict | None = None, version: str = "unknown"):
        self.version = version
        self.entities: dict[str, Entity] = {e.id: e for e in entities}
        self.aliases: list[Alias] = list(aliases)
        self.territorial_rules = territorial_rules or {}
        self.children: dict[str, list[str]] = defaultdict(list)
        self.by_type: dict[str, list[str]] = defaultdict(list)
        self.by_folded: dict[str, set[str]] = defaultdict(set)
        self.pincode_to_district: dict[str, str] = {}
        self.cnr_to_district: dict[str, str] = {}
        self.establishment_to_court: dict[str, str] = {}
        self._build()

    # -- construction -----------------------------------------------------
    def _build(self) -> None:
        for e in self.entities.values():
            self.by_type[e.type].append(e.id)
            if e.parent:
                self.children[e.parent].append(e.id)
        for e in self.entities.values():
            chain = []
            cur: Entity | None = e
            while cur is not None:
                chain.append(cur.id)
                cur = self.entities.get(cur.parent) if cur.parent else None
            e.path = "/" + "/".join(reversed(chain))
            e.state_id = chain[-1] if self.entities[chain[-1]].type == "state" else None
            for pin in e.attrs.get("pincodes", []):
                self.pincode_to_district[pin] = e.id
            if e.attrs.get("cnr_code"):
                self.cnr_to_district[e.attrs["cnr_code"].upper()] = e.id
            if e.attrs.get("establishment_code"):
                self.establishment_to_court[e.attrs["establishment_code"].upper()] = e.id
        for a in self.aliases:
            self.by_folded[a.folded].add(a.entity_id)
            self.by_folded[a.romanised].add(a.entity_id)

    # -- queries ----------------------------------------------------------
    def get(self, entity_id: str | None) -> Entity | None:
        return self.entities.get(entity_id) if entity_id else None

    def name(self, entity_id: str | None) -> str | None:
        e = self.get(entity_id)
        return e.name if e else None

    def is_within(self, entity_id: str, ancestor_id: str) -> bool:
        """True if `entity_id` is `ancestor_id` or lies inside it (path prefix check)."""
        e, a = self.get(entity_id), self.get(ancestor_id)
        if not e or not a:
            return False
        return e.path == a.path or e.path.startswith(a.path + "/")

    def ancestor_of_type(self, entity_id: str, type_: str) -> str | None:
        cur = self.get(entity_id)
        while cur is not None:
            if cur.type == type_:
                return cur.id
            cur = self.get(cur.parent)
        return None

    def descendants_of_type(self, ancestor_id: str, type_: str) -> list[str]:
        return [eid for eid in self.by_type.get(type_, []) if self.is_within(eid, ancestor_id)]

    def lookup_exact(self, text: str, type_: str | None = None) -> list[str]:
        """Entity ids whose alias equals `text` (after folding / romanising)."""
        ids = set(self.by_folded.get(fold(text), set())) | set(self.by_folded.get(fold(romanise(text)), set()))
        return sorted(i for i in ids if type_ is None or self.entities[i].type == type_)

    def aliases_of_type(self, type_: str | None) -> list[Alias]:
        return [a for a in self.aliases if type_ is None or self.entities[a.entity_id].type == type_]

    def courts_for(self, district_id: str, category: str, subtype: str | None = None) -> list[str]:
        """Courts in a district that hear a category, ordered by the territorial
        decision table (e.g. criminal bail -> Sessions Court first)."""
        rules = self.territorial_rules.get(category, {})
        kinds = rules.get(subtype or "", rules.get("default", []))
        courts = self.descendants_of_type(district_id, "court")
        ordered: list[str] = []
        for kind in kinds:
            ordered += [c for c in courts if self.entities[c].attrs.get("kind") == kind and c not in ordered]
        ordered += [c for c in courts if category in self.entities[c].attrs.get("categories", []) and c not in ordered]
        return ordered


def load_gazetteer(path: str | Path) -> Gazetteer:
    """Load a gazetteer JSON bundle (see data/gazetteer/sample_gazetteer.json)."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    entities, aliases = [], []
    for item in raw["entities"]:
        e = Entity(id=item["id"], type=item["type"], name=item["name"], parent=item.get("parent"),
                   lgd_code=item.get("lgd_code"), valid_from=item.get("valid_from"),
                   valid_to=item.get("valid_to"), attrs=item.get("attrs", {}))
        entities.append(e)
        for text in {item["name"], *item.get("aliases", [])}:
            aliases.append(Alias(entity_id=e.id, text=text, script=_script(text), folded=fold(text),
                                 romanised=fold(romanise(text)), phonetic=phonetic_key(text)))
    rules = {k: v for k, v in raw.get("territorial_rules", {}).items() if not k.startswith("_")}
    return Gazetteer(entities, aliases, rules, version=raw.get("version", "unknown"))
