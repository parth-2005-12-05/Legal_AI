"""Prefix trie for type-ahead (design: Layer 4).

Powers "Which police station?" suggestions in the customer chat and in the
review console. Search can be constrained to a known parent: once the district
is fixed, only entities inside that district are suggested.
"""

from __future__ import annotations

from .model import Gazetteer
from ..normalise import fold


class _Node:
    __slots__ = ("children", "ids")

    def __init__(self) -> None:
        self.children: dict[str, _Node] = {}
        self.ids: set[str] = set()


class PrefixTrie:
    def __init__(self, gazetteer: Gazetteer, entity_type: str | None = None) -> None:
        self.gaz = gazetteer
        self.root = _Node()
        for alias in gazetteer.aliases_of_type(entity_type):
            for key in {alias.folded, alias.romanised}:
                self._insert(key, alias.entity_id)

    def _insert(self, key: str, entity_id: str) -> None:
        node = self.root
        for ch in key:
            node = node.children.setdefault(ch, _Node())
        node.ids.add(entity_id)

    def search(self, prefix: str, within: str | None = None, limit: int = 10) -> list[str]:
        """Entity ids having an alias starting with `prefix`, optionally restricted
        to entities inside `within` (an ancestor entity id)."""
        node = self.root
        for ch in fold(prefix):
            node = node.children.get(ch)
            if node is None:
                return []
        found: set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            found |= n.ids
            stack.extend(n.children.values())
        if within:
            found = {i for i in found if self.gaz.is_within(i, within)}
        return sorted(found, key=lambda i: self.gaz.entities[i].name)[:limit]
