"""Jurisdiction gazetteer and the algorithms that search it.

    model.py        Gazetteer: entities, aliases, materialised paths, pincode / CNR indexes
    ahocorasick.py  one-pass multi-pattern scanner (design: Layer 3)
    trie.py         prefix search for type-ahead, constrained by parent (Layer 4)
    fuzzy.py        trigram index + BK-tree + phonetic key + Jaro-Winkler ranking (Layer 5)
    resolver.py     tree dynamic programme that picks one consistent jurisdiction
"""

from .model import Entity, Gazetteer, load_gazetteer

__all__ = ["Entity", "Gazetteer", "load_gazetteer"]
