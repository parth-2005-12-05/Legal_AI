"""Entity extraction (design: "Entity extraction algorithms").

A layered, deterministic pipeline: exact patterns first, dictionary scans next,
fuzzy matching only for what is left. Every layer emits Candidates carrying a
text span, page, rule id and score; nothing is decided until fusion.

    labels.py    Layer 1: bilingual label -> value anchoring
    grammars.py  Layer 2: regex grammars (CNR, case no, FIR, sections, land ids, pincode)
    dates.py     date grammar + validation + role from the nearest label
    doctype.py   document-type detection from header keywords
    statutes.py  IPC <-> BNS table and section -> case sub-type map
    engine.py    runs all layers (plus the gazetteer scan and fuzzy layer) on a Document
"""

from .engine import Extractor

__all__ = ["Extractor"]
