"""Layer 1: label-value anchoring.

Indian legal documents are semi-structured: the value almost always sits right
after a label ("P.S. Kotwali", "जिला गाजियाबाद", "FIR No. 123/2023"). A table
of bilingual labels drives a key-value extractor that grabs the text to the
right of the label, within a window of about 60 characters, stopping at the end
of the line, a separator, or the next label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# field -> labels (English / Hindi). Order does not matter; longer labels win.
LABELS: dict[str, list[str]] = {
    "state": ["State", "राज्य", "प्रदेश"],
    "district": ["District", "Distt.", "Distt", "Dist.", "Janpad", "जिला", "ज़िला", "जनपद"],
    "tehsil": ["Tehsil", "Taluka", "Mandal", "Sub-division", "Sub Division", "तहसील", "परगना"],
    "village": ["Village", "Mauza", "Gram", "ग्राम", "मौजा", "गांव"],
    "police_station": ["Police Station", "P.S.", "PS", "Thana", "थाना", "पुलिस स्टेशन"],
    "court": ["In the Court of", "In The Court Of", "Before the", "न्यायालय", "माननीय"],
    "pincode": ["Pin Code", "Pincode", "PIN", "Pin", "पिन कोड", "पिन"],
    "opposing_party": ["Name of Accused", "Accused", "Opposite Party", "Respondent", "Defendant",
                       "आरोपी", "अभियुक्त", "विपक्षी"],
}
# Labels that are ordinary words in running text and only count at a line start.
LINE_START_ONLY = {"Before the", "माननीय", "न्यायालय", "Gram", "State", "Pin", "PIN"}
CASE_SENSITIVE = {"PS", "PIN"}
# Fields whose labels are also common words ("the accused forged ...") only count
# when a ":" or "-" follows the label.
REQUIRE_SEPARATOR = {"opposing_party", "state"}

WINDOW = 60
_STOP = re.compile(r"[\n;|]|,|\s{3,}| - ")
# Court names contain commas ("Chief Judicial Magistrate, Ghaziabad"), so the
# court label only stops at the end of the line.
_STOP_BY_FIELD = {"court": re.compile(r"[\n;|]")}

HEADER_LINES = 6  # the header block of an FIR / court order holds most jurisdiction fields


@dataclass
class AnchoredValue:
    field: str
    label: str
    value: str
    span: tuple[int, int]   # span of the value in the document text
    in_header: bool


def _label_regex(label: str) -> re.Pattern:
    body = re.escape(label).replace(r"\ ", r"\s*")
    prefix = r"(?:^|(?<=\n))\s*" if label in LINE_START_ONLY else r"(?<![\w\u0900-\u097f])"
    suffix = r"(?![\w\u0900-\u097f])" if label[-1].isalnum() else ""
    flags = 0 if label in CASE_SENSITIVE else re.IGNORECASE
    return re.compile(prefix + "(" + body + ")" + suffix, flags)


_COMPILED = [(field, label, _label_regex(label))
             for field, labels in LABELS.items()
             for label in sorted(labels, key=len, reverse=True)]


def find_labels(text: str) -> list[tuple[str, str, int, int]]:
    """All label occurrences as (field, label, start, end); overlapping shorter
    labels are dropped ("Police Station" wins over "PS")."""
    hits = []
    for field, label, rx in _COMPILED:
        for m in rx.finditer(text):
            hits.append((field, label, m.start(1), m.end(1)))
    hits.sort(key=lambda h: (h[2], -(h[3] - h[2])))
    kept: list[tuple[str, str, int, int]] = []
    for h in hits:
        if kept and h[2] < kept[-1][3]:
            continue
        kept.append(h)
    return kept


def anchor_values(text: str) -> list[AnchoredValue]:
    labels = find_labels(text)
    header_end = sum(len(line) + 1 for line in text.split("\n")[:HEADER_LINES])
    out: list[AnchoredValue] = []
    for idx, (field, label, start, end) in enumerate(labels):
        next_label = labels[idx + 1][2] if idx + 1 < len(labels) else len(text)
        # skip separators right after the label: ":", "-", ".", "of", "no."
        if field in REQUIRE_SEPARATOR and not re.match(r"\s*[:\-–]", text[end:]):
            continue
        m = re.match(r"[\s:\-–.]*(?:of\s+)?", text[end:])
        v_start = end + (m.end() if m else 0)
        limit = min(v_start + WINDOW, next_label, len(text))
        stop = _STOP_BY_FIELD.get(field, _STOP).search(text, v_start, limit)
        v_end = stop.start() if stop else limit
        value = text[v_start:v_end].strip(" .:-")
        if not value:
            continue
        v_start = text.index(value, v_start)
        out.append(AnchoredValue(field, label, value, (v_start, v_start + len(value)), start < header_end))
    return out
