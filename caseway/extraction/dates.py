"""Date extraction with validation and roles.

Grammar: dd/mm/yyyy, dd-mm-yy, dd.mm.yyyy, "15 Nov 2026", "15 नवंबर 2026".
Validation: must be a real calendar date, year 1950..today.
Role: taken from the nearest label in the ~50 characters before the date
(occurrence, registration, order, next_hearing, invoice). A date with no label
nearby gets role "unknown" and a lower score.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    "जनवरी": 1, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6, "जुलाई": 7, "अगस्त": 8,
    "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10, "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}
_MONTH_ALT = "|".join(sorted(map(re.escape, MONTHS), key=len, reverse=True))
DATE = re.compile(
    r"(?<!\d)(?P<d>[0-3]?\d)[./-](?P<m>[01]?\d)[./-](?P<y>(?:19|20)?\d{2})(?!\d)"
    r"|(?<!\d)(?P<d2>[0-3]?\d)(?:st|nd|rd|th)?\s+(?P<mon>" + _MONTH_ALT + r")\.?,?\s+(?P<y2>(?:19|20)\d{2})(?!\d)",
    re.IGNORECASE)

ROLE_LABELS = {
    "occurrence": ["date of occurrence", "date of incident", "occurrence", "incident on", "घटना की तिथि",
                   "घटना दिनांक", "घटना"],
    "registration": ["date of report", "date of registration", "registered on", "reported on", "दर्ज",
                     "रिपोर्ट की तिथि", "सूचना की तिथि"],
    "order": ["order dated", "date of order", "आदेश दिनांक", "आदेश"],
    "next_hearing": ["next date", "next hearing", "date fixed", "अगली तिथि", "नियत तिथि", "अगली सुनवाई"],
    "invoice": ["invoice date", "bill date", "date of purchase", "purchased on", "बिल दिनांक"],
}
LOOKBACK = 50


@dataclass
class DateHit:
    date: _dt.date
    role: str
    span: tuple[int, int]
    text: str
    score: float


def _role(text: str, start: int) -> str:
    window = text[max(0, start - LOOKBACK):start].lower()
    best, best_pos = "unknown", -1
    for role, labels in ROLE_LABELS.items():
        for label in labels:
            pos = window.rfind(label)
            if pos > best_pos:
                best, best_pos = role, pos
    return best


def find_dates(text: str, today: _dt.date | None = None) -> list[DateHit]:
    today = today or _dt.date.today()
    hits = []
    for m in DATE.finditer(text):
        if m["d"]:
            d, mo, y = int(m["d"]), int(m["m"]), m["y"]
        else:
            d, mo, y = int(m["d2"]), MONTHS[m["mon"].lower()], m["y2"]
        year = int(y)
        if len(y) == 2:
            year += 2000 if year <= today.year % 100 else 1900
        try:
            date = _dt.date(year, mo, d)
        except ValueError:
            continue  # not a real calendar date
        role = _role(text, m.start())
        # next hearing dates may be in the future; everything else may not.
        if year < 1950 or (date > today and role != "next_hearing"):
            continue
        hits.append(DateHit(date, role, m.span(), m.group(0), 1.0 if role != "unknown" else 0.6))
    return hits
