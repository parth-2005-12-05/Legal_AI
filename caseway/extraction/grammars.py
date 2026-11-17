"""Layer 2: regex grammars for identifiers.

Each identifier type is a small grammar compiled once and run on normalised,
case-insensitive text (patterns follow the design document). Post-checks
"validate, don't just match": years must be 1950..current year, and a CNR's
first four letters must map to a known state + district in the gazetteer.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass

I = re.IGNORECASE

# CNR (e-Courts unique case id): SS DD EE NNNNNN YYYY, e.g. MHAU010012342022.
CNR = re.compile(r"\b(?P<cnr>(?P<sd>[A-Z]{4})(?P<est>\d{2})(?P<num>\d{6})(?P<year>\d{4}))\b")

CASE_TYPE = (r"(?P<ctype>O\.?\s?S\.?|C\.?\s?S\.?\s?(?:\((?:OS|COMM)\))?|Crl\.?\s?A\.?|Crl\.?\s?Rev\.?"
             r"|CRL\.?\s?M\.?\s?C\.?|W\.?\s?P\.?\s?(?:\((?:C|Crl)\))?|M\.?\s?A\.?\s?C\.?\s?T\.?"
             r"|R\.?\s?C\.?\s?A\.?|C\.?\s?C\.?|S\.?\s?T\.?|Bail\s?Appl?n?\.?|HMA|CMP|Execution"
             r"|Complaint\s?Case)")
CASE_NO = re.compile(r"(?<![A-Za-z])" + CASE_TYPE +
                     r"\s*(?:No\.?|Number|सं\.?|संख्या)?\s*(?P<num>\d{1,6})\s*(?:/|of|सन्|वर्ष)\s*"
                     r"(?P<year>(?:19|20)?\d{2})\b", I)

FIR_NO = re.compile(r"(?:\bF[I1l]R\b|\bCase\s?Crime\b|\bCrime\b|प्रथम\s?सूचना|मु[.0]?\s?अ[.0]?\s?स[.0]?)"
                    r"\s*(?:No\.?|सं\.?|संख्या|Number)?\s*[:\-]?\s*(?P<num>\d{1,5})\s*[/-]\s*"
                    r"(?P<year>\d{4}|\d{2})\b", I)

SEC_ITEM = r"\d{1,3}[A-Z]?(?:-[A-Z])?(?:\(\d+\))?(?:\([a-z]\))?"
ACT = (r"(?P<act>I\.?\s?P\.?\s?C\.?|BNSS|BNS|Cr\.?\s?P\.?\s?C\.?|C\.?\s?P\.?\s?C\.?|NDPS(?:\s?Act)?|POCSO(?:\s?Act)?"
       r"|SC/ST(?:\s?Act)?|Arms\s?Act|N\.?\s?I\.?\s?Act|D\.?\s?V\.?\s?Act|D\.?\s?P\.?\s?Act|HMA|Hindu\s?Marriage\s?Act"
       r"|I\.?\s?T\.?\s?Act|Consumer\s?Protection\s?Act|भा\.?\s?द\.?\s?वि\.?|भा\.?\s?द\.?\s?सं\.?|बीएनएस)")
SECTIONS = re.compile(r"(?:\bu/s\b|\bsections?\b|\bunder\s+section\b|धारा|अंतर्गत\s+धारा)\s*"
                      r"(?P<list>" + SEC_ITEM + r"(?:\s*(?:,|/|&|\band\b|व|एवं)\s*" + SEC_ITEM + r")*)"
                      r"\s*(?:of\s+(?:the\s+)?)?" + ACT + "?", I)
_SEC_SPLIT = re.compile(r"\s*(?:,|/|&|\band\b|व|एवं)\s*", I)

LAND_ID = re.compile(r"(?P<kind>Khasra|Khata|Khatauni|Survey|Gata|खसरा|खाता|खतौनी|गाटा)\s*"
                     r"(?:No\.?|सं\.?|संख्या)?\s*[:\-]?\s*(?P<id>\d+(?:/\d+)?[A-Z]?)\b", I)

PINCODE = re.compile(r"(?<!\d)(?P<pin>[1-9]\d{5})(?!\d)")

_ACT_CANON = [
    (r"i\.?\s?p\.?\s?c|भा\.?\s?द\.?\s?(?:वि|सं)", "IPC"), (r"bnss", "BNSS"), (r"bns|बीएनएस", "BNS"),
    (r"cr\.?\s?p\.?\s?c", "CrPC"), (r"c\.?\s?p\.?\s?c", "CPC"), (r"ndps", "NDPS"), (r"pocso", "POCSO"),
    (r"sc/st", "SCST"), (r"arms", "Arms"), (r"n\.?\s?i\.?\s?act", "NI"), (r"d\.?\s?v\.?\s?act", "DV"),
    (r"d\.?\s?p\.?\s?act", "DP"), (r"hma|hindu\s?marriage", "HMA"), (r"i\.?\s?t\.?\s?act", "IT"),
    (r"consumer", "CPA"),
]


def canonical_act(raw: str | None) -> str | None:
    if not raw:
        return None
    for pattern, canon in _ACT_CANON:
        if re.fullmatch(pattern + r"\.?(?:\s?act)?", raw.strip().lower()):
            return canon
    return raw.strip().upper()


def normalise_year(raw: str) -> int | None:
    """Two-digit years: <= current yy -> 20yy, else 19yy. Valid range 1950..now."""
    now = _dt.date.today().year
    year = int(raw)
    if len(raw) == 2:
        year += 2000 if year <= now % 100 else 1900
    return year if 1950 <= year <= now else None


@dataclass
class Hit:
    kind: str
    value: str
    span: tuple[int, int]
    parts: dict


def find_cnr(text: str) -> list[Hit]:
    hits = []
    for m in CNR.finditer(text):
        year = normalise_year(m["year"])
        if year:
            hits.append(Hit("cnr", m["cnr"], m.span("cnr"),
                            {"district_code": m["sd"], "establishment": m["sd"] + m["est"], "year": year}))
    return hits


def find_case_numbers(text: str) -> list[Hit]:
    hits = []
    for m in CASE_NO.finditer(text):
        year = normalise_year(m["year"])
        if year:
            ctype = re.sub(r"[\s.]", "", m["ctype"]).upper()
            hits.append(Hit("case_number", f"{ctype} {int(m['num'])}/{year}", m.span(),
                            {"prefix": ctype, "num": int(m["num"]), "year": year}))
    return hits


def find_fir_numbers(text: str) -> list[Hit]:
    hits = []
    for m in FIR_NO.finditer(text):
        year = normalise_year(m["year"])
        if year:
            hits.append(Hit("fir_number", f"{int(m['num'])}/{year}", m.span(), {"num": int(m["num"]), "year": year}))
    return hits


def find_sections(text: str) -> list[Hit]:
    hits = []
    for m in SECTIONS.finditer(text):
        act = canonical_act(m["act"])
        for item in _SEC_SPLIT.split(m["list"]):
            item = item.strip().upper().replace("-", "")
            if item:
                hits.append(Hit("section", f"{act or '?'} {item}", m.span(), {"act": act, "section": item}))
    return hits


def find_land_ids(text: str) -> list[Hit]:
    return [Hit("land_id", f"{m['kind'].title()} {m['id']}", m.span(), {"kind": m["kind"], "id": m["id"]})
            for m in LAND_ID.finditer(text)]


def find_pincodes(text: str) -> list[Hit]:
    return [Hit("pincode", m["pin"], m.span("pin"), {}) for m in PINCODE.finditer(text)]
