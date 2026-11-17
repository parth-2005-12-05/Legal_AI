"""Tests for extraction: regex grammars, dates, labels, doc type and the engine."""

import datetime as dt

from caseway.extraction import Extractor, grammars
from caseway.extraction.dates import find_dates
from caseway.extraction.doctype import detect_doc_type
from caseway.extraction.labels import anchor_values
from caseway.models import Document
from caseway.normalise import normalise
from conftest import SAMPLES


def test_cnr_requires_valid_year():
    assert [h.value for h in grammars.find_cnr("CNR UPGZ010012342022")] == ["UPGZ010012342022"]
    assert grammars.find_cnr("UPGZ010012341899") == []


def test_case_numbers():
    hits = grammars.find_case_numbers("O.S. No. 123/2019 and Crl.A. 12 of 2021 and HMA 45/21")
    assert [h.value for h in hits] == ["OS 123/2019", "CRLA 12/2021", "HMA 45/2021"]


def test_fir_numbers_english_and_hindi():
    assert grammars.find_fir_numbers("FIR No. 0123/2023")[0].value == "123/2023"
    assert grammars.find_fir_numbers(normalise("मु0अ0स0 123/23"))[0].value == "123/2023"
    assert grammars.find_fir_numbers("F1R No 77/2024")[0].value == "77/2024"


def test_sections_list_and_acts():
    keys = [h.value for h in grammars.find_sections("u/s 420, 467, 468 & 120-B IPC")]
    assert keys == ["IPC 420", "IPC 467", "IPC 468", "IPC 120B"]
    assert [h.value for h in grammars.find_sections("Section 318(4) BNS")] == ["BNS 318(4)"]
    assert [h.value for h in grammars.find_sections("धारा 302 भादवि")] == ["IPC 302"]
    assert [h.value for h in grammars.find_sections("u/s 138 N.I. Act")] == ["NI 138"]


def test_dates_validated_and_roled():
    text = "Date of occurrence: 14.03.2023, registered on 16 March 2023, next date 31/02/2024"
    hits = find_dates(text, today=dt.date(2026, 11, 17))
    assert [(h.date.isoformat(), h.role) for h in hits] == [("2023-03-14", "occurrence"),
                                                           ("2023-03-16", "registration")]


def test_label_requires_separator_for_party():
    vals = anchor_values("Name of Accused: Ramesh Kumar\nThe accused forged the deed")
    assert [(v.field, v.value) for v in vals] == [("opposing_party", "Ramesh Kumar")]


def test_doc_type():
    assert detect_doc_type("FIRST INFORMATION REPORT\nP.S. Kotwali")[0] == "fir"
    assert detect_doc_type("Khatauni\nKhasra No. 12")[0] == "revenue_record"


def test_engine_on_sample_fir(gaz, fuzzy, statutes):
    ex = Extractor(gaz, fuzzy, statutes)
    doc = Document("d1", normalise((SAMPLES / "fir_ghaziabad.txt").read_text(encoding="utf-8")))
    out = ex.extract(doc)
    got = {(c.field, c.value) for c in out.candidates}
    assert out.doc_type == "fir"
    assert ("district", "UP-GZB") in got
    assert ("police_station", "UP-GZB-PS-KOT") in got
    assert ("fir_number", "123/2023") in got
    assert ("opposing_party", "Ramesh Kumar") in got
    assert ("section", "IPC 420") in got
    assert ("date_occurrence", "2023-03-14") in got
    # every candidate carries its evidence span
    for c in out.candidates:
        assert c.text and c.span[1] > c.span[0] and c.rule


def test_section_without_act_uses_occurrence_date(gaz, fuzzy, statutes):
    ex = Extractor(gaz, fuzzy, statutes)
    doc = Document("d1", "Date of occurrence: 10/08/2024\nu/s 318(4)")
    keys = {c.value for c in ex.extract(doc).candidates if c.field == "section"}
    assert keys == {"BNS 318(4)"}
    assert statutes.case_type("BNS 318(4)") == ("criminal", "cheating_fraud")
