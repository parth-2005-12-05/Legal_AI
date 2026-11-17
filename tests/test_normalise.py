"""Tests for caseway/normalise.py: digits, OCR look-alikes, romanisation, phonetic keys."""

from caseway.normalise import fold, normalise, phonetic_key, romanise


def test_devanagari_digits_become_ascii():
    assert normalise("FIR सं. ०१२३/२०२३") == "FIR सं. 0123/2023"


def test_lookalikes_fixed_only_between_digits():
    assert normalise("Year 2O23") == "Year 2023"
    assert normalise("u/s 498A IPC and 120-B") == "u/s 498A IPC and 120-B"


def test_whitespace_collapsed_but_lines_kept():
    assert normalise("a   b\n\n c") == "a b\n\nc"


def test_romanise_with_schwa_deletion():
    assert romanise("कोतवाली") == "kotvaalii"
    assert romanise("थाना") == "thaanaa"


def test_phonetic_key_merges_variants():
    assert phonetic_key("Ghaziabad") == phonetic_key("Gaziabad") == phonetic_key("Ghaziyabad")
    assert phonetic_key("Kotwali") == phonetic_key("Kotwaali") == phonetic_key("कोतवाली")


def test_fold():
    assert fold("  Kotwali  Nagar. ") == "kotwali nagar"
