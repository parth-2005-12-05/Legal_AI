"""Tests for the case-type classifier tiers."""

import pytest

from caseway.classifier import CaseClassifier
from caseway.classifier.rules import RuleClassifier


@pytest.fixture(scope="module")
def clf(settings, statutes):
    rules = RuleClassifier(settings.data_dir / "classifier" / "taxonomy.json", statutes)
    return CaseClassifier(rules, settings.data_dir / "classifier" / "seed_examples.json")


def test_sections_decide(clf):
    r = clf.classify("", ["IPC 420", "IPC 467"], ["fir"])
    assert (r.primary_top, r.primary_sub, r.tier) == ("criminal", "cheating_fraud", "rules")


def test_bns_maps_like_ipc(clf):
    r = clf.classify("", ["BNS 85"], [])
    assert (r.primary_top, r.primary_sub) == ("criminal", "dowry_cruelty")


def test_bail_need_is_promoted(clf):
    r = clf.classify("my brother was arrested, we need bail", ["IPC 420"], ["fir"])
    assert r.primary_sub == "bail"
    assert any(sub == "cheating_fraud" for _, sub, _ in r.labels)


@pytest.mark.parametrize("text,expected", [
    ("tenant has not paid rent for a year and refuses to vacate", "tenancy"),
    ("husband ne ghar se nikal diya, guzara bhatta chahiye", "maintenance"),
    ("the builder has not handed over my flat for 3 years", "builder_delay"),
    ("cheque given for repayment bounced, notice sent", "cheque_bounce"),
])
def test_free_text(clf, text, expected):
    assert clf.classify(text).primary_sub == expected


def test_negation_suppresses_keyword(settings, statutes):
    rules = RuleClassifier(settings.data_dir / "classifier" / "taxonomy.json", statutes)
    subs = {v.sub for v in rules.votes([], [], [], "this is not a theft matter")}
    assert "theft" not in subs
    subs = {v.sub for v in rules.votes([], [], [], "there was a theft at my shop")}
    assert "theft" in subs
