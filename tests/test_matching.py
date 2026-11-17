"""Tests for eligibility, scoring, assignment algorithms and the partner_review router."""

import itertools
from dataclasses import replace

import pytest

from caseway.config import load_settings
from caseway.matching import CaseFacts, MatchingEngine, load_lawyers
from caseway.matching.assignment import batch_assign, greedy_rank
from caseway.matching.router import CUSTOMER, PARTNER_QUEUE, SENIOR_REVIEW, decide_route
from caseway.models import LawyerMatch


@pytest.fixture(scope="module")
def lawyers(settings):
    return load_lawyers(settings.data_dir / "lawyers.json")


def engine(gaz, lawyers, **matching):
    s = load_settings(overrides={"matching": matching} if matching else None)
    from caseway.classifier.rules import RuleClassifier
    from caseway.extraction.statutes import StatuteTable
    rules = RuleClassifier(s.data_dir / "classifier" / "taxonomy.json",
                           StatuteTable(s.data_dir / "classifier" / "statutes.json"))
    return MatchingEngine(gaz, lawyers, s.matching, rules.all_subtypes())


def facts(**kw):
    base = dict(case_id="c1", state="UP", district="UP-GZB", court="UP-GZB-C-DCDRC", top="consumer",
                sub="builder_delay", customer_language="en", fee_band=2)
    base.update(kw)
    return CaseFacts(**base)


def test_eligibility_rules(gaz, lawyers):
    e = engine(gaz, lawyers)
    res = e.filter.check(lawyers, facts(top="criminal", sub="cheating_fraud", court="UP-GZB-C-CJM",
                                        opposing_parties=["Ramesh Kumaar"]))
    assert "on leave" in res.rejected["L010"]
    assert any("conflict" in r for r in res.rejected["L002"])       # fuzzy name hit
    assert any("does not practise" in r for r in res.rejected["L008"])   # Meerut only
    assert {c["lawyer_id"] for c in res.conflicts} == {"L002", "L006"}


def test_urgent_bail_only_urgent_lawyers(gaz, lawyers):
    e = engine(gaz, lawyers)
    res = e.filter.check(lawyers, facts(top="criminal", sub="bail", court="UP-GZB-C-DSJ"))
    assert {l.id for l in res.eligible} == {"L001", "L002"}   # urgent-flagged and available <= 24h
    assert any("urgent" in r for r in res.rejected["L003"])


def test_scores_have_breakdown_and_reasons(gaz, lawyers):
    d = engine(gaz, lawyers).match(facts())
    assert d.matches[0].lawyer_id == "L007"
    m = d.matches[0]
    assert set(m.breakdown) >= {"spec", "court", "exp", "lang", "prox", "fee", "load"}
    assert "practises in" in m.reasons[0] and "->" in m.reasons[0]
    assert "win" not in str(m.public_profile).lower()


def test_greedy_skips_full_lawyers():
    ms = [LawyerMatch(i, i, s, {}, [], {}) for i, s in (("a", 0.9), ("b", 0.8), ("c", 0.7))]
    assert [m.lawyer_id for m in greedy_rank(ms, {"a": 0, "b": 1, "c": 1}, 3)] == ["b", "c"]


def test_min_cost_flow_matches_brute_force():
    scores = {"c1": {"A": 0.9, "B": 0.8}, "c2": {"A": 0.85, "B": 0.2}, "c3": {"A": 0.5, "B": 0.6, "C": 0.4}}
    cap = {"A": 1, "B": 1, "C": 1}
    got = batch_assign(scores, cap, load_step=0.0)
    best, best_total = None, -1
    for perm in itertools.permutations(["A", "B", "C"]):
        pairs = dict(zip(scores, perm))
        if all(l in scores[c] for c, l in pairs.items()):
            total = sum(scores[c][l] for c, l in pairs.items())
            if total > best_total:
                best, best_total = pairs, total
    assert got == best   # greedy would give A to c1 and strand c2 with B (0.2)


def test_min_cost_flow_load_balances():
    scores = {f"c{i}": {"A": 0.9, "B": 0.85} for i in range(4)}
    got = batch_assign(scores, {"A": 4, "B": 4}, load_step=0.05)
    assert sorted(got.values()).count("A") == 2


def _m(score, lid="x"):
    return LawyerMatch(lid, lid, score, {}, [], {})


def test_router_partner_switch(settings):
    off = replace(settings.matching, partner_review=False)
    on = replace(settings.matching, partner_review=True)
    ms = [_m(0.8, "a"), _m(0.6, "b")]
    assert decide_route(ms, off, conflict=False, urgent=False)[0] == CUSTOMER
    assert decide_route(ms, on, conflict=False, urgent=False)[0] == PARTNER_QUEUE


@pytest.mark.parametrize("kw,ms,flag", [
    ({"conflict": True, "urgent": False}, [_m(0.8, "a"), _m(0.6, "b")], "conflict"),
    ({"conflict": False, "urgent": True}, [_m(0.8, "a"), _m(0.6, "b")], "urgent"),
    ({"conflict": False, "urgent": False}, [_m(0.3, "a")], "low_score"),
    ({"conflict": False, "urgent": False}, [_m(0.70, "a"), _m(0.69, "b")], "close_call"),
    ({"conflict": False, "urgent": False}, [], "no_match"),
])
@pytest.mark.parametrize("partner", [False, True])
def test_risky_cases_always_go_to_senior_review(settings, kw, ms, flag, partner):
    s = replace(settings.matching, partner_review=partner)
    route, _, reviewers, flags = decide_route(ms, s, **kw)
    assert route == SENIOR_REVIEW and flag in flags
    assert reviewers == ["senior_lawyer", "admin"]
