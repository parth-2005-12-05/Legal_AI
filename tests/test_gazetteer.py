"""Tests for the gazetteer: model, Aho-Corasick, trie, fuzzy matcher and the
tree-DP jurisdiction resolver."""

from caseway.gazetteer.ahocorasick import AhoCorasick
from caseway.gazetteer.fuzzy import BKTree, damerau_levenshtein, jaro_winkler, weighted_edit_distance
from caseway.gazetteer.resolver import JurisdictionResolver
from caseway.gazetteer.trie import PrefixTrie
from caseway.models import Candidate


def cand(field, value, source="gazetteer_scan", score=1.0, start=0):
    return Candidate(field, value, value, "d", 1, (start, start + 1), "test", score, source)


def test_materialised_paths(gaz):
    assert gaz.get("UP-GZB-PS-KOT").path == "/UP/UP-GZB/UP-GZB-PS-KOT"
    assert gaz.is_within("UP-GZB-T-MDN-V-SKR", "UP-GZB")
    assert not gaz.is_within("UP-MRT-PS-KOT", "UP-GZB")
    assert gaz.pincode_to_district["201002"] == "UP-GZB"
    assert gaz.cnr_to_district["UPGZ"] == "UP-GZB"


def test_lookup_exact_cross_script(gaz):
    assert set(gaz.lookup_exact("Kotwali", "police_station")) == {"UP-GZB-PS-KOT", "UP-MRT-PS-KOT"}
    assert gaz.lookup_exact("गाजियाबाद", "district") == ["UP-GZB"]


def test_aho_corasick_longest_and_word_boundaries():
    ac = AhoCorasick()
    for key in ("sadar", "sadar bazar", "bazar"):
        ac.add(key, key)
    found = [m.key for m in ac.find_longest("FIR at Sadar Bazar police station")]
    assert found == ["sadar bazar"]
    assert ac.find_all("sadarbazar") == []          # not on a word boundary


def test_aho_corasick_devanagari_boundary():
    ac = AhoCorasick()
    ac.add("कोतवाल", 1)
    assert ac.find_all("थाना कोतवाली") == []        # matra continues the word


def test_trie_constrained_by_parent(gaz):
    trie = PrefixTrie(gaz, "police_station")
    assert set(trie.search("kot")) >= {"UP-GZB-PS-KOT", "UP-MRT-PS-KOT"}
    assert trie.search("kot", within="UP-MRT") == ["UP-MRT-PS-KOT"]


def test_string_metrics():
    assert damerau_levenshtein("kotwali", "kotwlai") == 1
    assert jaro_winkler("ghaziabad", "gaziabad") > 0.9
    assert weighted_edit_distance("rnodinagar", "modinagar") < 0.5   # rn <-> m is cheap
    tree = BKTree()
    for w in ("meerut", "merut", "lucknow"):
        tree.add(w)
    assert {w for w, _ in tree.search("meerat", 1)} == {"meerut"}


def test_fuzzy_accepts_only_with_clear_gap(fuzzy):
    # two "Kotwali" stations: without context the top two tie, so nothing is accepted
    res = fuzzy.match("Kotwaali", "police_station")
    assert res.accepted is None and len(res.candidates) >= 2
    # with the district known, the Ghaziabad one wins clearly
    res = fuzzy.match("Kotwaali", "police_station", within="UP-GZB")
    assert res.accepted and res.accepted.entity_id == "UP-GZB-PS-KOT"


def test_resolver_picks_consistent_hierarchy(gaz, settings):
    r = JurisdictionResolver(gaz, settings.fusion.source_weights)
    res = r.resolve([
        cand("police_station", "UP-GZB-PS-KOT", "doc_header_label"),
        cand("police_station", "UP-MRT-PS-KOT", "doc_header_label"),   # homonym
        cand("district", "UP-GZB", "inferred", 0.7),                     # pincode
        cand("court", "UP-GZB-C-CJM", "doc_label"),
    ])
    assert res.chosen["district"] == "UP-GZB"
    assert res.chosen["police_station"] == "UP-GZB-PS-KOT"
    assert res.chosen["state"] == "UP"
    assert res.margin > 0.3


def test_resolver_reports_conflict(gaz, settings):
    r = JurisdictionResolver(gaz, settings.fusion.source_weights)
    res = r.resolve([cand("district", "UP-GZB", "doc_header_label"),
                     cand("court", "UP-MRT-C-CJM", "doc_header_label")])
    # equally strong evidence for two districts: reported, and the margin is tiny
    assert res.issues
    assert res.margin < 0.2
