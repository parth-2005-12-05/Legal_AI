"""Ensemble case-type classifier.

Combines the tiers into one result:

* If a hard rule fired (statute section or case-number prefix), the rules decide
  ("Tier 1 rules override when they fire"); the statistical models only add
  secondary labels.
* Otherwise the BM25-kNN and Naive Bayes distributions are averaged, nudged by
  keyword rules and by the document-type prior.
* Multi-label: every sub-type above the threshold is returned (a dowry case can
  be criminal 498A and family maintenance at once); the primary label is the
  highest, except that a live bail need is always promoted to primary because
  it is the urgent service the customer needs now.

Probabilities are not yet calibrated (isotonic regression needs verified cases);
until then thresholds should be treated as conservative.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..models import CaseTypeResult
from .bm25 import BM25KNN
from .features import featurise
from .naive_bayes import MultinomialNB
from .rules import RuleClassifier

BAIL_SUBTYPES = ("bail", "anticipatory_bail")


class CaseClassifier:
    def __init__(self, rules: RuleClassifier, seed_path: str | Path, threshold: float = 0.3, k: int = 20):
        self.rules, self.threshold = rules, threshold
        with open(seed_path, encoding="utf-8") as fh:
            examples = json.load(fh)["examples"]
        self.train(examples, k)

    def train(self, examples: list[list[str]], k: int = 20) -> None:
        """(Re)train Tier 2 from [top, sub, text] rows. Called monthly with verified cases."""
        texts = [e[2] for e in examples]
        labels = [f"{e[0]}/{e[1]}" for e in examples]
        feats = [featurise(t) for t in texts]
        self.knn = BM25KNN(feats, labels, texts, k=k)
        self.nb = MultinomialNB().fit(feats, labels)

    def classify(self, text: str, sections: list[str] | None = None, doc_types: list[str] | None = None,
                 case_prefixes: list[str] | None = None) -> CaseTypeResult:
        votes = self.rules.votes(sections or [], doc_types or [], case_prefixes or [], text)
        feats = featurise(text)
        knn_p, similar = self.knn.predict(feats) if text.strip() else ({}, [])
        nb_p = self.nb.predict_proba(feats) if text.strip() else {}
        stat: dict[str, float] = defaultdict(float)
        for dist in (knn_p, nb_p):
            for label, p in dist.items():
                stat[label] += p / 2

        hard = [v for v in votes if v.kind in ("section", "prefix")]
        rule_scores: dict[str, float] = defaultdict(float)
        top_prior: dict[str, float] = defaultdict(float)
        for v in votes:
            if v.sub:
                rule_scores[f"{v.top}/{v.sub}"] += v.weight
            else:
                top_prior[v.top] += v.weight

        scores: dict[str, float] = defaultdict(float)
        if hard:
            tier = "rules"
            m = max(rule_scores.values(), default=0) or 1.0
            for label, s in rule_scores.items():
                scores[label] = 0.95 * s / m
            for label, p in stat.items():
                scores[label] = max(scores[label], 0.5 * p)
        else:
            tier = "ensemble" if votes else ("statistical" if stat else "none")
            m = max(rule_scores.values(), default=0) or 1.0
            for label in set(stat) | set(rule_scores):
                kw = rule_scores.get(label, 0) / m
                scores[label] = (0.6 * stat.get(label, 0) + 0.4 * kw) if rule_scores else stat.get(label, 0)
        # document-type / prefix top-level priors (e.g. an FIR favours criminal)
        for label in list(scores):
            top = label.split("/")[0]
            if top in top_prior:
                scores[label] = min(1.0, scores[label] * (1 + 0.5 * min(top_prior[top], 1.0)))
        if not scores and top_prior:
            top = max(top_prior, key=top_prior.get)
            return CaseTypeResult(top, None, 0.6, [(top, "", 0.6)], [v.evidence for v in votes], "rules")
        if not scores:
            return CaseTypeResult(None, None, 0.0, [], [], "none")

        # normalise into a distribution-like range and pick labels
        total = sum(scores.values())
        if tier != "rules" and total > 1.0:
            scores = {k: v / total for k, v in scores.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        labels = [(l.split("/")[0], l.split("/")[1], round(p, 3)) for l, p in ranked if p >= self.threshold]
        if not labels:
            l, p = ranked[0]
            labels = [(l.split("/")[0], l.split("/")[1], round(p, 3))]

        primary = labels[0]
        bail_votes = [v for v in votes if v.sub in BAIL_SUBTYPES]
        if bail_votes and any(t == "criminal" for t, _, _ in labels):
            sub = "anticipatory_bail" if any(v.sub == "anticipatory_bail" for v in bail_votes) else "bail"
            primary = ("criminal", sub, max(primary[2], scores.get(f"criminal/{sub}", 0)))
            labels = [primary] + [l for l in labels if l[1] != sub]

        evidence = [v.evidence for v in votes]
        evidence += [f"similar to: '{t}' ({lab})" for t, lab, _ in similar[:3]]
        return CaseTypeResult(primary[0], primary[1], round(primary[2], 3), labels, evidence, tier)
