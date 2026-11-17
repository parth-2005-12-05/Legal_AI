"""Multinomial Naive Bayes over the shared features (design: Tier 2).

Small, fast, and works with very little data, which is why it is the cold-start
workhorse. Complement NB / logistic regression and isotonic calibration (so a
0.9 really means ~90%) are the planned upgrades once verified cases accumulate;
they need a few hundred labelled examples to be meaningful.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class MultinomialNB:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.class_counts: Counter = Counter()
        self.feature_counts: dict[str, Counter] = defaultdict(Counter)
        self.totals: Counter = Counter()
        self.vocab: set[str] = set()

    def fit(self, docs: list[list[str]], labels: list[str]) -> "MultinomialNB":
        for feats, label in zip(docs, labels):
            self.class_counts[label] += 1
            self.feature_counts[label].update(feats)
            self.totals[label] += len(feats)
            self.vocab.update(feats)
        return self

    def predict_proba(self, feats: list[str], temperature: float = 0.1) -> dict[str, float]:
        """Softmax over per-class log-likelihoods. Naive Bayes is notoriously
        over-confident; the temperature flattens it until calibration exists.
        Features never seen in training are ignored (they carry no signal)."""
        known = [f for f in feats if f in self.vocab]
        if not known:
            return {}
        n_docs = sum(self.class_counts.values())
        v = len(self.vocab)
        logs = {}
        for c in self.class_counts:
            lp = math.log(self.class_counts[c] / n_docs)
            denom = self.totals[c] + self.alpha * v
            counts = self.feature_counts[c]
            lp += sum(math.log((counts[f] + self.alpha) / denom) for f in known)
            logs[c] = lp / max(len(known), 1) / temperature
        m = max(logs.values())
        exp = {c: math.exp(lp - m) for c, lp in logs.items()}
        z = sum(exp.values())
        return {c: e / z for c, e in exp.items()}
