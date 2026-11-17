"""BM25 retrieval as a k-NN classifier (design: Tier 2).

Index past verified cases (at cold start: the seed examples). For a new
description, take the top-k most similar documents and vote by type, weighted
by BM25 score. Fully explainable: the result names the similar past cases.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avgdl = sum(self.lengths) / max(len(docs), 1)
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        q = Counter(query)
        out = []
        for d, dl in zip(self.docs, self.lengths):
            s = 0.0
            for t in q:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class BM25KNN:
    def __init__(self, docs: list[list[str]], labels: list[str], texts: list[str], k: int = 20):
        self.index, self.labels, self.texts, self.k = BM25(docs), labels, texts, k

    def predict(self, query: list[str]) -> tuple[dict[str, float], list[tuple[str, str, float]]]:
        """Return (label -> probability, top similar examples as (text, label, score))."""
        scored = sorted(enumerate(self.index.scores(query)), key=lambda x: -x[1])[: self.k]
        votes: dict[str, float] = defaultdict(float)
        for i, s in scored:
            if s > 0:
                votes[self.labels[i]] += s
        total = sum(votes.values())
        probs = {k: v / total for k, v in votes.items()} if total else {}
        similar = [(self.texts[i], self.labels[i], round(s, 3)) for i, s in scored[:5] if s > 0]
        return probs, similar
