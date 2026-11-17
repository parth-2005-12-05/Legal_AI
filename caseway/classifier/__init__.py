"""Case-type classification in tiers, strongest evidence first.

    rules.py        Tier 1: statute section, document type, case-number prefix and
                    keyword rules with NegEx-style negation handling
    features.py     word uni/bigrams + character 3-5-grams on romanised text
    bm25.py         Tier 2a: BM25 retrieval used as a k-NN classifier ("similar past cases")
    naive_bayes.py  Tier 2b: multinomial Naive Bayes on the same features
    ensemble.py     combines the tiers into one CaseTypeResult (multi-label aware)

Tier 3 (a small local encoder) is optional in the design and deliberately not
built in v1; the ensemble keeps a slot for it.
"""

from .ensemble import CaseClassifier

__all__ = ["CaseClassifier"]
