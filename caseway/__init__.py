"""CaseWay: offline legal case-to-lawyer matching system (first version).

Package layout (mirrors the design document's services):

    caseway/config.py        settings.yaml loader (partner_review switch lives here)
    caseway/models.py        shared dataclasses: Candidate, FieldResult, CaseRecord ...
    caseway/normalise.py     Unicode NFC, Devanagari digits, romanisation, OCR fixes
    caseway/ingest/          sanitiser, born-digital text layer, OCR adapters + ROVER vote
    caseway/extraction/      label anchoring, regex grammars, dates, doc-type, statutes
    caseway/gazetteer/       jurisdiction gazetteer, Aho-Corasick, trie, fuzzy matcher,
                             tree-DP jurisdiction resolver
    caseway/classifier/      Tier-1 rules + Tier-2 BM25-kNN / Naive Bayes ensemble
    caseway/fusion.py        evidence fusion, consistency checks, confidence routing
    caseway/matching/        eligibility filter, weighted score, greedy + min-cost flow,
                             routing (customer / partner queue / senior review)
    caseway/security/        envelope encryption, sealed uploads, hash-chained audit log
    caseway/store.py         SQLite persistence (stand-in for PostgreSQL in v1)
    caseway/chat.py          scripted follow-up questions with gazetteer choices
    caseway/pipeline.py      orchestrates one case end to end
    caseway/api/             FastAPI app (intake, review console, partner queue)
    caseway/cli.py           command line entry point (demo, audit verify)

Nothing in this package makes a network call or uses an external AI service.
"""

__version__ = "0.1.0"
