# CaseWay: offline case-to-lawyer matching (v0.1)

First working version of the system described in *Offline Case-to-Lawyer Matching System — Design*.
A case arrives as a customer chat plus documents (FIR, court order, revenue record ...). CaseWay
reads where and when the case is from, what type it is, and suggests the right lawyers, with
the evidence behind every field and every match. Everything runs on your own machine: no cloud
service, no external AI, no outbound network calls.

## Quick start (Windows)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest              # 64 tests
.venv\Scripts\python -m caseway demo        # three sample cases end to end
.venv\Scripts\python -m uvicorn caseway.api.app:app --port 8000   # API, docs at /docs
```

Runtime data (SQLite DB, encrypted documents, development keys, audit log) is written to `var/`,
which git ignores.

## The partner review switch

`config/settings.yaml`:

```yaml
matching:
  partner_review: false          # true = every match waits for a partner first
  conflict_reviewers: [senior_lawyer, admin]
```

* **Off (default):** the customer sees the top 3 lawyers straight away and their pick assigns the case.
* **On:** every match waits in the partner queue; the partner accepts or overrides with a reason,
  then the customer sees the list.
* **Always, whatever the switch says:** conflict-of-interest hits, urgent bail matters, low scores,
  close calls and "no eligible lawyer" stop for a named senior lawyer or admin.
* Switching is that one setting plus a restart. The partner code paths always exist; with the
  switch off the API returns 404 for `/partner/*`, so the role stays hidden.

## How a case flows

```
chat + uploads ──► sealed with public key (intake can't decrypt) ──► job queue
   ──► unseal + sanitise + envelope-encrypt ──► text layer / OCR ──► normalise
   ──► extraction (labels, regex grammars, dates, Aho-Corasick scan, fuzzy match)
   ──► tree-DP jurisdiction resolver ──► case-type classifier ──► evidence fusion
   ──► low confidence? → paralegal review (with scripted follow-up questions)
   ──► eligibility filter → weighted score → greedy / min-cost-flow assignment
   ──► router: senior review | partner queue (switch on) | customer sees top 3
   ──► customer pick → assigned; every step in the hash-chained audit log
```

## Code map

| Path | What it does | Design section |
|---|---|---|
| `config/settings.yaml`, `caseway/config.py` | All thresholds, weights and the partner_review switch | Partner review switch |
| `caseway/normalise.py` | NFC, Devanagari digits, OCR look-alikes, romanisation, phonetic key | OCR step 7 |
| `caseway/ingest/` | Magic-byte sanitiser, born-digital PDF text, OCR adapter + ROVER vote | Document ingestion |
| `caseway/extraction/` | Label anchoring, CNR / case / FIR / section / land-id grammars, dates, doc type, IPC↔BNS | Entity extraction, layers 1-2 |
| `caseway/gazetteer/` | Gazetteer model, Aho-Corasick, prefix trie, fuzzy matcher (trigram + BK-tree + phonetic + Jaro-Winkler + OCR-weighted edit distance), tree-DP resolver | Layers 3-5, Jurisdiction gazetteer |
| `caseway/classifier/` | Statute / doc-type / prefix / keyword rules with negation, BM25-kNN, Naive Bayes, ensemble | Case-type classification |
| `caseway/fusion.py` | Noisy-OR evidence fusion, consistency checks, auto / provisional / review | Accuracy strategy |
| `caseway/matching/` | Eligibility filter, weighted score with reasons, greedy + min-cost flow, router | Lawyer matching |
| `caseway/security/` | KEK → case key → DEK envelope encryption, crypto-shredding, X25519 sealed uploads, audit hash chain | Security design |
| `caseway/store.py` | SQLite store (sensitive columns encrypted), review queue, feedback | Data zone |
| `caseway/chat.py` | Scripted follow-up questions with gazetteer choices | Accuracy strategy 3 |
| `caseway/pipeline.py` | Orchestrates one case; status machine; review / partner / customer actions | End-to-end flow |
| `caseway/api/app.py`, `caseway/cli.py` | HTTP API and command line | — |
| `data/` | SAMPLE gazetteer, lawyer roster, taxonomy, statute tables, seed examples, sample FIR | — |
| `tests/` | 64 tests covering every module and the end-to-end flows | — |

## What is done vs. what remains

This v0.1 covers the core logic of Phases 1-3 so the whole flow can be exercised on one machine.
The 16-week plan's infrastructure, data and hardening work remains.

**Done**
* Full pipeline from chat + upload to an assigned lawyer, with the case status machine.
* partner_review switch (default off), senior-review safety net, partner queue hidden while off.
* Extraction layers 1-5, Hindi + English, with spans / rule ids / scores on every candidate.
* Tree-DP jurisdiction resolver with hierarchy checks; court derived from a territorial decision table.
* Tier 1 rules + Tier 2 BM25-kNN / Naive Bayes ensemble, multi-label, bail promoted as urgent.
* Eligibility (Bar Council, court, conflict of interest, leave, capacity, language, urgency),
  the design's weighted score with plain-language reasons, priority-queue greedy and min-cost flow.
* Envelope encryption, crypto-shredding, sealed uploads, hash-chained audit log without personal data.
* Learning signals captured (customer picks, partner / senior decisions, field corrections).

**Remaining (by phase)**
* *Phase 0 – data & infra:* import real LGD / Census, pincode and e-Courts establishment data
  (the bundled gazetteer and codes are illustrative only); label the 500-1,000 case gold set;
  network zones, deny-all egress, Vault + HSM/TPM (replacing `LocalKeyManager`), internal CA,
  offline package mirror; PostgreSQL with row-level security + MinIO (replacing SQLite / files).
* *Phase 1 – intake & OCR:* OpenCV preprocessing, layout and script detection, PaddleOCR as the
  second engine (the ROVER vote is ready), qpdf / mutool flattening, ClamAV wiring, Redis / RabbitMQ
  job queue with separate DMZ and processing services, CER measurement on the gold set.
* *Phase 2 – extraction:* full bilingual label table, more grammars (revenue court ids, BSA), learned
  OCR confusion matrix, per-field thresholds tuned from precision-recall curves, legal-team sign-off on
  the IPC↔BNS and territorial tables (both marked DRAFT).
* *Phase 3 – classification & matching:* ~50 seed examples per sub-type (3 today), isotonic calibration,
  Complement NB / logistic regression, real lawyer roster and court-distance table, back-test against
  past assignments, learning-to-rank on the captured feedback after 6 months.
* *Phase 4 – console & hardening:* review console UI (page image + highlighted span), customer web app,
  customer chat answering endpoint for follow-up questions, staff SSO + MFA, customer OTP / passkeys,
  RBAC + ABAC, retention jobs, backups, penetration test. **The v1 API has no authentication; do not expose it.**
* *Phase 5 – pilot:* shadow mode, accuracy reports, monthly retraining.

## Decisions still needed from the team (from the design)
What "offline" covers for customers, customer login method, which states and languages first,
who maintains the lawyer roster, retention periods, and whether the optional local model (Tier 3)
stays out of scope for launch.
