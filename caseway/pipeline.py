"""CaseWay application service: one case end to end.

This is the orchestration layer the API and CLI call. It keeps the design's
zone boundaries visible in code even though v1 runs in a single process:

  DMZ side        create_case(), upload()        - uses ONLY the sealing public key
  Processing side process()                      - unseals, sanitises, extracts,
                                                   resolves, classifies, fuses, matches
  Admin side      submit_field_review(), senior_decision(), partner_decision()
  Customer side   customer_view(), customer_pick()

Case status machine
    intake -> paralegal_review (low-confidence fields)
           -> senior_review   (conflict / urgent bail / low score / close call)
           -> partner_queue   (only when matching.partner_review is true)
           -> customer_choice -> assigned
    any    -> shredded        (crypto-shred on retention expiry or DPDP request)

Every step writes an audit entry (ids and reason codes only).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from .chat import follow_up_questions
from .classifier import CaseClassifier
from .classifier.rules import RuleClassifier
from .config import Settings, load_settings
from .extraction import Extractor
from .extraction.statutes import StatuteTable
from .fusion import JURISDICTION_FIELDS, Fusion
from .gazetteer import load_gazetteer
from .gazetteer.fuzzy import FuzzyMatcher
from .gazetteer.resolver import JurisdictionResolver
from .gazetteer.trie import PrefixTrie
from .ingest import ocr as ocr_mod
from .ingest.sanitiser import RejectedFile, sanitise
from .ingest.text_layer import pdf_page_count, pdf_pages_text
from .matching import CaseFacts, MatchingEngine, load_lawyers
from .matching.router import CUSTOMER, PARTNER_QUEUE, SENIOR_REVIEW, next_route_after_senior_review
from .models import CaseTypeResult, Document, FieldResult, MatchDecision
from .normalise import normalise
from .security import envelope
from .security.audit import AuditLog
from .security.keys import LocalKeyManager
from .security.sealed import load_or_create_keypair, seal, unseal
from .store import Store, new_id

STATUS_FOR_ROUTE = {CUSTOMER: "customer_choice", PARTNER_QUEUE: "partner_queue", SENIOR_REVIEW: "senior_review"}


class WorkflowError(RuntimeError):
    """An action is not allowed in the case's current state."""


class PartnerReviewDisabled(WorkflowError):
    """Partner actions while matching.partner_review is false."""


class CaseWay:
    def __init__(self, settings: Settings | None = None):
        self.settings = s = settings or load_settings()
        var = s.var_dir
        for sub in ("keys", "objects", "quarantine"):
            (var / sub).mkdir(parents=True, exist_ok=True)

        # data zone
        self.store = Store(var / "caseway.sqlite3")
        self.keys = LocalKeyManager(var / "keys")
        self.audit = AuditLog(var / "audit.log.jsonl")
        self._processing_private, self.dmz_public = load_or_create_keypair(var / "keys")

        # reference data + algorithms
        data = s.data_dir
        self.gaz = load_gazetteer(data / "gazetteer" / "sample_gazetteer.json")
        self.statutes = StatuteTable(data / "classifier" / "statutes.json")
        self.fuzzy = FuzzyMatcher(self.gaz, s.fuzzy.accept_score, s.fuzzy.accept_gap,
                                  s.fuzzy.trigram_min_overlap, s.fuzzy.bk_max_distance)
        self.extractor = Extractor(self.gaz, self.fuzzy, self.statutes)
        self.resolver = JurisdictionResolver(self.gaz, s.fusion.source_weights)
        self.rules = RuleClassifier(data / "classifier" / "taxonomy.json", self.statutes)
        self.classifier = CaseClassifier(self.rules, data / "classifier" / "seed_examples.json",
                                         s.classifier.multi_label_threshold, s.classifier.bm25_k)
        self.fusion = Fusion(self.gaz, s.fusion).attach_statutes(self.statutes)
        self.lawyers = load_lawyers(data / "lawyers.json")
        self.matcher = MatchingEngine(self.gaz, self.lawyers, s.matching, self.rules.all_subtypes())
        self.tries = {t: PrefixTrie(self.gaz, t) for t in JURISDICTION_FIELDS}

    @property
    def partner_review(self) -> bool:
        return self.settings.matching.partner_review

    # ------------------------------------------------------------------ DMZ side
    def create_case(self, chat_text: str, customer_language: str = "hi", fee_band: int | None = None) -> str:
        case_id = new_id("case")
        self.keys.create_case_key(case_id)
        blob = envelope.encrypt(self.keys, case_id, "chat", (chat_text or "").encode("utf-8"))
        self.store.create_case(case_id, customer_language, fee_band, blob)
        self.audit.append("customer", "case.created", case_id)
        return case_id

    def upload(self, case_id: str, filename: str, data: bytes) -> str:
        """Seal the upload with the processing zone's PUBLIC key and drop it in
        quarantine. Only the job id crosses into processing."""
        self._require_case(case_id)
        path = self.settings.var_dir / "quarantine" / f"{new_id('q')}.sealed"
        path.write_bytes(seal(self.dmz_public, data))
        job_id = self.store.add_job(case_id, str(path), re.sub(r"[^\w.\-]", "_", filename)[:80])
        self.audit.append("customer", "document.uploaded", case_id, job_id=job_id, size=len(data))
        return job_id

    # ---------------------------------------------------------- processing side
    def _require_case(self, case_id: str):
        row = self.store.case(case_id)
        if row is None:
            raise KeyError(case_id)
        if row["status"] == "shredded":
            raise WorkflowError("case has been deleted")
        return row

    def _run_jobs(self, case_id: str) -> list[str]:
        """Unseal each pending upload, sanitise it and store it envelope-encrypted.
        Returns problems found (rejected files) for the reviewer."""
        problems = []
        for job in self.store.pending_jobs(case_id):
            sealed_path = Path(job["sealed_path"])
            try:
                data = unseal(self._processing_private, sealed_path.read_bytes())
                info = sanitise(data, self.settings.ingest.max_file_mb, self.settings.ingest.allow_plain_text)
            except RejectedFile as exc:
                self.store.finish_job(job["id"], "rejected", str(exc))
                self.audit.append("system", "document.rejected", case_id, job_id=job["id"], reason=str(exc))
                problems.append(f"upload rejected: {exc}")
                sealed_path.unlink(missing_ok=True)
                continue
            doc_id = new_id("doc")
            obj = self.settings.var_dir / "objects" / case_id
            obj.mkdir(parents=True, exist_ok=True)
            (obj / f"{doc_id}.bin").write_bytes(envelope.encrypt(self.keys, case_id, doc_id, data))
            sealed_path.unlink(missing_ok=True)
            n_pages = pdf_page_count(data) if info.mime == "application/pdf" else 1
            self.store.add_document(doc_id, case_id, info.mime, str(obj / f"{doc_id}.bin"), n_pages, info.flags)
            self.store.finish_job(job["id"], "done")
            self.audit.append("system", "document.stored", case_id, doc_id=doc_id, mime=info.mime,
                              pages=n_pages, av_scan=info.av_scan, flags=info.flags)
        return problems

    def _all_docs(self, case_id: str) -> tuple[list[Document], list[str]]:
        """Decrypt every stored document of the case in memory (for this job only)
        and turn it into page Documents: text layer first, OCR for images."""
        docs, problems = [], []
        for row in self.store.documents(case_id):
            raw = envelope.decrypt(self.keys, case_id, row["id"], Path(row["object_path"]).read_bytes())
            if row["mime"] == "text/plain":
                docs.append(Document(row["id"], normalise(raw.decode("utf-8")), 1, "born_digital"))
            elif row["mime"] == "application/pdf":
                for i, text in enumerate(pdf_pages_text(raw), start=1):
                    if text is not None:
                        docs.append(Document(f"{row['id']}#p{i}", normalise(text), i, "born_digital"))
                    else:
                        problems.append(f"{row['id']} page {i} has no text layer and needs OCR")
            else:
                try:
                    text, conf = ocr_mod.ocr_image(raw)
                    docs.append(Document(row["id"], normalise(text), 1, "ocr_high_conf", ocr_confidence=conf))
                except ocr_mod.OCRUnavailable as exc:
                    problems.append(f"{row['id']}: {exc}")
        return docs, problems

    def process(self, case_id: str) -> dict:
        """Run the full pipeline for a case and route it. Returns a summary."""
        row = self._require_case(case_id)
        problems = self._run_jobs(case_id)
        docs, more = self._all_docs(case_id)
        problems += more
        chat = envelope.decrypt(self.keys, case_id, "chat", row["chat_blob"]).decode("utf-8")
        if chat.strip():
            docs.append(Document("chat", normalise(chat), 1, "chat", doc_type="chat"))
        self.audit.append("system", "case.decrypted_for_processing", case_id, documents=len(docs))

        candidates, doc_types, prefixes = [], [], []
        for d in docs:
            ex = self.extractor.extract(d)
            candidates += ex.candidates
            doc_types.append(ex.doc_type)
            prefixes += [c.extra.get("prefix") for c in ex.candidates if c.field == "case_number"]
        resolution = self.resolver.resolve([c for c in candidates if c.field in JURISDICTION_FIELDS])
        sections = sorted({c.value for c in candidates if c.field == "section"})
        all_text = "\n".join(d.text for d in docs)
        case_type = self.classifier.classify(all_text, sections, doc_types, [p for p in prefixes if p])
        fused = self.fusion.fuse(candidates, resolution, case_type)
        if problems:
            fused.issues += problems
            if not any(d.source != "chat" for d in docs) and self.store.documents(case_id):
                fused.needs_review.append("documents")

        result = {
            "fields": {k: v.to_dict() for k, v in fused.fields.items()},
            "sections": fused.sections,
            "issues": fused.issues,
            "needs_review": fused.needs_review,
            "case_type": asdict(case_type),
            "opposing_parties": sorted({c.value for c in candidates if c.field == "opposing_party"}),
            "document_language": "hi" if len(re.findall(r"[ऀ-ॿ]", all_text)) > len(all_text) * 0.2 else "en",
            "documents": [{"doc_id": d.doc_id, "doc_type": d.doc_type, "source": d.source} for d in docs],
            "resolution": {"margin": resolution.margin, "district_totals": resolution.district_totals},
        }
        result["follow_up_questions"] = follow_up_questions(result["fields"], self.gaz, list(self.rules.types),
                                                            row["customer_language"])
        self._save_result(case_id, result)
        self.audit.append("system", "case.extracted", case_id, needs_review=fused.needs_review,
                          auto=fused.auto, tier=case_type.tier)

        if fused.needs_review:
            self.store.update_case(case_id, status="paralegal_review")
            self.store.enqueue(case_id, "fields", ["paralegal"], [f"check {f}" for f in fused.needs_review])
            self.audit.append("system", "case.routed", case_id, route="paralegal_review",
                              fields=fused.needs_review)
            return self.summary(case_id)
        self._match_and_route(case_id, result)
        return self.summary(case_id)

    # ------------------------------------------------------------- matching
    def _facts(self, case_id: str, result: dict) -> CaseFacts:
        row = self.store.case(case_id)
        f = result["fields"]
        ct = result["case_type"]
        top, sub = ct["primary_top"], ct["primary_sub"]
        if f.get("case_type", {}).get("status") == "verified" and f["case_type"]["value"]:
            top, _, sub = f["case_type"]["value"].partition("/")
            sub = sub or None
        labels = [tuple(x) for x in ct["labels"]] if ct["labels"] else []
        if not labels or labels[0][1] != sub:
            labels = [(top, sub, 1.0)] + [l for l in labels if l[1] != sub]
        return CaseFacts(case_id=case_id, state=f["state"]["value"], district=f["district"]["value"],
                         court=f["court"]["value"], top=top, sub=sub, labels=labels,
                         customer_language=row["customer_language"] or "hi",
                         document_language=result.get("document_language", "hi"),
                         fee_band=row["fee_band"], opposing_parties=result.get("opposing_parties", []),
                         next_hearing=(f.get("date_next_hearing") or {}).get("value"))

    def _match_and_route(self, case_id: str, result: dict) -> MatchDecision:
        decision = self.matcher.match(self._facts(case_id, result))
        self._save_decision(case_id, decision)
        status = STATUS_FOR_ROUTE[decision.route]
        self.store.update_case(case_id, status=status)
        if decision.route == SENIOR_REVIEW:
            self.store.enqueue(case_id, "senior", decision.reviewers, decision.route_reasons)
        elif decision.route == PARTNER_QUEUE:
            self.store.enqueue(case_id, "partner", ["partner"], decision.route_reasons)
        self.audit.append("system", "case.matched", case_id, route=decision.route, flags=decision.flags,
                          lawyer_ids=[m.lawyer_id for m in decision.matches],
                          conflict_lawyer_ids=[c["lawyer_id"] for c in decision.conflicts])
        return decision

    # ------------------------------------------------------------ admin side
    def submit_field_review(self, case_id: str, reviewer: str, corrections: dict[str, str]) -> dict:
        """Paralegal confirms or corrects fields, then matching runs on verified facts.
        Corrections become learning data (gazetteer aliases, classifier labels)."""
        row = self._require_case(case_id)
        if row["status"] != "paralegal_review":
            raise WorkflowError(f"case is in {row['status']}, not paralegal_review")
        result = self._load_result(case_id)
        for name, value in corrections.items():
            if name in JURISDICTION_FIELDS and value and value not in self.gaz.entities:
                raise ValueError(f"unknown {name} id: {value}")
            old = result["fields"].get(name, {})
            display = self.gaz.name(value) if name in JURISDICTION_FIELDS else value
            result["fields"][name] = FieldResult(name, value, display, 1.0, "verified",
                                                 issues=[f"verified by {reviewer}"]).to_dict()
            if old.get("value") != value:
                payload = {"field": name, "old": old.get("value"), "new": value}
                if name in JURISDICTION_FIELDS and old.get("evidence"):
                    payload["alias_suggestion"] = old["evidence"][0].get("text")
                self.store.add_feedback(case_id, "field_correction", payload)
        for name in result["needs_review"]:
            fr = result["fields"].get(name, {})
            if fr.get("status") in ("review", "missing") and name not in corrections:
                if name == "documents":
                    continue
                raise ValueError(f"field '{name}' still needs a verified value")
        if result["fields"]["district"]["value"] and not result["fields"]["state"]["value"]:
            st = self.gaz.ancestor_of_type(result["fields"]["district"]["value"], "state")
            result["fields"]["state"] = FieldResult("state", st, self.gaz.name(st), 1.0, "verified").to_dict()
        result["needs_review"] = []
        self._save_result(case_id, result)
        self.store.resolve_items(case_id, "fields", reviewer, {"corrected": sorted(corrections)})
        self.audit.append(reviewer, "case.fields_verified", case_id, fields=sorted(corrections))
        self._match_and_route(case_id, result)
        return self.summary(case_id)

    def senior_decision(self, case_id: str, reviewer: str, approve: bool, reason: str,
                        lawyer_order: list[str] | None = None) -> dict:
        """A named senior lawyer / admin clears (or stops) a flagged match."""
        row = self._require_case(case_id)
        if row["status"] != "senior_review":
            raise WorkflowError(f"case is in {row['status']}, not senior_review")
        if not reason:
            raise ValueError("a reason is required")
        decision = self._load_decision(case_id)
        if lawyer_order:
            self._reorder(decision, lawyer_order)
        self.store.resolve_items(case_id, "senior", reviewer, {"approve": approve, "reason": reason})
        self.store.add_feedback(case_id, "senior_decision",
                                {"approve": approve, "order": [m["lawyer_id"] for m in decision["matches"]]})
        if approve and decision["matches"]:
            route = next_route_after_senior_review(self.settings.matching)
            decision["route"] = route
            self._save_decision_raw(case_id, decision)
            self.store.update_case(case_id, status=STATUS_FOR_ROUTE[route])
            if route == PARTNER_QUEUE:
                self.store.enqueue(case_id, "partner", ["partner"], ["cleared by senior review"])
        else:
            self.store.update_case(case_id, status="paralegal_review")
            self.store.enqueue(case_id, "fields", ["paralegal"], [f"senior review stopped the match: {reason}"])
        self.audit.append(reviewer, "case.senior_decision", case_id, approve=approve, reason_code=reason[:40])
        return self.summary(case_id)

    def partner_decision(self, case_id: str, partner: str, accept: bool, lawyer_id: str | None = None,
                         reason: str | None = None) -> dict:
        """Partner accepts the top match or overrides it with a reason.
        Only available when matching.partner_review is true."""
        if not self.partner_review:
            raise PartnerReviewDisabled("partner review is switched off (matching.partner_review: false)")
        row = self._require_case(case_id)
        if row["status"] != "partner_queue":
            raise WorkflowError(f"case is in {row['status']}, not partner_queue")
        decision = self._load_decision(case_id)
        if not accept:
            if not (lawyer_id and reason):
                raise ValueError("an override needs a lawyer_id and a reason")
            self._reorder(decision, [lawyer_id])
        self._save_decision_raw(case_id, decision)
        self.store.resolve_items(case_id, "partner", partner, {"accept": accept, "lawyer_id": lawyer_id})
        self.store.add_feedback(case_id, "partner_decision", {"accept": accept, "chosen": decision["matches"][0]["lawyer_id"]})
        self.store.update_case(case_id, status="customer_choice")
        self.audit.append(partner, "case.partner_decision", case_id, accept=accept,
                          lawyer_id=decision["matches"][0]["lawyer_id"], reason_code=(reason or "")[:40])
        return self.summary(case_id)

    def _reorder(self, decision: dict, lawyer_order: list[str]) -> None:
        by_id = {m["lawyer_id"]: m for m in decision["matches"]}
        facts_case = decision["case_id"]
        for lid in lawyer_order:
            if lid not in by_id:
                lawyer = next((l for l in self.lawyers if l.id == lid), None)
                if lawyer is None:
                    raise ValueError(f"unknown lawyer {lid}")
                facts = self._facts(facts_case, self._load_result(facts_case))
                elig = self.matcher.filter.check([lawyer], facts)
                if not elig.eligible:
                    raise ValueError(f"{lid} is not eligible: {elig.rejected.get(lid)}")
                by_id[lid] = asdict(self.matcher.scorer.score(facts, lawyer))
        ordered = [by_id[l] for l in lawyer_order] + [m for k, m in by_id.items() if k not in lawyer_order]
        decision["matches"] = ordered[: self.settings.matching.top_n]

    # ---------------------------------------------------------- customer side
    def customer_view(self, case_id: str) -> dict:
        """The top 3 with neutral facts only (Bar Council of India advertising rules:
        no 'best', no win rates)."""
        row = self._require_case(case_id)
        if row["status"] not in ("customer_choice", "assigned"):
            return {"case_id": case_id, "status": row["status"], "lawyers": [],
                    "message": "Your case is being checked by our team. We will show suggested lawyers shortly."}
        decision = self._load_decision(case_id)
        self.audit.append("customer", "case.suggestions_viewed", case_id)
        return {"case_id": case_id, "status": row["status"], "recommended": decision["matches"][0]["lawyer_id"],
                "lawyers": [{"lawyer_id": m["lawyer_id"], **m["public_profile"]} for m in decision["matches"]],
                "assigned": row["assigned_lawyer"]}

    def customer_pick(self, case_id: str, lawyer_id: str) -> dict:
        row = self._require_case(case_id)
        if row["status"] != "customer_choice":
            raise WorkflowError(f"case is in {row['status']}, not customer_choice")
        decision = self._load_decision(case_id)
        ids = [m["lawyer_id"] for m in decision["matches"]]
        if lawyer_id not in ids:
            raise ValueError("pick one of the suggested lawyers")
        self.store.update_case(case_id, status="assigned", assigned_lawyer=lawyer_id)
        self.store.add_feedback(case_id, "customer_pick", {"lawyer_id": lawyer_id, "rank": ids.index(lawyer_id) + 1,
                                                           "shown": ids})
        self.audit.append("customer", "case.assigned", case_id, lawyer_id=lawyer_id, rank=ids.index(lawyer_id) + 1)
        return self.summary(case_id)

    # ----------------------------------------------------------------- misc
    def typeahead(self, entity_type: str, prefix: str, within: str | None = None) -> list[dict]:
        trie = self.tries.get(entity_type)
        if trie is None:
            raise ValueError(f"unknown type {entity_type}")
        return [{"id": i, "label": self.gaz.name(i)} for i in trie.search(prefix, within)]

    def shred_case(self, case_id: str, actor: str, reason: str) -> None:
        """Crypto-shred the case key and purge derived rows (retention / DPDP)."""
        self.keys.shred(case_id)
        obj = self.settings.var_dir / "objects" / case_id
        if obj.exists():
            for p in obj.iterdir():
                p.unlink()
            obj.rmdir()
        self.store.purge_case_rows(case_id)
        self.audit.append(actor, "case.shredded", case_id, reason_code=reason[:40])

    def summary(self, case_id: str) -> dict:
        row = self.store.case(case_id)
        out = {"case_id": case_id, "status": row["status"], "partner_review": self.partner_review}
        if row["result_blob"]:
            r = self._load_result(case_id)
            out["fields"] = {k: {"value": v["value"], "display": v["display"], "confidence": v["confidence"],
                                 "status": v["status"], "issues": v["issues"]} for k, v in r["fields"].items()}
            out["case_type"] = {k: r["case_type"][k] for k in ("primary_top", "primary_sub", "probability", "tier", "evidence")}
            out["sections"] = r["sections"]
            out["needs_review"] = r["needs_review"]
            out["issues"] = r["issues"]
            out["follow_up_questions"] = r.get("follow_up_questions", [])
        if row["decision_json"]:
            d = json.loads(row["decision_json"])
            out["match"] = {"route": d["route"], "route_reasons": d["route_reasons"], "reviewers": d["reviewers"],
                            "flags": d["flags"],
                            "lawyers": [{"lawyer_id": m["lawyer_id"], "name": m["name"], "score": m["score"],
                                         "reasons": m["reasons"]} for m in d["matches"]]}
        if row["assigned_lawyer"]:
            out["assigned_lawyer"] = row["assigned_lawyer"]
        return out

    # persistence helpers (results are encrypted with the case key)
    def _save_result(self, case_id: str, result: dict) -> None:
        blob = envelope.encrypt(self.keys, case_id, "result", json.dumps(result, ensure_ascii=False).encode())
        self.store.update_case(case_id, result_blob=blob)

    def _load_result(self, case_id: str) -> dict:
        row = self.store.case(case_id)
        return json.loads(envelope.decrypt(self.keys, case_id, "result", row["result_blob"]))

    def _save_decision(self, case_id: str, decision: MatchDecision) -> None:
        self._save_decision_raw(case_id, asdict(decision))

    def _save_decision_raw(self, case_id: str, decision: dict) -> None:
        self.store.update_case(case_id, decision_json=json.dumps(decision, ensure_ascii=False))

    def _load_decision(self, case_id: str) -> dict:
        return json.loads(self.store.case(case_id)["decision_json"])
