"""FastAPI application exposing CaseWay over HTTP.

Run locally:   .venv\\Scripts\\python -m uvicorn caseway.api.app:app --port 8000
Docs:          http://localhost:8000/docs   (served locally, no CDN needed for the API itself)

Endpoint groups mirror the design's zones:
    customer (DMZ)   POST /cases, POST /cases/{id}/documents,
                     GET /cases/{id}/suggestions, POST /cases/{id}/pick, GET /typeahead
    processing       POST /cases/{id}/process   (a queue worker calls this in production)
    admin (VPN+MFA)  GET /review/queue, GET /cases/{id}, POST /review/{id}/fields,
                     POST /review/{id}/senior, GET /audit/verify
    partner          GET /partner/queue, POST /partner/{id}/decision
                     -> these return 404 while matching.partner_review is false, so the
                        partner role stays hidden until the switch is turned on.

v1 has NO authentication: the X-Actor header only labels audit entries. SSO + MFA,
OTP/passkey customer login and RBAC/ABAC are Phase 4 work; do not expose this
service beyond a development machine until then.
"""

from __future__ import annotations

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from ..ingest.sanitiser import RejectedFile
from ..pipeline import CaseWay, PartnerReviewDisabled, WorkflowError

app = FastAPI(title="CaseWay", version="0.1.0",
              description="Offline case-to-lawyer matching (first version)")
_service: CaseWay | None = None


def service() -> CaseWay:
    global _service
    if _service is None:
        _service = CaseWay()   # settings are read once here: change config + restart to apply
    return _service


def _wrap(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except PartnerReviewDisabled:
        raise HTTPException(404, "Not found")
    except KeyError:
        raise HTTPException(404, "case not found")
    except (WorkflowError, ValueError, RejectedFile) as exc:
        raise HTTPException(409, str(exc))


class NewCase(BaseModel):
    chat_text: str
    customer_language: str = "hi"
    fee_band: int | None = None


class Pick(BaseModel):
    lawyer_id: str


class FieldReview(BaseModel):
    corrections: dict[str, str]


class SeniorDecision(BaseModel):
    approve: bool
    reason: str
    lawyer_order: list[str] | None = None


class PartnerDecision(BaseModel):
    accept: bool
    lawyer_id: str | None = None
    reason: str | None = None


@app.get("/health")
def health():
    s = service()
    return {"status": "ok", "partner_review": s.partner_review, "gazetteer": s.gaz.version}


# ---- customer -------------------------------------------------------------
@app.post("/cases")
def create_case(body: NewCase):
    return {"case_id": service().create_case(body.chat_text, body.customer_language, body.fee_band)}


@app.post("/cases/{case_id}/documents")
async def upload(case_id: str, file: UploadFile = File(...)):
    data = await file.read()
    return {"job_id": _wrap(service().upload, case_id, file.filename or "upload", data)}


@app.get("/cases/{case_id}/suggestions")
def suggestions(case_id: str):
    return _wrap(service().customer_view, case_id)


@app.post("/cases/{case_id}/pick")
def pick(case_id: str, body: Pick):
    return _wrap(service().customer_pick, case_id, body.lawyer_id)


@app.get("/typeahead")
def typeahead(type: str, q: str, within: str | None = None):
    return _wrap(service().typeahead, type, q, within)


# ---- processing -----------------------------------------------------------
@app.post("/cases/{case_id}/process")
def process(case_id: str):
    return _wrap(service().process, case_id)


# ---- admin ----------------------------------------------------------------
@app.get("/cases/{case_id}")
def case_detail(case_id: str, x_actor: str = Header("staff")):
    s = service()
    out = _wrap(s.summary, case_id) if s.store.case(case_id) else None
    if out is None:
        raise HTTPException(404, "case not found")
    s.audit.append(x_actor, "case.viewed", case_id)
    return out


@app.get("/review/queue")
def review_queue(kind: str | None = None):
    if kind == "partner" and not service().partner_review:
        raise HTTPException(404, "Not found")
    items = service().store.open_items(kind)
    if not service().partner_review:
        items = [i for i in items if i["kind"] != "partner"]
    return items


@app.post("/review/{case_id}/fields")
def review_fields(case_id: str, body: FieldReview, x_actor: str = Header("paralegal")):
    return _wrap(service().submit_field_review, case_id, x_actor, body.corrections)


@app.post("/review/{case_id}/senior")
def senior(case_id: str, body: SeniorDecision, x_actor: str = Header("senior_lawyer")):
    return _wrap(service().senior_decision, case_id, x_actor, body.approve, body.reason, body.lawyer_order)


@app.get("/audit/verify")
def audit_verify():
    ok, bad = service().audit.verify()
    return {"intact": ok, "first_bad_seq": bad, "anchor": service().audit.anchor()}


# ---- partner (hidden while the switch is off) ------------------------------
@app.get("/partner/queue")
def partner_queue():
    if not service().partner_review:
        raise HTTPException(404, "Not found")
    return service().store.open_items("partner")


@app.post("/partner/{case_id}/decision")
def partner_decision(case_id: str, body: PartnerDecision, x_actor: str = Header("partner")):
    return _wrap(service().partner_decision, case_id, x_actor, body.accept, body.lawyer_id, body.reason)
