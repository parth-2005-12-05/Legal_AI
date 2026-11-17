"""End-to-end tests: the full pipeline, the partner_review switch, and the HTTP API."""

import pytest
from fastapi.testclient import TestClient

import caseway.api.app as api
from caseway.pipeline import PartnerReviewDisabled, WorkflowError
from conftest import SAMPLES

CONSUMER_CHAT = ("The builder has not given possession of my flat in Indirapuram, Ghaziabad "
                 "for 3 years. I want a refund.")


def test_fir_case_with_conflict_goes_to_senior_review(app):
    cid = app.create_case("My brother was arrested in a cheating case by Kotwali police in Ghaziabad. Need bail.",
                          "hi", 3)
    app.upload(cid, "fir.txt", (SAMPLES / "fir_ghaziabad.txt").read_bytes())
    s = app.process(cid)
    assert s["fields"]["district"]["value"] == "UP-GZB"
    assert s["fields"]["police_station"]["value"] == "UP-GZB-PS-KOT"
    assert s["case_type"]["primary_sub"] == "bail"
    assert s["status"] == "senior_review"
    assert {"conflict", "urgent"} <= set(s["match"]["flags"])
    # the customer sees nothing until the senior review clears it
    assert app.customer_view(cid)["lawyers"] == []
    s = app.senior_decision(cid, "senior_lawyer:t", True, "checked")
    assert s["status"] == "customer_choice"       # partner review is off by default
    assert app.customer_pick(cid, "L001")["status"] == "assigned"
    assert app.store.feedback("customer_pick")[0]["payload"]["rank"] == 1


def test_default_switch_off_customer_sees_top3_directly(app):
    assert app.partner_review is False
    cid = app.create_case(CONSUMER_CHAT, "en", 2)
    s = app.process(cid)
    assert s["status"] == "customer_choice"
    view = app.customer_view(cid)
    assert 1 <= len(view["lawyers"]) <= 3 and view["recommended"] == "L007"
    with pytest.raises(PartnerReviewDisabled):
        app.partner_decision(cid, "partner:t", True)


def test_switch_on_every_match_waits_for_partner(partner_app):
    app = partner_app
    cid = app.create_case(CONSUMER_CHAT, "en", 2)
    s = app.process(cid)
    assert s["status"] == "partner_queue"
    assert app.customer_view(cid)["lawyers"] == []
    with pytest.raises(ValueError):
        app.partner_decision(cid, "partner:t", False)              # override needs a reason
    s = app.partner_decision(cid, "partner:t", False, lawyer_id="L005", reason="client asked for Hindi")
    assert s["status"] == "customer_choice"
    assert app.customer_view(cid)["recommended"] == "L005"


def test_switch_on_flagged_case_goes_senior_then_partner(partner_app):
    app = partner_app
    cid = app.create_case("arrested in cheating case, need bail", "hi", 3)
    app.upload(cid, "fir.txt", (SAMPLES / "fir_ghaziabad.txt").read_bytes())
    assert app.process(cid)["status"] == "senior_review"
    assert app.senior_decision(cid, "admin:t", True, "ok")["status"] == "partner_queue"


def test_low_confidence_goes_to_paralegal_with_questions(app):
    cid = app.create_case("मेरी पत्नी गुजारा भत्ता मांग रही है, मामला परिवार न्यायालय गाजियाबाद में है", "hi")
    s = app.process(cid)
    assert s["status"] == "paralegal_review" and "case_type" in s["needs_review"]
    assert any(q["field"] == "case_type" for q in s["follow_up_questions"])
    with pytest.raises(WorkflowError):
        app.customer_pick(cid, "L004")
    s = app.submit_field_review(cid, "paralegal:t", {"case_type": "family/maintenance"})
    assert s["status"] == "customer_choice"
    assert s["match"]["lawyers"][0]["lawyer_id"] in {"L004", "L005"}


def test_data_encrypted_at_rest_and_shredding(app):
    cid = app.create_case("secret words about Ramesh", "en")
    app.upload(cid, "fir.txt", (SAMPLES / "fir_ghaziabad.txt").read_bytes())
    app.process(cid)
    raw_db = (app.settings.var_dir / "caseway.sqlite3").read_bytes()
    assert b"Ramesh" not in raw_db and b"secret words" not in raw_db
    log = (app.settings.var_dir / "audit.log.jsonl").read_text(encoding="utf-8")
    assert "Ramesh" not in log
    app.shred_case(cid, "admin:t", "dpdp_request")
    assert app.store.case(cid)["status"] == "shredded"
    with pytest.raises(WorkflowError):
        app.process(cid)
    assert app.audit.verify()[0]


def test_rejected_upload(app):
    cid = app.create_case("theft of my bike", "en")
    app.upload(cid, "evil.exe", b"MZ\x90\x00" + bytes(range(256)) * 4)
    s = app.process(cid)
    assert any("rejected" in i for i in s["issues"])


@pytest.fixture
def client(app):
    api._service = app
    yield TestClient(api.app)
    api._service = None


def test_api_flow_and_hidden_partner_endpoints(client):
    assert client.get("/health").json()["partner_review"] is False
    cid = client.post("/cases", json={"chat_text": CONSUMER_CHAT, "customer_language": "en", "fee_band": 2}).json()["case_id"]
    files = {"file": ("fir.txt", (SAMPLES / "fir_ghaziabad.txt").read_bytes(), "text/plain")}
    assert client.post(f"/cases/{cid}/documents", files=files).status_code == 200
    assert client.post(f"/cases/{cid}/process").status_code == 200
    assert client.get("/partner/queue").status_code == 404
    assert client.post(f"/partner/{cid}/decision", json={"accept": True}).status_code == 404
    assert client.get("/typeahead", params={"type": "police_station", "q": "kot", "within": "UP-MRT"}).json() == \
        [{"id": "UP-MRT-PS-KOT", "label": "Kotwali, Meerut"}]
    assert client.get("/audit/verify").json()["intact"] is True
    assert client.get("/cases/nope/suggestions").status_code == 404


def test_switch_is_one_setting_in_the_yaml(tmp_path):
    """Turning partner review on is only a config change read at startup (restart)."""
    from caseway.config import DEFAULT_CONFIG, load_settings
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace("partner_review: false", "partner_review: true")
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(text, encoding="utf-8")
    assert load_settings().matching.partner_review is False
    assert load_settings(cfg).matching.partner_review is True
