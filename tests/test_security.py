"""Tests for envelope encryption, crypto-shredding, sealed uploads and the audit chain."""

import json

import pytest
from cryptography.exceptions import InvalidTag

from caseway.ingest.sanitiser import RejectedFile, sanitise
from caseway.security import envelope
from caseway.security.audit import AuditLog
from caseway.security.keys import CaseKeyMissing, LocalKeyManager
from caseway.security.sealed import load_or_create_keypair, seal, unseal


def test_envelope_roundtrip_and_binding(tmp_path):
    km = LocalKeyManager(tmp_path)
    km.create_case_key("c1")
    km.create_case_key("c2")
    blob = envelope.encrypt(km, "c1", "doc1", b"secret FIR text")
    assert b"secret" not in blob
    assert envelope.decrypt(km, "c1", "doc1", blob) == b"secret FIR text"
    with pytest.raises(InvalidTag):
        envelope.decrypt(km, "c2", "doc1", blob)       # moved to another case
    with pytest.raises(InvalidTag):
        envelope.decrypt(km, "c1", "doc2", blob)       # relabelled object


def test_crypto_shred_makes_data_unreadable(tmp_path):
    km = LocalKeyManager(tmp_path)
    km.create_case_key("c1")
    blob = envelope.encrypt(km, "c1", "doc1", b"x")
    assert km.shred("c1")
    with pytest.raises(CaseKeyMissing):
        envelope.decrypt(km, "c1", "doc1", blob)


def test_sealed_box_public_key_cannot_open(tmp_path):
    priv, pub = load_or_create_keypair(tmp_path)
    sealed = seal(pub, b"upload bytes")
    assert unseal(priv, sealed) == b"upload bytes"
    other_priv, _ = load_or_create_keypair(tmp_path / "other")
    with pytest.raises(InvalidTag):
        unseal(other_priv, sealed)


def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.append("system", "test.event", f"c{i}", n=i)
    assert log.verify() == (True, None)
    lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    entry = json.loads(lines[2])
    entry["actor"] = "intruder"
    lines[2] = json.dumps(entry)
    (tmp_path / "audit.jsonl").write_text("\n".join(lines) + "\n")
    assert log.verify() == (False, 3)


def test_audit_refuses_personal_data(tmp_path):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "a.jsonl").append("x", "y", "c1", name="Ramesh")


def test_sanitiser_checks_magic_bytes():
    assert sanitise(b"%PDF-1.7\n/JavaScript (x)").flags == ["pdf_active_content:/JavaScript"]
    assert sanitise(b"\x89PNG\r\n\x1a\n....").mime == "image/png"
    with pytest.raises(RejectedFile):
        sanitise(b"MZ\x90\x00\x03\x00binary", allow_text=False)
    with pytest.raises(RejectedFile):
        sanitise(b"x" * (2 * 1024 * 1024), max_mb=1)
