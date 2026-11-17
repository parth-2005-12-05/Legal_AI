"""SQLite persistence (v1 stand-in for the PostgreSQL data zone).

Tables
    cases     one row per case; the chat text and the extraction result are
              stored ENCRYPTED with the case key (sensitive columns are
              encrypted at application level, as the design requires)
    jobs      job queue between intake and processing: only a job id, case id
              and the path of the SEALED upload cross from the DMZ side
    documents sanitised documents, stored as envelope-encrypted objects on disk
              (MinIO in production)
    queue     review work items: paralegal field review, senior review, partner queue
    feedback  customer picks, partner / senior decisions and field corrections,
              which feed weight learning, gazetteer aliases and classifier retraining

Moving to PostgreSQL later means swapping this class; callers use only its methods.
Row-level security and per-role access are Phase 4 work.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import threading
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT, updated_at TEXT,
    customer_language TEXT, fee_band INTEGER, chat_blob BLOB, result_blob BLOB,
    decision_json TEXT, assigned_lawyer TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, case_id TEXT, sealed_path TEXT, filename_hint TEXT,
    status TEXT, created_at TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, case_id TEXT, mime TEXT, object_path TEXT, pages INTEGER,
    flags TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, kind TEXT, status TEXT,
    reviewers TEXT, reasons TEXT, created_at TEXT, resolved_by TEXT, resolved_at TEXT, resolution TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, kind TEXT, payload TEXT, created_at TEXT
);
"""


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _exec(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.db.execute(sql, params)
            self.db.commit()
            return cur

    # cases
    def create_case(self, case_id: str, customer_language: str, fee_band: int | None, chat_blob: bytes) -> None:
        self._exec("INSERT INTO cases (id, status, created_at, updated_at, customer_language, fee_band, chat_blob)"
                   " VALUES (?, 'intake', ?, ?, ?, ?, ?)", (case_id, now(), now(), customer_language, fee_band, chat_blob))

    def case(self, case_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()

    def update_case(self, case_id: str, **cols) -> None:
        cols["updated_at"] = now()
        sets = ", ".join(f"{k} = ?" for k in cols)
        self._exec(f"UPDATE cases SET {sets} WHERE id = ?", (*cols.values(), case_id))

    def cases_by_status(self, status: str) -> list[str]:
        return [r["id"] for r in self.db.execute("SELECT id FROM cases WHERE status = ?", (status,))]

    # jobs
    def add_job(self, case_id: str, sealed_path: str, filename_hint: str) -> str:
        job_id = new_id("job")
        self._exec("INSERT INTO jobs VALUES (?, ?, ?, ?, 'pending', ?, NULL)",
                   (job_id, case_id, sealed_path, filename_hint, now()))
        return job_id

    def pending_jobs(self, case_id: str) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM jobs WHERE case_id = ? AND status = 'pending'", (case_id,)))

    def finish_job(self, job_id: str, status: str, error: str | None = None) -> None:
        self._exec("UPDATE jobs SET status = ?, error = ? WHERE id = ?", (status, error, job_id))

    # documents
    def add_document(self, doc_id: str, case_id: str, mime: str, object_path: str, pages: int, flags: list[str]) -> None:
        self._exec("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (doc_id, case_id, mime, object_path, pages, json.dumps(flags), now()))

    def documents(self, case_id: str) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM documents WHERE case_id = ? ORDER BY created_at", (case_id,)))

    # review queue
    def enqueue(self, case_id: str, kind: str, reviewers: list[str], reasons: list[str]) -> int:
        cur = self._exec("INSERT INTO queue (case_id, kind, status, reviewers, reasons, created_at)"
                         " VALUES (?, ?, 'open', ?, ?, ?)", (case_id, kind, json.dumps(reviewers), json.dumps(reasons), now()))
        return cur.lastrowid

    def open_items(self, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM queue WHERE status = 'open'" + (" AND kind = ?" if kind else "") + " ORDER BY id"
        rows = self.db.execute(sql, (kind,) if kind else ())
        return [{**dict(r), "reviewers": json.loads(r["reviewers"]), "reasons": json.loads(r["reasons"])} for r in rows]

    def resolve_items(self, case_id: str, kind: str, by: str, resolution: dict) -> None:
        self._exec("UPDATE queue SET status = 'done', resolved_by = ?, resolved_at = ?, resolution = ?"
                   " WHERE case_id = ? AND kind = ? AND status = 'open'",
                   (by, now(), json.dumps(resolution), case_id, kind))

    # feedback (learning signals)
    def add_feedback(self, case_id: str, kind: str, payload: dict) -> None:
        self._exec("INSERT INTO feedback (case_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
                   (case_id, kind, json.dumps(payload, ensure_ascii=False), now()))

    def feedback(self, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM feedback" + (" WHERE kind = ?" if kind else "") + " ORDER BY id"
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in self.db.execute(sql, (kind,) if kind else ())]

    # retention
    def purge_case_rows(self, case_id: str) -> None:
        """Remove derived rows after crypto-shredding (DPDP deletion)."""
        for table, col in (("documents", "case_id"), ("jobs", "case_id"), ("queue", "case_id"), ("feedback", "case_id")):
            self._exec(f"DELETE FROM {table} WHERE {col} = ?", (case_id,))
        self._exec("UPDATE cases SET status = 'shredded', chat_blob = NULL, result_blob = NULL, decision_json = NULL,"
                   " updated_at = ? WHERE id = ?", (now(), case_id))
