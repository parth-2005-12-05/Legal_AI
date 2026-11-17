"""Append-only, hash-chained audit log (design: "Audit").

Every read, write, decryption, assignment and export is logged: who, what case,
when, why (reason codes for overrides). Each entry stores the hash of the
previous entry, so editing or deleting any line breaks the chain and `verify()`
reports the first bad entry. `anchor()` returns the latest hash for the daily
offline anchor (printed or stored on WORM media).

Entries contain ids and reason codes only, never document text or personal details.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import threading
from pathlib import Path

GENESIS = "0" * 64
FORBIDDEN_KEYS = {"text", "name", "names", "phone", "address", "chat", "document"}


def _hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def _last(self) -> dict | None:
        last = None
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = json.loads(line)
        return last

    def append(self, actor: str, action: str, case_id: str | None = None, **details) -> dict:
        bad = FORBIDDEN_KEYS & set(details)
        if bad:
            raise ValueError(f"audit entries must not carry personal data fields: {sorted(bad)}")
        with self._lock:
            last = self._last()
            entry = {
                "seq": (last["seq"] + 1) if last else 1,
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "actor": actor, "action": action, "case_id": case_id, "details": details,
                "prev_hash": last["hash"] if last else GENESIS,
            }
            entry["hash"] = _hash(entry)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return entry

    def entries(self, case_id: str | None = None) -> list[dict]:
        with open(self.path, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        return [r for r in rows if case_id is None or r.get("case_id") == case_id]

    def verify(self) -> tuple[bool, int | None]:
        """(True, None) if the chain is intact, else (False, seq of first bad entry)."""
        prev = GENESIS
        for e in self.entries():
            if e.get("prev_hash") != prev or _hash(e) != e.get("hash"):
                return False, e.get("seq")
            prev = e["hash"]
        return True, None

    def anchor(self) -> str:
        last = self._last()
        return last["hash"] if last else GENESIS
