"""Key manager: KEK -> per-case keys, with crypto-shredding.

Hierarchy (design: "Documents at rest"):
    KEK (master key)          never leaves the HSM in production
      └─ case key (per case)  stored only wrapped by the KEK
           └─ DEK (per file)  stored only wrapped by the case key (see envelope.py)

Deleting a case key makes every copy of that case's documents unreadable,
including copies in backups: this is how retention deadlines and DPDP deletion
requests are enforced ("crypto-shredding").

DEVELOPMENT STAND-IN: the KEK lives in var/keys/kek.key. Production replaces
this class with a Vault Transit client (same three methods) whose KEK sits in an
HSM / TPM 2.0 and is unsealed by 3-of-5 Shamir shares held by named senior staff.
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CaseKeyMissing(KeyError):
    """Raised when a case key was shredded or never created."""


class LocalKeyManager:
    def __init__(self, key_dir: str | Path):
        self.dir = Path(key_dir)
        (self.dir / "cases").mkdir(parents=True, exist_ok=True)
        kek_path = self.dir / "kek.key"
        if not kek_path.exists():
            kek_path.write_bytes(AESGCM.generate_key(bit_length=256))
        self._kek = AESGCM(kek_path.read_bytes())

    def _case_path(self, case_id: str) -> Path:
        safe = "".join(ch for ch in case_id if ch.isalnum() or ch in "-_")
        return self.dir / "cases" / f"{safe}.key"

    def create_case_key(self, case_id: str) -> None:
        key = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)
        wrapped = nonce + self._kek.encrypt(nonce, key, case_id.encode())
        self._case_path(case_id).write_bytes(wrapped)

    def case_key(self, case_id: str) -> bytes:
        """Unwrap the case key in memory for the duration of one job."""
        path = self._case_path(case_id)
        if not path.exists():
            raise CaseKeyMissing(case_id)
        blob = path.read_bytes()
        return self._kek.decrypt(blob[:12], blob[12:], case_id.encode())

    def shred(self, case_id: str) -> bool:
        """Crypto-shred: destroy the case key. Irreversible by design."""
        path = self._case_path(case_id)
        if path.exists():
            path.write_bytes(os.urandom(path.stat().st_size))  # overwrite before unlinking
            path.unlink()
            return True
        return False
