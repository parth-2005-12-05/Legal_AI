"""Envelope encryption for documents and sensitive database columns.

Each object gets a fresh random AES-256-GCM data key (DEK). The DEK is wrapped
by the per-case key from the key manager; only the wrapped DEK is stored next to
the ciphertext. Associated data binds a blob to its case and object id, so a
blob copied into another case fails to decrypt.

Blob layout:  b"CWE1" | 12-byte DEK-wrap nonce | 48-byte wrapped DEK
              | 12-byte data nonce | ciphertext+tag
"""

from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .keys import LocalKeyManager

MAGIC = b"CWE1"


def encrypt(km: LocalKeyManager, case_id: str, object_id: str, plaintext: bytes) -> bytes:
    aad = f"{case_id}:{object_id}".encode()
    dek = AESGCM.generate_key(bit_length=256)
    case_key = km.case_key(case_id)
    wrap_nonce, data_nonce = os.urandom(12), os.urandom(12)
    wrapped = AESGCM(case_key).encrypt(wrap_nonce, dek, aad)
    ct = AESGCM(dek).encrypt(data_nonce, plaintext, aad)
    return MAGIC + wrap_nonce + wrapped + data_nonce + ct


def decrypt(km: LocalKeyManager, case_id: str, object_id: str, blob: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("not a CaseWay envelope")
    aad = f"{case_id}:{object_id}".encode()
    wrap_nonce, wrapped = blob[4:16], blob[16:64]
    data_nonce, ct = blob[64:76], blob[76:]
    dek = AESGCM(km.case_key(case_id)).decrypt(wrap_nonce, wrapped, aad)
    return AESGCM(dek).decrypt(data_nonce, ct, aad)
