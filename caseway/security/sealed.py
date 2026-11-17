"""Sealed boxes for the upload path (design: "Upload path").

The internet-facing intake (DMZ) holds ONLY the processing zone's X25519 public
key. Each upload is sealed on arrival with an ephemeral key pair: ECDH with the
public key, HKDF-SHA256 to derive an AES-256-GCM key, then encrypt. The DMZ
cannot open what it sealed, so a breach of the internet-facing server exposes
no documents. Only the processing zone, which holds the private key and has no
internet route, can unseal.

Sealed layout: 32-byte ephemeral public key | 12-byte nonce | ciphertext+tag
"""

from __future__ import annotations

import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_INFO = b"caseway-sealed-upload-v1"


def _derive(shared: bytes, eph_pub: bytes, recipient_pub: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=eph_pub + recipient_pub, info=_INFO).derive(shared)


def _raw(pub: X25519PublicKey) -> bytes:
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def seal(recipient_public: X25519PublicKey, plaintext: bytes) -> bytes:
    eph = X25519PrivateKey.generate()
    eph_pub = _raw(eph.public_key())
    key = _derive(eph.exchange(recipient_public), eph_pub, _raw(recipient_public))
    nonce = os.urandom(12)
    return eph_pub + nonce + AESGCM(key).encrypt(nonce, plaintext, None)


def unseal(recipient_private: X25519PrivateKey, sealed: bytes) -> bytes:
    eph_pub, nonce, ct = sealed[:32], sealed[32:44], sealed[44:]
    shared = recipient_private.exchange(X25519PublicKey.from_public_bytes(eph_pub))
    key = _derive(shared, eph_pub, _raw(recipient_private.public_key()))
    return AESGCM(key).decrypt(nonce, ct, None)


def load_or_create_keypair(key_dir: str | Path) -> tuple[X25519PrivateKey, X25519PublicKey]:
    """Development helper: in production the private key lives only in the
    processing zone and the DMZ is provisioned with the public key file alone."""
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    priv_path = key_dir / "processing_x25519.key"
    if not priv_path.exists():
        priv = X25519PrivateKey.generate()
        priv_path.write_bytes(priv.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                                 serialization.NoEncryption()))
        (key_dir / "dmz_x25519.pub").write_bytes(_raw(priv.public_key()))
    priv = X25519PrivateKey.from_private_bytes(priv_path.read_bytes())
    return priv, priv.public_key()
