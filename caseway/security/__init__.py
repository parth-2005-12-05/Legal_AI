"""Security building blocks (design: "Security design").

    keys.py      key manager: master key (KEK) -> per-case keys -> crypto-shredding.
                 v1 keeps the KEK in a local file as a development stand-in for
                 HashiCorp Vault Transit backed by an HSM / TPM.
    envelope.py  envelope encryption: each file gets a random AES-256-GCM data key
                 (DEK), wrapped by the per-case key, wrapped by the KEK.
    sealed.py    X25519 "sealed box" for the upload path: the internet-facing
                 intake holds only a public key, so it can encrypt but never decrypt.
    audit.py     append-only, hash-chained audit log (ids only, never document text).
"""
