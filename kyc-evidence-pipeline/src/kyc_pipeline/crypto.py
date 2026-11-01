"""Encryption of stored payloads and graph checkpoints (AES-256-GCM).

The key is the secret KYC_DATA_KEY: 32 random bytes, base64-encoded
(`python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"`).
Each value gets a random 96-bit nonce. The table name is authenticated with the
ciphertext, so a value cannot be moved to another table; swaps within a table are
caught by the audit log's record digests. The store records the key's ID and
refuses a different key. Key rotation is not supported yet.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os

from .credentials import get_secret

PREFIX = "enc1:"


class PayloadCipher:
    def __init__(self, key: bytes):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if len(key) != 32:
            raise ValueError("KYC_DATA_KEY must be 32 bytes, base64-encoded")
        self._aead = AESGCM(key)
        self.key_id = hmac.new(key, b"kyc-pipeline key id", hashlib.sha256).hexdigest()[:16]

    @classmethod
    def from_secret(cls) -> "PayloadCipher | None":
        value = get_secret("KYC_DATA_KEY")
        if not value:
            return None
        try:
            key = base64.b64decode(value, validate=True)
        except ValueError:
            raise ValueError("KYC_DATA_KEY must be base64-encoded") from None
        return cls(key)

    def encrypt_bytes(self, data: bytes, aad: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + self._aead.encrypt(nonce, data, aad)

    def decrypt_bytes(self, blob: bytes, aad: bytes) -> bytes:
        from cryptography.exceptions import InvalidTag
        try:
            return self._aead.decrypt(blob[:12], blob[12:], aad)
        except InvalidTag:
            raise ValueError("An encrypted value failed authentication: wrong key, or the stored value was altered") from None

    def seal(self, text: str, table: str) -> str:
        return PREFIX + self.key_id + ":" + base64.b64encode(self.encrypt_bytes(text.encode("utf-8"), table.encode())).decode()

    def open(self, value: str, table: str) -> str:
        key_id, _, blob = value[len(PREFIX):].partition(":")
        if key_id != self.key_id:
            raise ValueError("A stored value was encrypted with a different key")
        return self.decrypt_bytes(base64.b64decode(blob), table.encode()).decode("utf-8")


class CheckpointCipher:
    """LangGraph CipherProtocol over the same key, for EncryptedSerializer."""

    def __init__(self, cipher: PayloadCipher):
        self.cipher = cipher

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "aesgcm-" + self.cipher.key_id, self.cipher.encrypt_bytes(plaintext, b"checkpoint")

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != "aesgcm-" + self.cipher.key_id:
            raise ValueError("A checkpoint was encrypted with a different key")
        return self.cipher.decrypt_bytes(ciphertext, b"checkpoint")


def checkpoint_serde(cipher: PayloadCipher | None):
    """Serializer for the graph checkpointer: encrypted when a key is configured."""
    from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    return EncryptedSerializer(CheckpointCipher(cipher), JsonPlusSerializer()) if cipher else None
