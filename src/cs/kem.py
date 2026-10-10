"""ML-KEM (FIPS 203) Key Encapsulation Mechanism wrapper.

Provides standard keypair generation, encapsulation, and decapsulation interfaces
supporting ML-KEM-768 and ML-KEM-1024 using the cryptography package.
Shared secrets are 32-byte (256-bit) symmetric keys.
"""

from __future__ import annotations
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.mlkem import (
    MLKEM768PrivateKey, MLKEM1024PrivateKey,
    MLKEM768PublicKey, MLKEM1024PublicKey
)

_VARIANTS_SK = {
    "ml-kem-768": MLKEM768PrivateKey,
    "ml-kem-1024": MLKEM1024PrivateKey,
}

_VARIANTS_PK = {
    "ml-kem-768": MLKEM768PublicKey,
    "ml-kem-1024": MLKEM1024PublicKey,
}

def _get_key_classes(variant: str):
    key = variant.strip().lower()
    if key == "ml-kem-512":
        raise NotImplementedError("The standard cryptography library does not support ML-KEM-512.")
    if key not in _VARIANTS_SK:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_VARIANTS_SK.keys())}")
    return _VARIANTS_PK[key], _VARIANTS_SK[key]

def generate_keypair(variant: str = "ml-kem-768") -> tuple[bytes, bytes]:
    """Generate an ML-KEM public and private key pair."""
    _, sk_cls = _get_key_classes(variant)
    sk = sk_cls.generate()
    pk = sk.public_key()
    
    pk_bytes = pk.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    sk_bytes = sk.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    return pk_bytes, sk_bytes

def encapsulate(public_key: bytes, variant: str = "ml-kem-768") -> tuple[bytes, bytes]:
    """Encapsulate a 256-bit shared secret using recipient's public key."""
    pk_cls, _ = _get_key_classes(variant)
    pk_obj = pk_cls.from_public_bytes(public_key)
    
    ciphertext, shared_secret = pk_obj.encapsulate()
    return shared_secret, ciphertext

def decapsulate(ciphertext: bytes, secret_key: bytes, variant: str = "ml-kem-768") -> bytes:
    """Decapsulate ciphertext using recipient's private key to recover the 256-bit shared secret."""
    _, sk_cls = _get_key_classes(variant)
    sk_obj = sk_cls.from_private_bytes(secret_key)
    
    shared_secret = sk_obj.decapsulate(ciphertext)
    return shared_secret

if __name__ == "__main__":
    variant_name = "ml-kem-768"
    print(f"Testing ML-KEM ({variant_name})...")

    # Bob: Generate keypair
    pk, sk = generate_keypair(variant_name)
    print(f"Public Key size:  {len(pk)} bytes")
    print(f"Secret Key size:  {len(sk)} bytes")

    # Alice: Encapsulate
    alice_secret, ct = encapsulate(pk, variant_name)
    print(f"Ciphertext size:  {len(ct)} bytes")
    print(f"Shared secret:    {alice_secret.hex()} ({len(alice_secret)} bytes)")

    # Bob: Decapsulate
    bob_secret = decapsulate(ct, sk, variant_name)

    assert alice_secret == bob_secret, "Shared secret mismatch!"
    print("Status: ML-KEM key exchange verified successfully.")