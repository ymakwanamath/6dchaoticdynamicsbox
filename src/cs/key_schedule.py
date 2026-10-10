"""Derives chaotic map parameters and initial values from a 256-bit secret key."""

import hashlib

def derive_chaotic_parameters(key_256: bytes) -> tuple[list[float], dict[str, list[float] | float]]:
    if len(key_256) < 13:
        raise ValueError("Key must be at least 13 bytes long.")

    # 1. Parameter extraction based on updated cryptographic bounds
    alpha = [4.1 + (key_256[i] / 255.0) * 0.8 for i in range(0, 4)]    # Range: [4.1, 4.9]
    beta  = [0.8 + (key_256[i] / 255.0) * 0.4 for i in range(4, 8)]    # Range: [0.8, 1.2]
    mu    = [2.1 + (key_256[i] / 255.0) * 2.8 for i in range(8, 10)]   # Range: [2.1, 4.9]
    delta = [3.1 + (key_256[i] / 255.0) * 4.4 for i in range(10, 12)]  # Range: [3.1, 7.5]
    gamma = 1.1 + (key_256[12] / 255.0) * 2.4                          # Range: [1.1, 3.5]

    params = {
        "alpha": alpha,
        "beta": beta,
        "mu": mu,
        "delta": delta,
        "gamma": gamma
    }

    # 2. Initial values generation via SHAKE-256
    shake_digest = hashlib.shake_256(key_256).digest(32)  # 256 bits
    bit_string = format(int.from_bytes(shake_digest, 'big'), '0256b')
    
    # Divide 256 bits into 6 equal blocks (42 bits per block)
    initial_states = []
    for i in range(6):
        block_bits = bit_string[i*42 : (i+1)*42]
        val = int(block_bits, 2) / (2**42)  # Map to [0, 1)
        initial_states.append(val)

    return initial_states, params

if __name__ == "__main__":
    from cs.kem import generate_keypair, encapsulate

    # 1. Establish 256-bit shared secret via ML-KEM
    pk, sk = generate_keypair("ml-kem-768")
    shared_secret, ct = encapsulate(pk, "ml-kem-768")

    print(f"Using ML-KEM Shared Secret (first 16 hex): {shared_secret.hex()[:16]}...")

    # 2. Derive parameters and initial values
    ivs, p = derive_chaotic_parameters(shared_secret)
    print(f"Initial Values (x1-x6): {ivs}")
    print(f"Parameters: {p}")