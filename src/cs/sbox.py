"""Key-Dependent Chaotic Galois Field S-Box Generation."""

import hashlib
import numpy as np
from cs.key_schedule import derive_chaotic_parameters
from cs.chaotic import iterate_6d_map

# The 30 standard irreducible polynomials of degree 8 over GF(2)
IRREDUCIBLE_POLYNOMIALS = [
    283, 285, 299, 301, 313, 319, 333, 351, 355, 357, 361, 369, 375, 
    379, 391, 395, 397, 415, 419, 425, 433, 445, 451, 463, 471, 477, 
    487, 499, 501, 505
]

def _gf_mult(a: int, b: int, poly: int) -> int:
    """Multiplies two numbers in GF(2^8) using a specific irreducible polynomial."""
    p = 0
    for _ in range(8):
        if b & 1: 
            p ^= a
        a <<= 1
        if a & 0x100: 
            a ^= poly
        b >>= 1
    return p

def _gf_inv(a: int, poly: int) -> int:
    """Finds the multiplicative inverse of 'a' in GF(2^8)."""
    if a == 0: 
        return 0
    # Brute-force search is mathematically guaranteed and efficient for 256 elements
    for i in range(1, 256):
        if _gf_mult(a, i, poly) == 1:
            return i
    return 0

def generate_dynamic_sbox(k_round: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a dynamic 256-element S-box using chaotic Galois Field selection.
    """
    digest = hashlib.shake_256(k_round).digest(32)
    ivs, params = derive_chaotic_parameters(digest)
    
    # Generate deterministic chaotic randomness
    _, byte_stream = iterate_6d_map(ivs, params, length=100, burn_in=500)
    
    # Flatten 100x4 array into 400 linear bytes
    chaotic_bytes = byte_stream.flatten()
    
    # 1. Chaotically select the GF(2^8) irreducible polynomial
    poly_idx = chaotic_bytes[0] % 30
    poly = IRREDUCIBLE_POLYNOMIALS[poly_idx]
    
    # 2. Construct chaotic, strictly invertible affine matrix via LU decomposition
    L = np.eye(8, dtype=np.uint8)
    U = np.eye(8, dtype=np.uint8)
    
    # Extract chaotic bits to fill the lower/upper triangular matrices
    bit_stream = np.unpackbits(chaotic_bytes[1:10])
    idx = 0
    
    for i in range(1, 8):
        for j in range(i):
            L[i, j] = bit_stream[idx]
            idx += 1
            
    for i in range(8):
        for j in range(i + 1, 8):
            U[i, j] = bit_stream[idx]
            idx += 1
            
    M = np.dot(L, U) % 2
    k2 = chaotic_bytes[10]
    
    # 3. Build the S-Box
    sbox = np.empty(256, dtype=np.uint8)
    for x in range(256):
        # Invert in the chaotically selected Galois Field
        inv_x = _gf_inv(x, poly)
        
        # Apply chaotic affine matrix
        v_bits = np.array([(inv_x >> i) & 1 for i in range(8)], dtype=np.uint8)
        v_out_bits = np.dot(M, v_bits) % 2
        
        v_out = 0
        for i in range(8):
            v_out |= (int(v_out_bits[i]) << i)
            
        sbox[x] = v_out ^ k2
        
    inv_sbox = np.empty(256, dtype=np.uint8)
    inv_sbox[sbox] = np.arange(256, dtype=np.uint8)
    
    return sbox, inv_sbox

if __name__ == "__main__":
    from cs.kem import generate_keypair, encapsulate
    
    pk, sk = generate_keypair("ml-kem-768")
    shared_secret, ct = encapsulate(pk, "ml-kem-768")
    
    sbox, inv_sbox = generate_dynamic_sbox(shared_secret[:8])
    
    is_bijective = np.array_equal(inv_sbox[sbox], np.arange(256))
    print(f"Bijective / Invertibility Check: {'PASSED' if is_bijective else 'FAILED'}")