"""64-Bit ARX PCBC Diffusion (No Multiplication)."""

import hashlib
import numpy as np

MASK_64 = 0xFFFFFFFFFFFFFFFF

def generate_diffusion_keystream(k_master: bytes, length: int) -> np.ndarray:
    kd = bytearray()
    counter = 0
    while len(kd) < length:
        block_bytes = counter.to_bytes(4, byteorder="big")
        payload = k_master + block_bytes
        kd.extend(hashlib.shake_256(payload).digest(32))
        counter += 1
    return np.frombuffer(kd[:length], dtype=np.uint8)

def _pad_to_64bit(arr: np.ndarray) -> np.ndarray:
    rem = len(arr) % 8
    if rem == 0:
        return arr
    return np.pad(arr, (0, 8 - rem), mode='constant')

def _rotl64(x: int, shift: int) -> int:
    """Bitwise Rotate Left for 64-bit integers."""
    return ((x << shift) | (x >> (64 - shift))) & MASK_64

def _rotr64(x: int, shift: int) -> int:
    """Bitwise Rotate Right for 64-bit integers."""
    return ((x >> shift) | (x << (64 - shift))) & MASK_64

def bidirectional_diffusion(v: np.ndarray, kd: np.ndarray) -> np.ndarray:
    orig_len = len(v)
    v_padded = _pad_to_64bit(v)
    k_padded = _pad_to_64bit(kd)
    
    v64 = v_padded.view(np.uint64).tolist()
    k64 = k_padded.view(np.uint64).tolist()
    L = len(v64)
    
    # Pass 1: Forward ARX PCBC
    p1 = [0] * L
    prev_c = k64[-1] if L > 0 else 0
    
    for i in range(L):
        x = (v64[i] + k64[i] + prev_c) & MASK_64
        
        # 3-Round ARX Mixing (Replaces Prime Multiplication)
        for _ in range(3):
            x = _rotl64(x, 21)
            x = (x + k64[i]) & MASK_64
            x ^= prev_c
            
        p1[i] = x
        prev_c = x
        
    # Pass 2: Backward ARX PCBC
    p2 = [0] * L
    next_c = k64[0] if L > 0 else 0
    
    for i in range(L - 1, -1, -1):
        x = (p1[i] + k64[L - 1 - i] + next_c) & MASK_64
        
        for _ in range(3):
            x = _rotl64(x, 21)
            x = (x + k64[L - 1 - i]) & MASK_64
            x ^= next_c
            
        p2[i] = x
        next_c = x
        
    out_64 = np.array(p2, dtype=np.uint64)
    return out_64.view(np.uint8)[:orig_len]

# ==========================================
# Inverse Functions (Required for Decryption)
# ==========================================

def inverse_bidirectional_diffusion(vb: np.ndarray, kd: np.ndarray) -> np.ndarray:
    orig_len = len(vb)
    vb_padded = _pad_to_64bit(vb)
    k_padded = _pad_to_64bit(kd)
    
    vb64 = vb_padded.view(np.uint64).tolist()
    k64 = k_padded.view(np.uint64).tolist()
    L = len(vb64)
    
    # Reverse Pass 2
    p1 = [0] * L
    next_c = k64[0] if L > 0 else 0
    
    for i in range(L - 1, -1, -1):
        x = vb64[i]
        
        # Exact Reverse of 3-Round ARX Mixing
        for _ in range(3):
            x ^= next_c
            x = (x - k64[L - 1 - i]) & MASK_64
            x = _rotr64(x, 21)
            
        p1[i] = (x - k64[L - 1 - i] - next_c) & MASK_64
        next_c = vb64[i]
        
    # Reverse Pass 1
    v64 = [0] * L
    prev_c = k64[-1] if L > 0 else 0
    
    for i in range(L):
        x = p1[i]
        
        for _ in range(3):
            x ^= prev_c
            x = (x - k64[i]) & MASK_64
            x = _rotr64(x, 21)
            
        v64[i] = (x - k64[i] - prev_c) & MASK_64
        prev_c = p1[i]
        
    out_64 = np.array(v64, dtype=np.uint64)
    return out_64.view(np.uint8)[:orig_len]