"""6D Chaotic Map Iterator."""

import numpy as np
from cs.key_schedule import derive_chaotic_parameters

import numpy as np

TABLE_SIZE = 65536
LUT_MASK = TABLE_SIZE - 1
LUT_SCALE = float(TABLE_SIZE)
SIN_LUT = np.sin(2.0 * np.pi * np.arange(TABLE_SIZE, dtype=np.float64) / TABLE_SIZE)

def iterate_6d_map(initial_states: list[float], params: dict, length: int, burn_in: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterates the 6D chaotic map and returns the keystreams.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - float_stream: shape (length, 6) in range [0, 1). 
            - byte_stream: shape (length, 4) as uint8 in range [0, 255]. Derived only from x1, x2, x3, x4.
    """
    total_steps = length + burn_in
    X = np.empty((total_steps, 6), dtype=np.float64)
    
    X[0] = initial_states
    
    a = params["alpha"]
    b = params["beta"]
    m = params["mu"]
    d = params["delta"]
    g = params["gamma"]
    
    for n in range(total_steps - 1):
        x1, x2, x3, x4, x5, x6 = X[n]
        
        nx1 = (a[0] * x1 + b[0] * x2 + g * x5) % 1.0
        nx2 = (a[1] * x2 + b[1] * x3 + g * x6) % 1.0
        nx3 = (a[2] * x3 + b[2] * x4 + g * x5) % 1.0
        nx4 = (a[3] * x4 + b[3] * x1 + g * x6) % 1.0
        
        # Discretize continuous inputs into exact integer indices for the LUT
        idx1 = int(x1 * LUT_SCALE) & LUT_MASK
        idx2 = int(x2 * LUT_SCALE) & LUT_MASK
        idx3 = int(x3 * LUT_SCALE) & LUT_MASK
        idx4 = int(x4 * LUT_SCALE) & LUT_MASK
        
        nx5 = (m[0] * x5 + d[0] * SIN_LUT[idx1] * SIN_LUT[idx3]) % 1.0
        nx6 = (m[1] * x6 + d[1] * SIN_LUT[idx2] * SIN_LUT[idx4]) % 1.0
        
        X[n+1] = nx1, nx2, nx3, nx4, nx5, nx6
        
    float_stream = X[burn_in:]
    
    # 1. View the exact 64-bit memory of the floats (no math applied)
    raw_bits = float_stream[:, :4].view(np.uint64)
    
    # 2. Shift right to discard the noisy Least Significant Bits, 
    # then cast to uint8 to grab the 8 bits right above them.
    byte_stream = (raw_bits >> 24).astype(np.uint8)
    
    return float_stream, byte_stream

if __name__ == "__main__":
    from cs.kem import generate_keypair, encapsulate
    
    # 1. Establish 256-bit shared secret via ML-KEM
    pk, sk = generate_keypair("ml-kem-768")
    shared_secret, ct = encapsulate(pk, "ml-kem-768")
    
    print(f"Using ML-KEM Shared Secret (first 16 hex): {shared_secret.hex()[:16]}...")
    
    # 2. Derive properties
    ivs, parameters = derive_chaotic_parameters(shared_secret)
    
    # 3. Generate 100 values for all 6 dimensions
    sequence_length = 100
    float_seq, byte_seq = iterate_6d_map(ivs, parameters, length=sequence_length, burn_in=1000)
    
    print(f"\nSequence Length: {sequence_length}")
    print("First iteration (float):", float_seq[0])
    print("First iteration (bytes from scaling):", byte_seq[0])