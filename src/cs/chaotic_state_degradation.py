"""Chaotic State Degradation Analysis (Multi-Pair Phase Space)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from cs.kem import generate_keypair, encapsulate
from cs.key_schedule import derive_chaotic_parameters

# Output Directory
OUT_DIR = Path("data/analysis/chaotic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# LUT Constants from chaotic.py
TABLE_SIZE = 65536
LUT_MASK = TABLE_SIZE - 1
LUT_SCALE = float(TABLE_SIZE)
SIN_LUT = np.sin(2.0 * np.pi * np.arange(TABLE_SIZE, dtype=np.float64) / TABLE_SIZE)

def iterate_6d_constrained(initial_states: list[float], params: dict, length: int, dtype) -> np.ndarray:
    """Iterates the 6D chaotic map with an artificially constrained precision."""
    burn_in = 1000
    total_steps = length + burn_in
    X = np.empty((total_steps, 6), dtype=dtype)
    
    X[0] = initial_states
    
    a = params["alpha"]
    b = params["beta"]
    m = params["mu"]
    d = params["delta"]
    g = params["gamma"]
    
    for n in range(total_steps - 1):
        x1, x2, x3, x4, x5, x6 = X[n]
        
        nx1 = dtype((a[0] * x1 + b[0] * x2 + g * x5) % 1.0)
        nx2 = dtype((a[1] * x2 + b[1] * x3 + g * x6) % 1.0)
        nx3 = dtype((a[2] * x3 + b[2] * x4 + g * x5) % 1.0)
        nx4 = dtype((a[3] * x4 + b[3] * x1 + g * x6) % 1.0)
        
        idx1 = int(x1 * LUT_SCALE) & LUT_MASK
        idx2 = int(x2 * LUT_SCALE) & LUT_MASK
        idx3 = int(x3 * LUT_SCALE) & LUT_MASK
        idx4 = int(x4 * LUT_SCALE) & LUT_MASK
        
        nx5 = dtype((m[0] * x5 + d[0] * SIN_LUT[idx1] * SIN_LUT[idx3]) % 1.0)
        nx6 = dtype((m[1] * x6 + d[1] * SIN_LUT[idx2] * SIN_LUT[idx4]) % 1.0)
        
        X[n+1] = nx1, nx2, nx3, nx4, nx5, nx6
        
    return X[burn_in:]

def analyze_degradation():
    print("Initializing chaotic parameters via ML-KEM...")
    pk, _ = generate_keypair("ml-kem-768")
    secret_key, _ = encapsulate(pk, "ml-kem-768")
    ivs, params = derive_chaotic_parameters(secret_key)
    
    iterations = 100000
    
    print(f"Generating {iterations} states at float64 precision...")
    states_64 = iterate_6d_constrained(ivs, params, iterations, np.float64)
    
    print(f"Generating {iterations} states at float32 precision...")
    states_32 = iterate_6d_constrained(ivs, params, iterations, np.float32)
    
    print("Plotting phase-space attractors...")
    
    # Define variable pairs to plot: (x1,x2), (x2,x3), (x3,x4), (x4,x1)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    pair_labels = [("x1", "x2"), ("x2", "x3"), ("x3", "x4"), ("x4", "x1")]
    
    # Create a 2x4 grid to accommodate the 4 pairs across 2 precision levels
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Phase-Space Degradation Analysis Across Main Variables", fontsize=16)
    
    for i, (idx_x, idx_y) in enumerate(pairs):
        x_label, y_label = pair_labels[i]
        
        # Row 0: Standard Precision (Float64)
        axes[0, i].scatter(states_64[:, idx_x], states_64[:, idx_y], s=0.01, color='blue', alpha=0.5)
        axes[0, i].set_title(f"Standard (float64) | {x_label} vs {y_label}")
        axes[0, i].set_xlabel(x_label)
        axes[0, i].set_ylabel(y_label)
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylim(0, 1)
        
        # Row 1: Constrained Precision (Float32)
        axes[1, i].scatter(states_32[:, idx_x], states_32[:, idx_y], s=0.01, color='red', alpha=0.5)
        axes[1, i].set_title(f"Constrained (float32) | {x_label} vs {y_label}")
        axes[1, i].set_xlabel(x_label)
        axes[1, i].set_ylabel(y_label)
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 1)
    
    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_file = OUT_DIR / "multi_phase_space_degradation.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Analysis complete. Results exported to: {out_file}")

if __name__ == "__main__":
    analyze_degradation()
    