"""S-Box Cryptographic Analysis Module."""

import numpy as np
from pathlib import Path

from cs.kem import generate_keypair, encapsulate
from cs.sbox import generate_dynamic_sbox

# Output Directory
SBOX_DIR = Path("data/analysis/sbox")
SBOX_DIR.mkdir(parents=True, exist_ok=True)

# Precompute 256x256 Dot Product Matrix over GF(2)
idx = np.arange(256)
DOT_MATRIX = np.zeros((256, 256), dtype=np.int8)
for i in range(8):
    DOT_MATRIX ^= ((idx[:, None] >> i) & 1) & ((idx[None, :] >> i) & 1)

def compute_nl(sbox: np.ndarray) -> int:
    """Computes the Non-linearity (NL) of the S-box using the Fast Walsh-Hadamard Transform."""
    min_nl = 128
    for v in range(1, 256):
        fx = DOT_MATRIX[sbox, v][:, None]
        wht = np.sum(1 - 2 * (DOT_MATRIX ^ fx), axis=0)
        nl_v = 128 - np.max(np.abs(wht)) / 2
        min_nl = min(min_nl, nl_v)
    return int(min_nl)

def compute_sac(sbox: np.ndarray) -> float:
    """Computes the Strict Avalanche Criterion (SAC) average value."""
    P = np.zeros((8, 8))
    for i in range(8):
        dx = 1 << i
        dy = sbox ^ sbox[idx ^ dx]
        for j in range(8):
            P[i, j] = np.sum((dy >> j) & 1)
    
    P /= 256.0
    return float(np.mean(P))

def compute_bic(sbox: np.ndarray) -> tuple[float, float]:
    """Computes the Bit Independence Criterion for Non-Linearity (BIC-NL) and SAC (BIC-SAC)."""
    bic_nl_sum = 0.0
    bic_sac_sum = 0.0
    
    for j in range(8):
        for k in range(j + 1, 8):
            h = ((sbox >> j) & 1) ^ ((sbox >> k) & 1)
            
            h_col = h[:, None]
            wht = np.sum(1 - 2 * (DOT_MATRIX ^ h_col), axis=0)
            bic_nl_sum += 128 - np.max(np.abs(wht)) / 2
            
            sac_h = 0
            for i in range(8):
                dx = 1 << i
                sac_h += np.sum(h ^ h[idx ^ dx]) / 256.0
            bic_sac_sum += sac_h / 8.0
            
    pairs = 28.0
    return (bic_nl_sum / pairs, bic_sac_sum / pairs)

def compute_du(sbox: np.ndarray) -> int:
    """Computes the Differential Uniformity (DU) from the Difference Distribution Table (DDT)."""
    ddt = np.zeros((256, 256), dtype=int)
    for dx in range(1, 256):
        dy = sbox ^ sbox[idx ^ dx]
        np.add.at(ddt[dx], dy, 1)
        
    return int(np.max(ddt))

def analyze_sboxes():
    """Generates S-boxes using ML-KEM keys and evaluates their properties."""
    out_file = SBOX_DIR / "sbox_metrics.txt"
    
    with open(out_file, "w") as f:
        # ---------------------------------------------------------
        # Table 1: Detailed Analysis for 5 S-boxes
        # ---------------------------------------------------------
        print("Generating and evaluating 5 dynamic S-boxes...")
        num_small = 5
        results_small = []
        sboxes_small = []
        
        for i in range(1, num_small + 1):
            pk, _ = generate_keypair("ml-kem-768")
            secret_key, _ = encapsulate(pk, "ml-kem-768")
            sbox, _ = generate_dynamic_sbox(secret_key[:8])
            
            nl = compute_nl(sbox)
            sac = compute_sac(sbox)
            bic_nl, bic_sac = compute_bic(sbox)
            du = compute_du(sbox)
            
            results_small.append((i, nl, sac, bic_nl, bic_sac, du))
            sboxes_small.append(sbox)
            print(f"Completed small batch S-Box {i}/{num_small}")

        # Print and Write the First 5 S-Boxes Array Contents
        f.write("===============================================================================================\n")
        f.write("Generated S-Box Arrays (First 5)\n")
        f.write("===============================================================================================\n")
        
        for i, sb in enumerate(sboxes_small, 1):
            f.write(f"S-Box {i}:\n")
            print(f"\nS-Box {i}:")
            for row in range(16):
                row_hex = " ".join([f"{val:02x}" for val in sb[row*16:(row+1)*16]])
                f.write(row_hex + "\n")
                print(row_hex)
            f.write("\n")

        # Table 1 Results
        header1 = f"{'S-Box ID':<10} | {'Non-Linearity':<15} | {'SAC':<10} | {'BIC-NL':<10} | {'BIC-SAC':<10} | {'Diff. Unif.'}\n"
        separator1 = "-" * len(header1) + "\n"
        
        f.write("=" * len(header1) + "\n")
        f.write("Dynamic S-Box Cryptographic Analysis (5 S-Boxes)\n")
        f.write("=" * len(header1) + "\n")
        f.write(header1)
        f.write(separator1)
        
        for r in results_small:
            f.write(f"{r[0]:<10} | {r[1]:<15} | {r[2]:<10.4f} | {r[3]:<10.2f} | {r[4]:<10.4f} | {r[5]}\n")
        f.write("=" * len(header1) + "\n\n\n")

        # ---------------------------------------------------------
        # Table 2: Averages for 10,000 S-boxes
        # ---------------------------------------------------------
        print("\nGenerating and evaluating 10,000 dynamic S-boxes (This will take a long time)...")
        num_large = 10000
        
        sum_nl, sum_sac, sum_bic_nl, sum_bic_sac, sum_du = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(1, num_large + 1):
            pk, _ = generate_keypair("ml-kem-768")
            secret_key, _ = encapsulate(pk, "ml-kem-768")
            sbox, _ = generate_dynamic_sbox(secret_key[:8])
            
            sum_nl += compute_nl(sbox)
            sum_sac += compute_sac(sbox)
            
            bic_n, bic_s = compute_bic(sbox)
            sum_bic_nl += bic_n
            sum_bic_sac += bic_s
            
            sum_du += compute_du(sbox)
            
            if i % 100 == 0:
                print(f"Processed {i}/{num_large} S-Boxes...")

        avg_nl = sum_nl / num_large
        avg_sac = sum_sac / num_large
        avg_bic_nl = sum_bic_nl / num_large
        avg_bic_sac = sum_bic_sac / num_large
        avg_du = sum_du / num_large

        header2 = f"{'Sample Size':<12} | {'Avg Non-Linearity':<18} | {'Avg SAC':<10} | {'Avg BIC-NL':<12} | {'Avg BIC-SAC':<12} | {'Avg Diff. Unif.'}\n"
        separator2 = "-" * len(header2) + "\n"
        
        f.write("=" * len(header2) + "\n")
        f.write("Dynamic S-Box Averages (10,000 S-Boxes)\n")
        f.write("=" * len(header2) + "\n")
        f.write(header2)
        f.write(separator2)
        f.write(f"{num_large:<12} | {avg_nl:<18.2f} | {avg_sac:<10.4f} | {avg_bic_nl:<12.2f} | {avg_bic_sac:<12.4f} | {avg_du:.2f}\n")
        f.write("=" * len(header2) + "\n")

    print(f"\nAnalysis complete. Results exported to: {out_file}")

if __name__ == "__main__":
    analyze_sboxes()