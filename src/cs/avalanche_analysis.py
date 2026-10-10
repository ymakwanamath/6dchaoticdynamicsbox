"""Avalanche Effect Analysis Module: NPCR and UACI."""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image

# Output Directory
AVALANCHE_DIR = Path("data/analysis/avalanche")
AVALANCHE_DIR.mkdir(parents=True, exist_ok=True)

def compute_npcr(c1: np.ndarray, c2: np.ndarray) -> float:
    """Computes Number of Pixels Change Rate (NPCR)."""
    diff = (c1 != c2).astype(np.float64)
    return float((np.sum(diff) / c1.size) * 100.0)

def compute_uaci(c1: np.ndarray, c2: np.ndarray) -> float:
    """Computes Unified Average Changing Intensity (UACI)."""
    diff = np.abs(c1.astype(np.float64) - c2.astype(np.float64))
    return float((np.sum(diff) / (c1.size * 255.0)) * 100.0)

def run_avalanche_analysis(image_filename: str):
    """Executes key and plaintext sensitivity analysis."""
    data_dir = Path("data/images")
    plain_path = data_dir / image_filename
    
    if not plain_path.exists():
        raise FileNotFoundError(f"Source image '{image_filename}' not found in {data_dir}")

    base_stem = plain_path.stem
    plain_img = np.array(Image.open(plain_path).convert("RGB"))
    
    # 1. Base Key Generation
    pk, _ = generate_keypair("ml-kem-768")
    key1, _ = encapsulate(pk, "ml-kem-768")
    
    # 2. Key 2 (1-bit difference from Key 1)
    key2 = bytearray(key1)
    key2[0] ^= 1
    key2 = bytes(key2)
    
    # 3. Plaintext 2 (1-bit difference from Plaintext 1)
    plain_img2 = plain_img.copy()
    plain_img2[0, 0, 0] ^= 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1_file = tmp_path / "p1.png"
        p2_file = tmp_path / "p2.png"
        c1_file = tmp_path / "c1.png"
        c2_k_file = tmp_path / "c2_k.png"
        c2_p_file = tmp_path / "c2_p.png"
        
        Image.fromarray(plain_img).save(p1_file)
        Image.fromarray(plain_img2).save(p2_file)
        
        # Encrypt operations
        encrypt_image(str(p1_file), key1, str(c1_file))     # C1: Base Key, Base Plaintext
        encrypt_image(str(p1_file), key2, str(c2_k_file))   # C2_K: Changed Key, Base Plaintext
        encrypt_image(str(p2_file), key1, str(c2_p_file))   # C2_P: Base Key, Changed Plaintext
        
        # Load encrypted arrays
        c1_arr = np.array(Image.open(c1_file).convert("RGB"))
        c2_k_arr = np.array(Image.open(c2_k_file).convert("RGB"))
        c2_p_arr = np.array(Image.open(c2_p_file).convert("RGB"))
        
    # Calculate Metrics
    npcr_key = compute_npcr(c1_arr, c2_k_arr)
    uaci_key = compute_uaci(c1_arr, c2_k_arr)
    
    npcr_plain = compute_npcr(c1_arr, c2_p_arr)
    uaci_plain = compute_uaci(c1_arr, c2_p_arr)
    
    # Export Results
    out_file = AVALANCHE_DIR / f"{base_stem}_avalanche_metrics.txt"
    with open(out_file, "w") as f:
        f.write("=================================================================\n")
        f.write(f"Avalanche Effect Analysis (NPCR & UACI) for: {image_filename}\n")
        f.write("=================================================================\n\n")
        
        f.write(f"{'Sensitivity Type':<25} | {'NPCR (%)':<15} | {'UACI (%)':<15}\n")
        f.write("-" * 62 + "\n")
        f.write(f"{'Key (1-bit change)':<25} | {npcr_key:<15.5f} | {uaci_key:<15.5f}\n")
        f.write(f"{'Plaintext (1-bit change)':<25} | {npcr_plain:<15.5f} | {uaci_plain:<15.5f}\n")
        f.write("=================================================================\n")
        f.write("* Ideal NPCR ~ 99.6094%\n")
        f.write("* Ideal UACI ~ 33.4635%\n")

    print(f"Avalanche analysis complete. Results saved to: {out_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cs.avalanche_analysis <image_filename>")
        sys.exit(1)

    run_avalanche_analysis(sys.argv[1])