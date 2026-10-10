"""Plaintext Avalanche Analysis: 1-Bit Sensitivity and Metrics."""

import tempfile
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image

# Output Directory
AVALANCHE_DIR = Path("data/analysis/sensitivity")
AVALANCHE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Academic Paper Matplotlib Style Configurations
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})

def modify_one_bit(img_arr: np.ndarray) -> np.ndarray:
    """Flips the least significant bit (LSB) of a random pixel and channel."""
    mod_arr = img_arr.copy()
    rng = np.random.default_rng()
    h, w, c = mod_arr.shape
    row = rng.integers(0, h)
    col = rng.integers(0, w)
    chan = rng.integers(0, c)
    mod_arr[row, col, chan] ^= 1
    print(f"[+] 1-Bit Modification applied at -> (Row: {row}, Col: {col}, Channel: {chan})")
    return mod_arr

def compute_difference_img(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Computes the absolute difference between two image arrays for plotting."""
    return np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)

def compute_bit_difference(img1: np.ndarray, img2: np.ndarray) -> tuple[int, int, float]:
    """Computes exact bit-level difference (Hamming distance)."""
    xor_diff = np.bitwise_xor(img1, img2)
    changed_bits = np.unpackbits(xor_diff).sum()
    total_bits = img1.size * 8
    percentage = (changed_bits / total_bits) * 100.0
    return int(changed_bits), int(total_bits), float(percentage)

def compute_npcr(c1: np.ndarray, c2: np.ndarray) -> float:
    """Computes Number of Pixels Change Rate (NPCR)."""
    diff = (c1 != c2).astype(np.float64)
    return float((np.sum(diff) / c1.size) * 100.0)

def compute_uaci(c1: np.ndarray, c2: np.ndarray) -> float:
    """Computes Unified Average Changing Intensity (UACI)."""
    diff = np.abs(c1.astype(np.float64) - c2.astype(np.float64))
    return float((np.sum(diff) / (c1.size * 255.0)) * 100.0)

def run_plaintext_analysis(image_filename: str):
    """Executes the plaintext 1-bit avalanche analysis."""
    data_dir = Path("data/images")
    plain_path = data_dir / image_filename
    
    if not plain_path.exists():
        raise FileNotFoundError(f"Source image '{image_filename}' not found in {data_dir}")

    base_stem = plain_path.stem
    plain1_arr = np.array(Image.open(plain_path).convert("RGB"))
    
    # Generate Key
    pk, _ = generate_keypair("ml-kem-768")
    master_key, _ = encapsulate(pk, "ml-kem-768")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1_file = tmp_path / "plain1.png"
        p2_file = tmp_path / "plain2.png"
        e1_file = tmp_path / "enc1.png"
        e2_file = tmp_path / "enc2.png"
        
        # Modify Plaintext
        plain2_arr = modify_one_bit(plain1_arr)
        Image.fromarray(plain1_arr).save(p1_file)
        Image.fromarray(plain2_arr).save(p2_file)
        
        # Encrypt
        print("[+] Encrypting original and modified images...")
        encrypt_image(str(p1_file), master_key, str(e1_file))
        encrypt_image(str(p2_file), master_key, str(e2_file))
        
        enc1_arr = np.array(Image.open(e1_file).convert("RGB"))
        enc2_arr = np.array(Image.open(e2_file).convert("RGB"))
        
    # Calculate Metrics
    diff_enc = compute_difference_img(enc1_arr, enc2_arr)
    changed_bits, total_bits, bit_pct = compute_bit_difference(enc1_arr, enc2_arr)
    npcr_val = compute_npcr(enc1_arr, enc2_arr)
    uaci_val = compute_uaci(enc1_arr, enc2_arr)
    
    # Output Results
    print("\n=======================================================")
    print(f" Plaintext Avalanche Metrics for: {image_filename}")
    print("=======================================================")
    print(f" Bits Changed : {changed_bits} / {total_bits} ({bit_pct:.4f}%)")
    print(f" NPCR         : {npcr_val:.5f}%  (Ideal: ~99.6094%)")
    print(f" UACI         : {uaci_val:.5f}%  (Ideal: ~33.4635%)")
    print("=======================================================\n")

    # Generate Visual Plot
    print("[+] Generating Visual Plot...")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), dpi=300)
    fig.suptitle(f"Plaintext Avalanche Effect (1-bit mod) - {base_stem}", fontweight="bold")
    
    titles = ["Plain 1", "Plain 2 (1-bit mod)", "Enc 1", "Enc 2", "|Enc 1 - Enc 2|"]
    images = [plain1_arr, plain2_arr, enc1_arr, enc2_arr, diff_enc]
    
    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
        
    fig.tight_layout()
    out_pdf = AVALANCHE_DIR / f"{base_stem}_plaintext_avalanche.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"[+] Plot saved to: {out_pdf}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cs.plaintext_avalanche <image_filename>")
        print("Example: uv run python -m cs.plaintext_avalanche lena.tiff")
        sys.exit(1)

    run_plaintext_analysis(sys.argv[1])