"""Image Sensitivity Analysis Module: Plaintext and Ciphertext Avalanche Effect."""

import tempfile
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image, decrypt_image

# Output Directory
SENSITIVITY_DIR = Path("data/analysis/sensitivity")
SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"    -> Flipped bit at (Row: {row}, Col: {col}, Channel: {chan})")
    return mod_arr

def compute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Computes the absolute difference between two image arrays."""
    return np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)

def compute_bit_difference(img1: np.ndarray, img2: np.ndarray) -> tuple[int, int, float]:
    """Computes the exact bit-level difference (Hamming distance) between two images."""
    # XORing the arrays identifies the bits that differ
    xor_diff = np.bitwise_xor(img1, img2)
    # unpackbits converts uint8 values to a flat array of 0s and 1s, which we sum
    changed_bits = np.unpackbits(xor_diff).sum()
    total_bits = img1.size * 8
    percentage = (changed_bits / total_bits) * 100.0
    return int(changed_bits), int(total_bits), float(percentage)

def run_image_sensitivity_analysis(image_filename: str):
    """Executes plaintext and ciphertext sensitivity analysis and plots the results."""
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
        
        # File paths for Plaintext Sensitivity
        p1_file = tmp_path / "plain1.png"
        p2_file = tmp_path / "plain2.png"
        e1_file = tmp_path / "enc1.png"
        e2_file = tmp_path / "enc2.png"
        
        # File paths for Ciphertext Sensitivity
        d1_file = tmp_path / "dec1.png"
        e1_mod_file = tmp_path / "enc1_mod.png"
        d2_file = tmp_path / "dec2.png"
        
        # =====================================================================
        # PART 1: Plaintext Sensitivity
        # =====================================================================
        print("[+] Processing Plaintext Sensitivity...")
        plain2_arr = modify_one_bit(plain1_arr)
        
        Image.fromarray(plain1_arr).save(p1_file)
        Image.fromarray(plain2_arr).save(p2_file)
        
        encrypt_image(str(p1_file), master_key, str(e1_file))
        encrypt_image(str(p2_file), master_key, str(e2_file))
        
        enc1_arr = np.array(Image.open(e1_file).convert("RGB"))
        enc2_arr = np.array(Image.open(e2_file).convert("RGB"))
        diff_enc = compute_difference(enc1_arr, enc2_arr)
        
        # Calculate and output exact bit differences
        changed_enc, total_enc, pct_enc = compute_bit_difference(enc1_arr, enc2_arr)
        print(f"    -> Avalanche Effect (Enc1 vs Enc2): {changed_enc} out of {total_enc} bits changed ({pct_enc:.4f}%)")

        # =====================================================================
        # PART 2: Ciphertext Sensitivity
        # =====================================================================
        print("[+] Processing Ciphertext Sensitivity...")
        decrypt_image(str(e1_file), master_key, str(d1_file))
        dec1_arr = np.array(Image.open(d1_file).convert("RGB"))
        
        enc1_mod_arr = modify_one_bit(enc1_arr)
        Image.fromarray(enc1_mod_arr).save(e1_mod_file)
        
        decrypt_image(str(e1_mod_file), master_key, str(d2_file))
        dec2_arr = np.array(Image.open(d2_file).convert("RGB"))
        diff_dec = compute_difference(dec1_arr, dec2_arr)
        
        # Calculate and output exact bit differences
        changed_dec, total_dec, pct_dec = compute_bit_difference(dec1_arr, dec2_arr)
        print(f"    -> Avalanche Effect (Dec1 vs Dec2): {changed_dec} out of {total_dec} bits changed ({pct_dec:.4f}%)")

    # =====================================================================
    # Plotting Part 1: Plaintext Sensitivity
    # =====================================================================
    print("[+] Generating Visual Plots...")
    fig1, axes1 = plt.subplots(1, 5, figsize=(15, 3.5), dpi=300)
    fig1.suptitle(f"Plaintext Sensitivity (Avalanche Effect) - {base_stem}", fontweight="bold")
    
    titles1 = ["Plain 1", "Plain 2 (1-bit mod)", "Enc 1", "Enc 2", "|Enc 1 - Enc 2|"]
    images1 = [plain1_arr, plain2_arr, enc1_arr, enc2_arr, diff_enc]
    
    for ax, title, img in zip(axes1, titles1, images1):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
        
    fig1.tight_layout()
    p1_out = SENSITIVITY_DIR / f"{base_stem}_plaintext_sensitivity.pdf"
    fig1.savefig(p1_out)
    fig1.savefig(p1_out.with_suffix(".png"), dpi=300)
    plt.close(fig1)

    # =====================================================================
    # Plotting Part 2: Ciphertext Sensitivity
    # =====================================================================
    fig2, axes2 = plt.subplots(1, 5, figsize=(15, 3.5), dpi=300)
    fig2.suptitle(f"Ciphertext Sensitivity (Decryption Avalanche) - {base_stem}", fontweight="bold")
    
    titles2 = ["Enc 1", "Enc 2 (1-bit mod)", "Dec 1 (Original)", "Dec 2 (Failed)", "|Dec 1 - Dec 2|"]
    images2 = [enc1_arr, enc1_mod_arr, dec1_arr, dec2_arr, diff_dec]
    
    for ax, title, img in zip(axes2, titles2, images2):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
        
    fig2.tight_layout()
    p2_out = SENSITIVITY_DIR / f"{base_stem}_ciphertext_sensitivity.pdf"
    fig2.savefig(p2_out)
    fig2.savefig(p2_out.with_suffix(".png"), dpi=300)
    plt.close(fig2)

    print(f"Analysis complete. Figures exported to:\n - {SENSITIVITY_DIR}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cs.image_sensitivity <image_filename>")
        print("Example: uv run python -m cs.image_sensitivity lena.tiff")
        sys.exit(1)

    run_image_sensitivity_analysis(sys.argv[1])