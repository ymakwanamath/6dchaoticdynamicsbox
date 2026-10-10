# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pillow",
#     "scikit-image"
# ]
# ///

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity
from cs.cipher import encrypt_image, decrypt_image

DATA_DIR = Path("data/images")
ANALYSIS_DIR = Path("data/analysis/image")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def compute_npcr(img1: np.ndarray, img2: np.ndarray) -> float:
    return np.sum(img1 != img2) / img1.size * 100

def compute_uaci(img1: np.ndarray, img2: np.ndarray) -> float:
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    return np.sum(diff) / (255 * img1.size) * 100

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if len(img1.shape) == 3:
        return structural_similarity(img1, img2, channel_axis=-1, data_range=255)
    return structural_similarity(img1, img2, data_range=255)

def analyze_key_sensitivity(image_filename: str):
    plain_path = DATA_DIR / image_filename
    
    if not plain_path.exists():
        print(f"Error: Target image '{image_filename}' not found in {DATA_DIR}")
        sys.exit(1)

    base_stem = plain_path.stem
    enc_path = DATA_DIR / f"{base_stem}_encrypted.png"
    failed_dec_path = DATA_DIR / f"{base_stem}_failed_decrypted.png"

    # Generate 32-byte Key A
    key_a = bytearray(os.urandom(32))
    
    # Generate Key B by flipping the Least Significant Bit (LSB) of the first byte
    key_b = bytearray(key_a)
    key_b[0] ^= 1  

    key_a = bytes(key_a)
    key_b = bytes(key_b)

    # 1. Encrypt with Key A
    encrypt_image(str(plain_path), key_a, str(enc_path))

    # 2. Decrypt with Key B (Failed Decryption)
    decrypt_image(str(enc_path), key_b, str(failed_dec_path))

    # 3. Load images into arrays for metrics
    img_plain = np.array(Image.open(plain_path))
    img_enc = np.array(Image.open(enc_path))
    img_failed = np.array(Image.open(failed_dec_path))

    # 4. Calculate metrics (Plain vs Failed Decrypted)
    npcr_val = compute_npcr(img_plain, img_failed)
    uaci_val = compute_uaci(img_plain, img_failed)
    ssim_val = compute_ssim(img_plain, img_failed)

    print(f"--- Key Sensitivity Metrics for {image_filename} (Plain vs Failed Decryption) ---")
    print(f"NPCR: {npcr_val:.4f}%")
    print(f"UACI: {uaci_val:.4f}%")
    print(f"SSIM: {ssim_val:.5f}")

    # 5. Generate Figure for the paper
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
    
    axes[0].imshow(img_plain)
    axes[0].set_title("Plaintext Image")
    axes[0].axis("off")

    axes[1].imshow(img_enc)
    axes[1].set_title("Encrypted Image (Key A)")
    axes[1].axis("off")

    axes[2].imshow(img_failed)
    axes[2].set_title("Failed Decryption (Key B)")
    axes[2].axis("off")

    plt.tight_layout()
    plot_out = ANALYSIS_DIR / f"{base_stem}_key_sensitivity_row.png"
    plt.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nFigure saved to: {plot_out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python key_sensitivity.py <image_filename_with_extension>")
        print("Example: uv run python key_sensitivity.py lena.png")
        sys.exit(1)

    analyze_key_sensitivity(sys.argv[1])