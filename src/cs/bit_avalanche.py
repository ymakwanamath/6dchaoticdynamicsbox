"""Strict Bit-Level Avalanche Analysis Module with Visual Plotting."""

import sys
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image

# Output Directory
AVALANCHE_DIR = Path("data/analysis/sensitivity")
AVALANCHE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})

def flip_random_bit_image(img_arr: np.ndarray) -> np.ndarray:
    mod_arr = img_arr.copy()
    rng = np.random.default_rng()
    h, w, c = mod_arr.shape
    row, col, chan = rng.integers(0, h), rng.integers(0, w), rng.integers(0, c)
    bit_pos = rng.integers(0, 8)
    mod_arr[row, col, chan] ^= (1 << bit_pos)
    return mod_arr

def flip_random_bit_key(key_bytes: bytes) -> bytes:
    key_ba = bytearray(key_bytes)
    rng = np.random.default_rng()
    byte_idx = rng.integers(0, len(key_ba))
    bit_pos = rng.integers(0, 8)
    key_ba[byte_idx] ^= (1 << bit_pos)
    return bytes(key_ba)

def compute_bit_difference(c1: np.ndarray, c2: np.ndarray) -> tuple[int, int, float]:
    xor_diff = np.bitwise_xor(c1, c2)
    changed_bits = np.unpackbits(xor_diff).sum()
    total_bits = c1.size * 8
    return int(changed_bits), int(total_bits), float((changed_bits / total_bits) * 100.0)

def compute_difference_img(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)

def run_visual_bit_avalanche(image_filename: str):
    data_dir = Path("data/images")
    plain_path = data_dir / image_filename
    
    if not plain_path.exists():
        raise FileNotFoundError(f"Source image '{image_filename}' not found in {data_dir}")

    base_stem = plain_path.stem
    plain1_arr = np.array(Image.open(plain_path).convert("RGB"))
    plain2_arr = flip_random_bit_image(plain1_arr)
    
    pk, _ = generate_keypair("ml-kem-768")
    master_key1, _ = encapsulate(pk, "ml-kem-768")
    master_key2 = flip_random_bit_key(master_key1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1_file, p2_file = tmp_path / "plain1.png", tmp_path / "plain2.png"
        c1_file = tmp_path / "c1.png"
        c2_key_file = tmp_path / "c2_key.png"
        c2_plain_file = tmp_path / "c2_plain.png"
        
        Image.fromarray(plain1_arr).save(p1_file)
        Image.fromarray(plain2_arr).save(p2_file)
        
        # Encrypt the variations
        encrypt_image(str(p1_file), master_key1, str(c1_file))
        encrypt_image(str(p1_file), master_key2, str(c2_key_file))
        encrypt_image(str(p2_file), master_key1, str(c2_plain_file))
        
        c1_arr = np.array(Image.open(c1_file).convert("RGB"))
        c2_key_arr = np.array(Image.open(c2_key_file).convert("RGB"))
        c2_plain_arr = np.array(Image.open(c2_plain_file).convert("RGB"))

    # Compute Bit Metrics
    k_changed, k_total, k_pct = compute_bit_difference(c1_arr, c2_key_arr)
    p_changed, p_total, p_pct = compute_bit_difference(c1_arr, c2_plain_arr)
    
    print("\n=======================================================")
    print(f" Strict Bit-Level Avalanche Analysis: {image_filename}")
    print("=======================================================")
    print(f" 1-Bit Key Change       : {k_pct:.4f}% bits changed")
    print(f" 1-Bit Plaintext Change : {p_pct:.4f}% bits changed")
    print("=======================================================\n")

    # Compute Absolute Difference Images for Plotting
    diff_key = compute_difference_img(c1_arr, c2_key_arr)
    diff_plain = compute_difference_img(c1_arr, c2_plain_arr)

    # Generate 2x3 Plot
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), dpi=300)
    fig.suptitle(f"Avalanche Visual Analysis - {base_stem}", fontweight="bold")
    
    # Row 1: Key Avalanche
    axes[0, 0].imshow(c1_arr)
    axes[0, 0].set_title("Original Ciphertext")
    axes[0, 1].imshow(c2_key_arr)
    axes[0, 1].set_title("C2 (1-Bit Key Change)")
    axes[0, 2].imshow(diff_key)
    axes[0, 2].set_title("|C1 - C2| (Key Diff)")
    
    # Row 2: Plaintext Avalanche
    axes[1, 0].imshow(c1_arr)
    axes[1, 0].set_title("Original Ciphertext")
    axes[1, 1].imshow(c2_plain_arr)
    axes[1, 1].set_title("C2 (1-Bit Plaintext Change)")
    axes[1, 2].imshow(diff_plain)
    axes[1, 2].set_title("|C1 - C2| (Plain Diff)")
    
    for ax in axes.flatten():
        ax.axis("off")
        
    fig.tight_layout()
    out_file = AVALANCHE_DIR / f"{base_stem}_avalanche_visual.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[+] Plot saved to: {out_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cs.bit_avalanche <image_filename>")
        sys.exit(1)

    run_visual_bit_avalanche(sys.argv[1])