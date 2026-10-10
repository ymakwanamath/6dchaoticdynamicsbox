"""Ciphertext Bit-Level Avalanche Analysis."""

import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image, decrypt_image

def flip_random_bit_image(img_arr: np.ndarray) -> np.ndarray:
    mod_arr = img_arr.copy()
    rng = np.random.default_rng()
    h, w, c = mod_arr.shape
    row, col, chan = rng.integers(0, h), rng.integers(0, w), rng.integers(0, c)
    bit_pos = rng.integers(0, 8)
    mod_arr[row, col, chan] ^= (1 << bit_pos)
    return mod_arr

def compute_bit_difference(p1: np.ndarray, p2: np.ndarray) -> float:
    xor_diff = np.bitwise_xor(p1, p2)
    changed_bits = np.unpackbits(xor_diff).sum()
    total_bits = p1.size * 8
    return float((changed_bits / total_bits) * 100.0)

def run_ciphertext_avalanche(image_filename: str):
    data_dir = Path("data/images")
    plain_path = data_dir / image_filename
    
    if not plain_path.exists():
        raise FileNotFoundError(f"Source image '{image_filename}' not found in {data_dir}")

    plain_arr = np.array(Image.open(plain_path).convert("RGB"))
    
    pk, _ = generate_keypair("ml-kem-768")
    master_key, _ = encapsulate(pk, "ml-kem-768")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p_file = tmp_path / "plain.png"
        c1_file = tmp_path / "c1.png"
        c2_file = tmp_path / "c2.png"
        d1_file = tmp_path / "d1.png"
        d2_file = tmp_path / "d2.png"
        
        Image.fromarray(plain_arr).save(p_file)
        
        # Encrypt the original image
        encrypt_image(str(p_file), master_key, str(c1_file))
        c1_arr = np.array(Image.open(c1_file).convert("RGB"))
        
        # Introduce a 1-bit change to the ciphertext
        c2_arr = flip_random_bit_image(c1_arr)
        Image.fromarray(c2_arr).save(c2_file)
        
        # Decrypt both ciphertexts
        decrypt_image(str(c1_file), master_key, str(d1_file))
        decrypt_image(str(c2_file), master_key, str(d2_file))
        
        d1_arr = np.array(Image.open(d1_file).convert("RGB"))
        d2_arr = np.array(Image.open(d2_file).convert("RGB"))

    # Compute Bit Metrics
    p_pct = compute_bit_difference(d1_arr, d2_arr)
    
    print("\n=======================================================")
    print(f" Ciphertext Sensitivity Analysis: {image_filename}")
    print("=======================================================")
    print(f" 1-Bit Ciphertext Change -> {p_pct:.6f}% Plaintext Bits Changed")
    print("=======================================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cs.ciphertext_avalanche <image_filename>")
        sys.exit(1)

    run_ciphertext_avalanche(sys.argv[1])