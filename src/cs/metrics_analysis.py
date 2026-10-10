"""Quantitative Metrics Analysis Module: NPCR, UACI, and SSIM."""

import tempfile
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image

# Output Directory
METRICS_DIR = Path("data/analysis/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def compute_npcr(c1: np.ndarray, c2: np.ndarray) -> float:
    """Computes Number of Pixels Change Rate (NPCR)."""
    diff = (c1 != c2).astype(np.float64)
    return float((np.sum(diff) / c1.size) * 100.0)


def compute_uaci(c1: np.ndarray, c2: np.ndarray) -> float:
    """Computes Unified Average Changing Intensity (UACI)."""
    diff = np.abs(c1.astype(np.float64) - c2.astype(np.float64))
    return float((np.sum(diff) / (c1.size * 255.0)) * 100.0)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes the Structural Similarity Index Measure (SSIM)."""
    # Assuming RGB images, channel_axis=2
    if img1.ndim == 3:
        return structural_similarity(img1, img2, channel_axis=2, data_range=255)
    return structural_similarity(img1, img2, data_range=255)


def run_metrics_analysis(image_filename: str):
    """Executes the encryption and calculates NPCR, UACI, and SSIM."""
    data_dir = Path("data/images")
    plain_path = data_dir / image_filename
    
    if not plain_path.exists():
        raise FileNotFoundError(f"Source image '{image_filename}' not found in {data_dir}")

    base_stem = plain_path.stem
    plain1_arr = np.array(Image.open(plain_path).convert("RGB"))
    
    # Generate 1-bit modified plaintext for NPCR/UACI
    plain2_arr = plain1_arr.copy()
    plain2_arr[0, 0, 0] ^= 1
    
    # Generate Master Key
    pk, _ = generate_keypair("ml-kem-768")
    master_key, _ = encapsulate(pk, "ml-kem-768")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1_file = tmp_path / "plain1.png"
        p2_file = tmp_path / "plain2.png"
        c1_file = tmp_path / "cipher1.png"
        c2_file = tmp_path / "cipher2.png"
        
        Image.fromarray(plain1_arr).save(p1_file)
        Image.fromarray(plain2_arr).save(p2_file)
        
        # Encrypt both
        encrypt_image(str(p1_file), master_key, str(c1_file))
        encrypt_image(str(p2_file), master_key, str(c2_file))
        
        c1_arr = np.array(Image.open(c1_file).convert("RGB"))
        c2_arr = np.array(Image.open(c2_file).convert("RGB"))

    # Calculate Metrics
    ssim_val = compute_ssim(plain1_arr, c1_arr)
    npcr_val = compute_npcr(c1_arr, c2_arr)
    uaci_val = compute_uaci(c1_arr, c2_arr)

    # Export to Table
    out_file = METRICS_DIR / f"{base_stem}_encryption_metrics.txt"
    with open(out_file, "w") as f:
        f.write("=========================================================\n")
        f.write(f"Cryptographic Metrics for: {image_filename}\n")
        f.write("=========================================================\n")
        f.write(f"{'Metric':<10} | {'Value':<15} | {'Ideal Theoretical Value'}\n")
        f.write("-" * 57 + "\n")
        f.write(f"{'SSIM':<10} | {ssim_val:<15.6f} | ~0.000000\n")
        f.write(f"{'NPCR':<10} | {npcr_val:<15.5f} | ~99.60940\n")
        f.write(f"{'UACI':<10} | {uaci_val:<15.5f} | ~33.46350\n")
        f.write("=========================================================\n")

    print(f"[+] Metrics calculation complete. Table written to: {out_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cs.metrics_analysis <image_filename>")
        sys.exit(1)

    run_metrics_analysis(sys.argv[1])