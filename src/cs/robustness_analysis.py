"""Robustness Analysis Module: Cropping and Noise Attacks."""

import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image, decrypt_image

# Output Directory
ROBUSTNESS_DIR = Path("data/analysis/robustness")
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

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


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def apply_salt_and_pepper_noise(img_arr: np.ndarray, density: float) -> np.ndarray:
    """Injects Salt & Pepper noise into an image array."""
    noisy = img_arr.copy()
    rng = np.random.default_rng(seed=42)
    # Generate probabilities for each pixel
    probs = rng.random(noisy.shape[:2])
    
    # Apply to all channels identically if RGB
    mask_salt = probs < (density / 2.0)
    mask_pepper = (probs >= (density / 2.0)) & (probs < density)
    
    noisy[mask_salt] = 255
    noisy[mask_pepper] = 0
    return noisy


def apply_cropping(img_arr: np.ndarray, crop_ratio: float) -> np.ndarray:
    """Zeroes out a central block of the image based on the area ratio."""
    cropped = img_arr.copy()
    h, w = cropped.shape[:2]
    
    # Calculate crop dimensions based on the area ratio
    crop_h = int(h * np.sqrt(crop_ratio))
    crop_w = int(w * np.sqrt(crop_ratio))
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    cropped[start_h:start_h + crop_h, start_w:start_w + crop_w] = 0
    return cropped


def run_robustness_tests(image_filename: str):
    """Executes noise and cropping attacks on the encrypted image."""
    data_dir = Path("data/images")
    plain_path = data_dir / image_filename
    
    if not plain_path.exists():
        raise FileNotFoundError(f"Source image '{image_filename}' not found in {data_dir}")

    base_stem = plain_path.stem
    plain_img = np.array(Image.open(plain_path).convert("RGB"))

    # Generate Key and Encrypt Original
    pk, _ = generate_keypair("ml-kem-768")
    master_key, _ = encapsulate(pk, "ml-kem-768")

    noise_levels = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%
    crop_ratios = [0.0625, 0.125, 0.25, 0.50]  # 1/16, 1/8, 1/4, 1/2 of area

    results_noise = []
    results_crop = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        tmp_plain = tmp_path / "plain.png"
        tmp_enc = tmp_path / "enc.png"
        tmp_mod = tmp_path / "mod_enc.png"
        tmp_dec = tmp_path / "dec.png"
        
        Image.fromarray(plain_img).save(tmp_plain)
        encrypt_image(str(tmp_plain), master_key, str(tmp_enc))
        enc_img = np.array(Image.open(tmp_enc).convert("RGB"))

        print("[+] Running Salt & Pepper Noise Attacks...")
        for density in noise_levels:
            noisy_enc = apply_salt_and_pepper_noise(enc_img, density)
            Image.fromarray(noisy_enc).save(tmp_mod)
            
            decrypt_image(str(tmp_mod), master_key, str(tmp_dec))
            dec_img = np.array(Image.open(tmp_dec).convert("RGB"))
            
            psnr_val = compute_psnr(plain_img, dec_img)
            results_noise.append((density, noisy_enc, dec_img, psnr_val))

        print("[+] Running Data Loss (Cropping) Attacks...")
        for ratio in crop_ratios:
            cropped_enc = apply_cropping(enc_img, ratio)
            Image.fromarray(cropped_enc).save(tmp_mod)
            
            decrypt_image(str(tmp_mod), master_key, str(tmp_dec))
            dec_img = np.array(Image.open(tmp_dec).convert("RGB"))
            
            psnr_val = compute_psnr(plain_img, dec_img)
            results_crop.append((ratio, cropped_enc, dec_img, psnr_val))

    # =========================================================================
    # Generate Visual Plot: Noise Attack
    # =========================================================================
    fig_n, axes_n = plt.subplots(2, len(noise_levels), figsize=(12, 6), dpi=300)
    fig_n.suptitle(f"Robustness Analysis: Salt & Pepper Noise Attack ({base_stem})", fontweight="bold")
    
    for idx, (density, n_enc, n_dec, psnr_val) in enumerate(results_noise):
        ax_enc = axes_n[0, idx]
        ax_dec = axes_n[1, idx]
        
        ax_enc.imshow(n_enc)
        ax_enc.axis("off")
        ax_enc.set_title(f"Noisy Cipher ({int(density*100)}%)")
        
        ax_dec.imshow(n_dec)
        ax_dec.axis("off")
        ax_dec.set_title(f"Decrypted\nPSNR: {psnr_val:.2f} dB")
        
    fig_n.tight_layout()
    noise_out = ROBUSTNESS_DIR / f"{base_stem}_noise_attack.pdf"
    fig_n.savefig(noise_out)
    fig_n.savefig(noise_out.with_suffix(".png"), dpi=300)
    plt.close(fig_n)

    # =========================================================================
    # Generate Visual Plot: Cropping Attack
    # =========================================================================
    fig_c, axes_c = plt.subplots(2, len(crop_ratios), figsize=(12, 6), dpi=300)
    fig_c.suptitle(f"Robustness Analysis: Data Loss / Cropping Attack ({base_stem})", fontweight="bold")
    
    for idx, (ratio, c_enc, c_dec, psnr_val) in enumerate(results_crop):
        ax_enc = axes_c[0, idx]
        ax_dec = axes_c[1, idx]
        
        ax_enc.imshow(c_enc)
        ax_enc.axis("off")
        ax_enc.set_title(f"Cropped Cipher ({(ratio*100):.1f}%)")
        
        ax_dec.imshow(c_dec)
        ax_dec.axis("off")
        ax_dec.set_title(f"Decrypted\nPSNR: {psnr_val:.2f} dB")
        
    fig_c.tight_layout()
    crop_out = ROBUSTNESS_DIR / f"{base_stem}_cropping_attack.pdf"
    fig_c.savefig(crop_out)
    fig_c.savefig(crop_out.with_suffix(".png"), dpi=300)
    plt.close(fig_c)

    # =========================================================================
    # Export Tabular Metrics
    # =========================================================================
    table_file = ROBUSTNESS_DIR / f"{base_stem}_robustness_metrics.txt"
    with open(table_file, "w") as f:
        f.write("========================================================\n")
        f.write(f"Robustness Analysis Metrics for: {image_filename}\n")
        f.write("========================================================\n\n")
        
        f.write("--- Salt & Pepper Noise Attack ---\n")
        f.write(f"{'Noise Density':<20} | {'Recovered PSNR (dB)'}\n")
        f.write("-" * 45 + "\n")
        for density, _, _, psnr_val in results_noise:
            f.write(f"{f'{int(density*100)}%':<20} | {psnr_val:.4f}\n")
            
        f.write("\n--- Data Loss (Cropping) Attack ---\n")
        f.write(f"{'Cropped Area Ratio':<20} | {'Recovered PSNR (dB)'}\n")
        f.write("-" * 45 + "\n")
        for ratio, _, _, psnr_val in results_crop:
            f.write(f"{f'{ratio*100:.1f}%':<20} | {psnr_val:.4f}\n")
            
    print(f"\n[+] Analysis complete. Visuals and metrics exported to:\n    - {ROBUSTNESS_DIR}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cs.robustness_analysis <image_filename_with_extension>")
        print("Example: uv run python -m cs.robustness_analysis lena.tiff")
        sys.exit(1)

    run_robustness_tests(sys.argv[1])