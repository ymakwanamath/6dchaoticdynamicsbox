# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pillow",
# ]
# ///

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from cs.cipher import encrypt_image

# Directories
DATA_DIR = Path("data/images")
ANALYSIS_DIR = Path("data/analysis/image")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_black_white_encryption():
    # 1. Setup Master Key and Paths
    master_key = os.urandom(32)
    
    black_in = DATA_DIR / "plain_black_highres.png"
    white_in = DATA_DIR / "plain_white_highres.png"
    black_out = DATA_DIR / "enc_black_highres.png"
    white_out = DATA_DIR / "enc_white_highres.png"

    # 2. Generate 1024x1024 RGB High-Resolution Black and White Images
    resolution = (512, 512, 3)
    img_black = np.zeros(resolution, dtype=np.uint8)
    img_white = np.full(resolution, 255, dtype=np.uint8)

    Image.fromarray(img_black).save(black_in)
    Image.fromarray(img_white).save(white_in)

    # 3. Encrypt Both Images
    encrypt_image(str(black_in), master_key, str(black_out))
    encrypt_image(str(white_in), master_key, str(white_out))

    # 4. Load Encrypted Arrays
    enc_black_img = Image.open(black_out)
    enc_white_img = Image.open(white_out)

    # 5. Plot as Subfigures
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    fig.suptitle("Encryption of High-Resolution Uniform Plaintexts (1024x1024)", fontweight="bold")

    axes[0].imshow(enc_black_img)
    axes[0].set_title("Encrypted Black Image")
    axes[0].axis("off")

    axes[1].imshow(enc_white_img)
    axes[1].set_title("Encrypted White Image")
    axes[1].axis("off")

    plt.tight_layout()

    # 6. Save Plot to Analysis Directory
    plot_path = ANALYSIS_DIR / "black_white_highres_encryption_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"High-resolution analysis saved to: {plot_path}")

if __name__ == "__main__":
    analyze_black_white_encryption()