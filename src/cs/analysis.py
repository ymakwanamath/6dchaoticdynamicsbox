"""Image Cryptanalysis Module: Publication-Quality Histogram and Correlation Analysis."""

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Output directory setup
ANALYSIS_IMG_DIR = Path("data/analysis/image")
ANALYSIS_IMG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Academic Paper Matplotlib Style Configurations
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Academic Palette: Muted Red, Green, Blue
PALETTE = {
    "Red": "#C0392B",
    "Green": "#27AE60",
    "Blue": "#2980B9"
}

def compute_correlation_coefficients(channel_arr: np.ndarray, num_pairs: int = 5000) -> dict[str, float]:
    """Computes Pearson correlation coefficients in horizontal, vertical, and diagonal directions."""
    H, W = channel_arr.shape
    rng = np.random.default_rng(seed=42)

    x = rng.integers(0, H - 1, size=num_pairs)
    y = rng.integers(0, W - 1, size=num_pairs)

    p = channel_arr[x, y].astype(np.float64)
    p_h = channel_arr[x, y + 1].astype(np.float64)
    p_v = channel_arr[x + 1, y].astype(np.float64)
    p_d = channel_arr[x + 1, y + 1].astype(np.float64)

    def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
        cov = np.cov(a, b)
        std_prod = np.std(a) * np.std(b)
        if std_prod == 0:
            return 0.0
        return float(cov[0, 1] / std_prod)

    return {
        "horizontal": pearson_corr(p, p_h),
        "vertical": pearson_corr(p, p_v),
        "diagonal": pearson_corr(p, p_d),
        "points": (p, p_h, p_v, p_d),
    }

def run_image_analysis(image_filename: str) -> None:
    """Generates academic 3x3 histogram and correlation figures and exports text metrics."""
    data_dir = Path("data/images")
    plain_path = data_dir / image_filename
    
    if not plain_path.exists():
        raise FileNotFoundError(f"Source image '{image_filename}' not found in {data_dir}")

    base_stem = plain_path.stem
    enc_path = data_dir / f"[Enc]{base_stem}.png"
    dec_path = data_dir / f"[Dec]{base_stem}.png"

    if not enc_path.exists():
        raise FileNotFoundError(f"Missing encrypted image: {enc_path}")
    if not dec_path.exists():
        raise FileNotFoundError(f"Missing decrypted image: {dec_path}")

    img_plain = np.array(Image.open(plain_path).convert("RGB"))
    img_enc = np.array(Image.open(enc_path).convert("RGB"))
    img_dec = np.array(Image.open(dec_path).convert("RGB"))

    images = [("Plaintext", img_plain), ("Ciphertext", img_enc), ("Decrypted", img_dec)]
    channels = [("Red", 0), ("Green", 1), ("Blue", 2)]

    # =========================================================================
    # 1. Publication-Quality 3x3 Histogram
    # =========================================================================
    fig_hist, axes_hist = plt.subplots(3, 3, figsize=(11, 8.5), dpi=300)

    for row_idx, (img_label, img_data) in enumerate(images):
        for col_idx, (chan_label, c_idx) in enumerate(channels):
            ax = axes_hist[row_idx, col_idx]
            channel_data = img_data[:, :, c_idx].flatten()
            color = PALETTE[chan_label]

            # Bar plot outline without heavy borders
            counts, bins = np.histogram(channel_data, bins=256, range=(0, 256))
            ax.fill_between(bins[:-1], counts, color=color, alpha=0.65, step="post")
            ax.plot(bins[:-1], counts, color=color, drawstyle="steps-post", lw=0.6)

            ax.set_xlim([0, 255])
            ax.set_title(f"{img_label} ({chan_label} Channel)", pad=6)
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Occurrences")
            ax.grid(True, linestyle="--")

            # Remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig_hist.tight_layout()
    hist_out = ANALYSIS_IMG_DIR / f"{base_stem}_histogram_3x3.pdf"
    fig_hist.savefig(hist_out)
    fig_hist.savefig(hist_out.with_suffix(".png"), dpi=300)
    plt.close(fig_hist)
    print(f"[+] Publication Histogram figures saved to:\n    - {hist_out}\n    - {hist_out.with_suffix('.png')}")

    # =========================================================================
    # 2. Publication-Quality 3x3 Correlation Scatter Plots (Red Channel)
    # =========================================================================
    directions = ["horizontal", "vertical", "diagonal"]
    dir_indices = {"horizontal": 1, "vertical": 2, "diagonal": 3}

    fig_corr, axes_corr = plt.subplots(3, 3, figsize=(11, 9.5), dpi=300)
    fig_corr.suptitle("Correlation Analysis (Red Channel)", fontweight="bold", y=1.02)

    for row_idx, (img_label, img_data) in enumerate(images):
        # Extract only the Red channel (index 0)
        red_channel_data = img_data[:, :, 0]
        corr_res = compute_correlation_coefficients(red_channel_data)

        for col_idx, direction in enumerate(directions):
            ax = axes_corr[row_idx, col_idx]
            dir_idx = dir_indices[direction]

            p = corr_res["points"][0]
            p_adj = corr_res["points"][dir_idx]
            r_val = corr_res[direction]
            color = PALETTE["Red"]

            ax.scatter(p, p_adj, s=2.5, color=color, alpha=0.25, edgecolors="none", rasterized=True)
            ax.set_title(f"{img_label} - {direction.capitalize()} ($r = {r_val:.4f}$)", pad=6)
            ax.set_xlim([0, 255])
            ax.set_ylim([0, 255])
            ax.set_xlabel(r"Pixel Value $P(x, y)$")
            ax.set_ylabel(r"Adjacent Pixel $P_{adj}$")
            ax.grid(True, linestyle="--")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig_corr.tight_layout()
    corr_out = ANALYSIS_IMG_DIR / f"{base_stem}_correlation_red_channel.pdf"
    fig_corr.savefig(corr_out)
    fig_corr.savefig(corr_out.with_suffix(".png"), dpi=300)
    plt.close(fig_corr)
    print(f"[+] Publication Correlation (Red Channel) figures saved to:\n    - {corr_out}\n    - {corr_out.with_suffix('.png')}")

    # =========================================================================
    # 3. Tabular Correlation Metrics Output
    # =========================================================================
    table_file = ANALYSIS_IMG_DIR / f"{base_stem}_correlation_coefficients.txt"
    with open(table_file, "w") as f:
        header = f"{'Image Layer':<15} | {'Channel':<10} | {'Horizontal':<12} | {'Vertical':<12} | {'Diagonal':<12}\n"
        divider = "=" * len(header) + "\n"
        sub_divider = "-" * len(header) + "\n"
        
        f.write(f"Correlation Coefficient Analysis: {image_filename}\n")
        f.write(divider)
        f.write(header)
        f.write(sub_divider)

        for img_label, img_data in images:
            for chan_label, c_idx in channels:
                res = compute_correlation_coefficients(img_data[:, :, c_idx])
                f.write(
                    f"{img_label:<15} | {chan_label:<10} | "
                    f"{res['horizontal']:>10.5f}   | "
                    f"{res['vertical']:>10.5f}   | "
                    f"{res['diagonal']:>10.5f}\n"
                )
            f.write(sub_divider)

    print(f"[+] Tabular data saved to: {table_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m cs.analysis <image_filename_with_extension>")
        print("Example: uv run python -m cs.analysis lena.tiff")
        sys.exit(1)

    run_image_analysis(sys.argv[1])