# Post-Quantum Chaotic Image Encryption

This repository contains the official Python implementation of a novel, highly secure image encryption architecture. The scheme is designed to address the vulnerabilities of classical low-dimensional chaotic systems and emerging quantum threats.

## 🚀 Key Features

*   **Post-Quantum Key Management:** Integrates ML-KEM-768 for secure master key exchange, combined with SHAKE-256 for dynamic session key derivation.
*   **Novel Discrete 6D Chaotic Map:** Utilizes a high-dimensional, volume-expanding chaotic system with strictly positive Lyapunov exponents, neutralizing phase-space reconstruction attacks.
*   **Dynamic Galois Field S-Box:** Employs a key-dependent 256-element substitution box based on irreducible polynomials in $GF(2^8)$ and invertible affine transformations.
*   **Multiplication-Free ARX Diffusion:** Features a highly efficient, bidirectional 64-bit Addition-Rotation-XOR (ARX) PCBC diffusion layer for optimal avalanche effects without the overhead of prime multiplication.

## 📁 Repository Structure

```text
.
├── src/
│   └── cs/
│       ├── __init__.py
│       ├── kem.py                 # Post-quantum key encapsulation
│       ├── key_schedule.py        # Key derivation (SHAKE-256)
│       ├── chaotic.py             # 6D chaotic map generation
│       ├── sbox.py                # Dynamic S-Box generation
│       ├── diffusion.py           # ARX PCBC diffusion layer
│       ├── cipher.py              # Main encryption/decryption API
│       ├── analysis.py            # Histogram and correlation plotting
│       ├── key_sensitivity.py     # NPCR, UACI, and SSIM testing
│       └── uniform_analysis.py    # Black & White plaintexts testing
├── data/
│   ├── images/                    # Place your input images here (e.g., lena.png)
│   └── analysis/
│       └── image/                 # Output directory for plots and metrics
├── pyproject.toml                 # Project dependencies
└── README.md
```

## 🛠️ Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for fast and reproducible Python execution and dependency management. 
*   Python >= 3.10
*   `uv` installed on your system.

## 💻 Usage & Commands

All commands should be executed from the **root directory** of the repository. Place your target plain image (e.g., `lena.png`) inside the `data/images/` directory before running the tests.

### 1. Encryption and Decryption
To encrypt and decrypt an image using the core cipher module:
*(Note: Ensure your `cs.cipher` module has CLI argument parsing enabled for these commands, or adapt to your specific CLI entry point).*

```bash
# Encrypt an image
uv run python -m cs.cipher encrypt data/images/lena.png

# Decrypt an image
uv run python -m cs.cipher decrypt data/images/[Enc]lena.png
```

### 2. Standard Cryptanalysis (Histograms & Correlation)
Generates publication-quality 3x3 histogram plots, correlation scatter plots (horizontal, vertical, diagonal), and outputs a tabular text file with the correlation coefficients.

```bash
uv run python -m cs.analysis lena.png
```
*Outputs will be saved to: `data/analysis/image/`*

### 3. Key Sensitivity Analysis (NPCR & UACI)
Tests the strict avalanche criterion of the key. It encrypts the image with a master key (Key A), alters a single bit to create Key B, and attempts decryption. It outputs NPCR, UACI, and SSIM metrics alongside a side-by-side visual comparison.

```bash
uv run python -m cs.key_sensitivity lena.png
```
*Outputs will be saved to: `data/analysis/image/lena_key_sensitivity_row.png`*

