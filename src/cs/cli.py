"""Command Line Interface for Chaotic Image Cipher."""

import click
import numpy as np
from pathlib import Path

from cs.kem import generate_keypair, encapsulate
from cs.cipher import encrypt_image, decrypt_image
from cs.key_schedule import derive_chaotic_parameters
from cs.chaotic import iterate_6d_map
from cs.analysis import run_image_analysis

# Workspace Directories
BASE_DIR = Path("workspace")
INPUT_DIR = BASE_DIR / "input"
ENC_DIR = BASE_DIR / "encrypted"
DEC_DIR = BASE_DIR / "decrypted"
KEY_FILE = BASE_DIR / "master.key"

# Data & Analysis Directories
DATA_IMG_DIR = Path("data/images")
ANALYSIS_IMG_DIR = Path("data/analysis/image")
ANALYSIS_CHAOS_DIR = Path("data/analysis/chaos")

# NIST Directories
NIST_SEQ_DIR = Path("data/nist/sequences")
NIST_RES_DIR = Path("data/nist/results")

def ensure_dirs():
    """Create necessary directories if they do not exist."""
    for d in [INPUT_DIR, ENC_DIR, DEC_DIR, DATA_IMG_DIR, ANALYSIS_IMG_DIR, ANALYSIS_CHAOS_DIR, NIST_SEQ_DIR, NIST_RES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def load_key(new_key: bool = False) -> bytes:
    """Loads the existing key or generates a new one if requested."""
    if new_key:
        pk, _ = generate_keypair("ml-kem-768")
        shared_secret, _ = encapsulate(pk, "ml-kem-768")
        with open(KEY_FILE, "wb") as f:
            f.write(shared_secret)
        click.echo(f"[!] Generated new ML-KEM key. Overwrote {KEY_FILE}")
        return shared_secret
        
    if not KEY_FILE.exists():
        raise click.ClickException("Master key not found. Run 'setup' or use the '--new-key' flag.")
    with open(KEY_FILE, "rb") as f:
        return f.read()

@click.group()
def cli():
    """Chaotic S-Box Image Cryptosystem CLI."""
    ensure_dirs()

@cli.command()
def setup():
    """Initialize directories and generate an initial ML-KEM master key."""
    if KEY_FILE.exists():
        click.echo(f"Key file already exists at {KEY_FILE}")
        return
    load_key(new_key=True)
    click.echo(f"Directories created. Key saved to {KEY_FILE}")

@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--new-key", is_flag=True, help="Generate and use a new ML-KEM master key.")
def encrypt(input_path, new_key):
    """Encrypt a single image (saves in the same directory as input)."""
    key = load_key(new_key)
    p = Path(input_path)
    out_path = p.parent / f"[Enc]{p.stem}.png"
    encrypt_image(str(p), key, str(out_path))
    click.echo(f"Saved: {out_path}")

@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
def decrypt(input_path):
    """Decrypt a single encrypted image."""
    key = load_key(new_key=False)
    p = Path(input_path)
    stem = p.stem.replace("[Enc]", "")
    out_path = p.parent / f"[Dec]{stem}.png"
    decrypt_image(str(p), key, str(out_path))
    click.echo(f"Saved: {out_path}")

@cli.command("test-image")
@click.argument("image_filename")
@click.option("--new-key", is_flag=True, help="Generate and use a new ML-KEM master key.")
def test_image(image_filename, new_key):
    """Encrypt and decrypt an image directly from the data/images/ directory."""
    key = load_key(new_key)
    p = DATA_IMG_DIR / image_filename
    
    if not p.exists():
        raise click.ClickException(f"Image not found at {p}. Ensure it is in data/images/")
    
    enc_path = p.parent / f"[Enc]{p.stem}.png"
    dec_path = p.parent / f"[Dec]{p.stem}.png"
    
    click.echo(f"Processing: {p.name}")
    click.echo(f"1. Encrypting to {enc_path.name}...")
    encrypt_image(str(p), key, str(enc_path))
    
    click.echo(f"2. Decrypting to {dec_path.name}...")
    decrypt_image(str(enc_path), key, str(dec_path))
    click.echo("Done.")

@cli.command("analyze-image")
@click.argument("image_filename")
def analyze_image(image_filename):
    """Run 9-panel histogram, correlation, and coefficient analysis."""
    run_image_analysis(image_filename)

@cli.command("batch-encrypt")
@click.option("--ext", default="*.tiff", help="File extension to process (default: *.tiff)")
@click.option("--new-key", is_flag=True, help="Generate and use a new ML-KEM master key.")
def batch_encrypt(ext, new_key):
    """Encrypt all images in the workspace/input directory."""
    key = load_key(new_key)
    files = list(INPUT_DIR.glob(ext))
    if not files:
        click.echo(f"No {ext} files found in {INPUT_DIR}")
        return
    for idx, f in enumerate(files, 1):
        out_path = ENC_DIR / f"[Enc]{f.stem}.png"
        click.echo(f"Encrypting {idx}/{len(files)}: {f.name}")
        encrypt_image(str(f), key, str(out_path))

@cli.command("batch-decrypt")
def batch_decrypt():
    """Decrypt all images in the workspace/encrypted directory."""
    key = load_key(new_key=False)
    files = [f for f in ENC_DIR.glob("*.png") if f.name.startswith("[Enc]")]
    if not files:
        click.echo(f"No encrypted files found in {ENC_DIR}")
        return
    for idx, f in enumerate(files, 1):
        stem = f.stem.replace("[Enc]", "")
        out_path = DEC_DIR / f"[Dec]{stem}.png"
        click.echo(f"Decrypting {idx}/{len(files)}: {f.name}")
        decrypt_image(str(f), key, str(out_path))

@cli.command("generate-nist")
@click.option("--bits", default=1000000, help="Number of bits to generate per sequence (default: 1,000,000).")
@click.option("--new-key", is_flag=True, help="Generate and use a new ML-KEM master key.")
def generate_nist(bits, new_key):
    """Generate raw binary sequences for NIST Statistical Test Suite."""
    key = load_key(new_key)
    num_bytes = (bits + 7) // 8
    
    click.echo(f"Generating {bits} bits ({num_bytes} bytes) for 4 dimensions...")
    ivs, params = derive_chaotic_parameters(key)
    _, byte_stream = iterate_6d_map(ivs, params, length=num_bytes, burn_in=100)
    
    for i in range(4):
        col_bytes = byte_stream[:, i]
        out_file = NIST_SEQ_DIR / f"seq_x{i+1}.bin"
        with open(out_file, "wb") as f:
            f.write(col_bytes.tobytes())
        click.echo(f"Saved x{i+1} binary sequence to {out_file}")
        
    click.echo("Sequences ready. Select '0' (Binary) when prompted by the NIST tool.")

if __name__ == "__main__":
    cli()