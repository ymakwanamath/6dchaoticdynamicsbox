"""Computational Cost and Performance Analysis Module."""

import time
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from cs.kem import generate_keypair, encapsulate
from cs.key_schedule import derive_chaotic_parameters
from cs.chaotic import iterate_6d_map
from cs.sbox import generate_dynamic_sbox
from cs.cipher import encrypt_image, decrypt_image

# Output Directory
PERF_DIR = Path("data/analysis/performance")
PERF_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_components(trials: int = 20) -> dict[str, float]:
    """Measures execution latency of individual algorithmic components (O(1) stages)."""
    # 1. ML-KEM-768 Key Encapsulation
    kem_times = []
    for _ in range(trials):
        pk, _ = generate_keypair("ml-kem-768")
        t0 = time.perf_counter()
        encapsulate(pk, "ml-kem-768")
        kem_times.append(time.perf_counter() - t0)

    # 2. Key Derivation
    pk, _ = generate_keypair("ml-kem-768")
    ss, _ = encapsulate(pk, "ml-kem-768")
    kd_times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        derive_chaotic_parameters(ss)
        kd_times.append(time.perf_counter() - t0)

    # 3. Dynamic S-Box Generation
    sbox_times = []
    seed = ss[:8]
    for _ in range(trials):
        t0 = time.perf_counter()
        generate_dynamic_sbox(seed)
        sbox_times.append(time.perf_counter() - t0)

    return {
        "ml_kem_ms": float(np.mean(kem_times) * 1000),
        "key_schedule_ms": float(np.mean(kd_times) * 1000),
        "sbox_ms": float(np.mean(sbox_times) * 1000),
    }


def benchmark_resolutions(resolutions: list[tuple[int, int]], trials: int = 5) -> list[dict]:
    """Measures chaotic map generation, full encryption, and decryption across dimensions."""
    pk, _ = generate_keypair("ml-kem-768")
    master_key, _ = encapsulate(pk, "ml-kem-768")
    ivs, params = derive_chaotic_parameters(master_key)

    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for width, height in resolutions:
            total_pixels = width * height
            raw_bytes_size = total_pixels * 3  # RGB channels
            size_mb = raw_bytes_size / (1024 * 1024)

            # Generate synthetic test image
            rng = np.random.default_rng(seed=42)
            arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            img_file = tmp_path / f"test_{width}x{height}.png"
            enc_file = tmp_path / f"enc_{width}x{height}.png"
            dec_file = tmp_path / f"dec_{width}x{height}.png"
            Image.fromarray(arr).save(img_file)

            # 1. Benchmark 6D Map Trajectory Generation (Keystream)
            chaos_times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                iterate_6d_map(ivs, params, length=total_pixels, burn_in=100)
                chaos_times.append(time.perf_counter() - t0)

            # 2. Benchmark Full Encryption Pipeline
            enc_times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                encrypt_image(str(img_file), master_key, str(enc_file))
                enc_times.append(time.perf_counter() - t0)

            # 3. Benchmark Full Decryption Pipeline
            dec_times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                decrypt_image(str(enc_file), master_key, str(dec_file))
                dec_times.append(time.perf_counter() - t0)

            mean_enc = float(np.mean(enc_times))
            mean_dec = float(np.mean(dec_times))
            mean_chaos = float(np.mean(chaos_times))
            throughput = size_mb / mean_enc if mean_enc > 0 else 0.0

            results.append({
                "resolution": f"{width}x{height}",
                "size_mb": size_mb,
                "chaos_ms": mean_chaos * 1000,
                "enc_time_s": mean_enc,
                "dec_time_s": mean_dec,
                "throughput_mb_s": throughput,
            })

    return results


def run_computational_analysis():
    print("[+] Benchmarking O(1) Cryptographic Components...")
    comp_metrics = benchmark_components()

    resolutions = [(256, 256), (512, 512), (1024, 1024)]
    print(f"[+] Benchmarking Resolutions: {resolutions}...")
    res_metrics = benchmark_resolutions(resolutions)

    out_file = PERF_DIR / "computational_cost.txt"
    with open(out_file, "w") as f:
        # Table 1: Theoretical Asymptotic Complexity
        f.write("========================================================================================\n")
        f.write("                       THEORETICAL COMPUTATIONAL COMPLEXITY                             \n")
        f.write("========================================================================================\n")
        f.write(f"{'Operational Phase':<35} | {'Time Complexity':<20} | {'Space Complexity'}\n")
        f.write("----------------------------------------------------------------------------------------\n")
        f.write(f"{'ML-KEM-768 Encapsulation':<35} | {'O(1)':<20} | {'O(1)'}\n")
        f.write(f"{'Parameter Derivation':<35} | {'O(1)':<20} | {'O(1)'}\n")
        f.write(f"{'Dynamic S-Box Construction':<35} | {'O(1)':<20} | {'O(1)'}\n")
        f.write(f"{'6D Chaotic Map Trajectory Gen.':<35} | {'O(M x N)':<20} | {'O(M x N)'}\n")
        f.write(f"{'Chaotic Masking (XOR)':<35} | {'O(M x N)':<20} | {'O(M x N)'}\n")
        f.write(f"{'Dynamic S-Box Substitution':<35} | {'O(M x N)':<20} | {'O(M x N)'}\n")
        f.write(f"{'Bidirectional Diffusion':<35} | {'O(M x N)':<20} | {'O(M x N)'}\n")
        f.write("----------------------------------------------------------------------------------------\n")
        f.write(f"{'TOTAL SYSTEM COMPLEXITY':<35} | {'O(M x N)':<20} | {'O(M x N)'}\n")
        f.write("========================================================================================\n\n")

        # Table 2: O(1) Precomputation Latency
        f.write("========================================================================================\n")
        f.write("               SETUP & PRECOMPUTATION LATENCY (O(1) OPERATIONS)                         \n")
        f.write("========================================================================================\n")
        f.write(f"  * ML-KEM-768 Key Encapsulation : {comp_metrics['ml_kem_ms']:.4f} ms\n")
        f.write(f"  * Chaotic Parameter Derivation  : {comp_metrics['key_schedule_ms']:.4f} ms\n")
        f.write(f"  * Dynamic S-Box Generation      : {comp_metrics['sbox_ms']:.4f} ms\n")
        f.write("========================================================================================\n\n")

        # Table 3: Empirical Execution Performance
        f.write("========================================================================================\n")
        f.write("                    EMPIRICAL EXECUTION TIME & THROUGHPUT                               \n")
        f.write("========================================================================================\n")
        f.write(f"{'Image Size':<12} | {'Data (MB)':<10} | {'Chaos Gen (ms)':<16} | {'Enc Time (s)':<14} | {'Dec Time (s)':<14} | {'Throughput (MB/s)'}\n")
        f.write("----------------------------------------------------------------------------------------\n")
        for r in res_metrics:
            f.write(
                f"{r['resolution']:<12} | {r['size_mb']:<10.4f} | {r['chaos_ms']:<16.2f} | "
                f"{r['enc_time_s']:<14.4f} | {r['dec_time_s']:<14.4f} | {r['throughput_mb_s']:<15.2f}\n"
            )
        f.write("========================================================================================\n")

    print(f"[+] Performance analysis complete. Output written to: {out_file}")


if __name__ == "__main__":
    run_computational_analysis()