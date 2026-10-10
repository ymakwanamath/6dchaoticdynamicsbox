"""Chaotic Map Analysis Module: Extended Parameter Domain, Stability, Bifurcation, LE, and Sensitivity."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from cs.key_schedule import derive_chaotic_parameters
from cs.kem import generate_keypair, encapsulate

# Output Directory
CHAOS_DIR = Path("data/analysis/chaos")
CHAOS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Academic Paper Matplotlib Style Configurations
# -------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "savefig.bbox": "tight",
})

def get_baseline_params():
    """Generates a random master key to derive baseline parameters."""
    pk, _ = generate_keypair("ml-kem-768")
    sk, _ = encapsulate(pk, "ml-kem-768")
    return derive_chaotic_parameters(sk)

def jacobian(X, params):
    """Computes the Jacobian matrix of the 6D map at state X."""
    x1, x2, x3, x4, x5, x6 = X
    a, b, m, d, g = params["alpha"], params["beta"], params["mu"], params["delta"], params["gamma"]
    
    J = np.zeros((6, 6))
    J[0, 0] = a[0]; J[0, 1] = b[0]; J[0, 4] = g
    J[1, 1] = a[1]; J[1, 2] = b[1]; J[1, 5] = g
    J[2, 2] = a[2]; J[2, 3] = b[2]; J[2, 4] = g
    J[3, 0] = b[3]; J[3, 3] = a[3]; J[3, 5] = g
    
    two_pi = 2.0 * np.pi
    J[4, 0] = d[0] * two_pi * np.cos(two_pi * x1) * np.sin(two_pi * x3)
    J[4, 2] = d[0] * two_pi * np.sin(two_pi * x1) * np.cos(two_pi * x3)
    J[4, 4] = m[0]
    
    J[5, 1] = d[1] * two_pi * np.cos(two_pi * x2) * np.sin(two_pi * x4)
    J[5, 3] = d[1] * two_pi * np.sin(two_pi * x2) * np.cos(two_pi * x4)
    J[5, 5] = m[1]
    
    return J

def map_step(X, params):
    """Single iteration of the 6D map."""
    x1, x2, x3, x4, x5, x6 = X
    a, b, m, d, g = params["alpha"], params["beta"], params["mu"], params["delta"], params["gamma"]
    two_pi = 2.0 * np.pi
    
    nx1 = (a[0] * x1 + b[0] * x2 + g * x5) % 1.0
    nx2 = (a[1] * x2 + b[1] * x3 + g * x6) % 1.0
    nx3 = (a[2] * x3 + b[2] * x4 + g * x5) % 1.0
    nx4 = (a[3] * x4 + b[3] * x1 + g * x6) % 1.0
    nx5 = (m[0] * x5 + d[0] * np.sin(two_pi * x1) * np.sin(two_pi * x3)) % 1.0
    nx6 = (m[1] * x6 + d[1] * np.sin(two_pi * x2) * np.sin(two_pi * x4)) % 1.0
    return np.array([nx1, nx2, nx3, nx4, nx5, nx6])

# =========================================================================
# 1. Phase Portrait
# =========================================================================
def plot_phase_portrait():
    print("[1/5] Generating Phase Portraits...")
    ivs, params = get_baseline_params()
    
    length = 15000
    X = np.empty((length, 6))
    X[0] = ivs
    for i in range(1, length):
        X[i] = map_step(X[i - 1], params)
        
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
    fig.suptitle("Phase Portraits of First Four Keystream Dimensions", fontweight="bold")
    
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // 3, idx % 3]
        ax.scatter(X[2000:, i], X[2000:, j], s=0.1, color="#2980B9", alpha=0.5, rasterized=True)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, linestyle="--", alpha=0.4)
        
    plt.tight_layout()
    plt.savefig(CHAOS_DIR / "phase_portrait.pdf")
    plt.savefig(CHAOS_DIR / "phase_portrait.png", dpi=300)
    plt.close()
    
    # Export Table (Sampled 1000 points to keep file size manageable)
    table_file = CHAOS_DIR / "phase_portrait_table.txt"
    with open(table_file, "w") as f:
        header = f"{'Iteration':<12} | {'x1':<12} | {'x2':<12} | {'x3':<12} | {'x4':<12}\n"
        f.write("=" * len(header.strip()) + "\n")
        f.write(header)
        f.write("-" * len(header.strip()) + "\n")
        
        sampled_indices = np.linspace(2000, length - 1, 1000, dtype=int)
        for i in sampled_indices:
            f.write(f"{i:<12} | {X[i, 0]:<12.6f} | {X[i, 1]:<12.6f} | {X[i, 2]:<12.6f} | {X[i, 3]:<12.6f}\n")
        f.write("=" * len(header.strip()) + "\n")

# =========================================================================
# 2. Stability Analysis (3D Extended: [0.5, 6.5] Range)
# =========================================================================
def plot_stability_analysis():
    print("[2/5] Generating 3D Stability Analysis (Extended Alpha Range [0.5, 6.5])...")
    ivs, params = get_baseline_params()
    
    alpha_pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.1, 4.1), (5.0, 5.0), (6.0, 6.0)]
    a2_range = np.linspace(0.5, 6.5, 60)
    a3_range = np.linspace(0.5, 6.5, 60)
    
    A2, A3 = np.meshgrid(a2_range, a3_range)
    
    fig = plt.figure(figsize=(18, 10), dpi=300)
    fig.suptitle(r"Minimum Non-Zero Eigenvalue Magnitude ($\min |\lambda|$) across Extended Parameter Space", fontweight="bold")
    
    table_file = CHAOS_DIR / "stability_analysis_table.txt"
    with open(table_file, "w") as f:
        header = f"{'a0':<8} | {'a1':<8} | {'a2':<10} | {'a3':<10} | {'min|lambda|':<15}\n"
        f.write("=" * len(header.strip()) + "\n")
        f.write(header)
        f.write("-" * len(header.strip()) + "\n")
        
        for idx, (a0, a1) in enumerate(alpha_pairs):
            ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
            Z = np.zeros_like(A2)
            
            for i, a3 in enumerate(a3_range):
                for j, a2 in enumerate(a2_range):
                    p_copy = params.copy()
                    p_copy["alpha"] = [a0, a1, a2, a3]
                    
                    J = jacobian(ivs, p_copy)
                    eigvals = np.abs(np.linalg.eigvals(J))
                    min_nz = np.min(eigvals[eigvals > 1e-10]) if np.any(eigvals > 1e-10) else 0
                    Z[i, j] = min_nz
                    
                    # Log the operational optimal pair configuration to keep the file reasonably sized
                    if a0 == 4.1 and a1 == 4.1:
                        f.write(f"{a0:<8} | {a1:<8} | {a2:<10.4f} | {a3:<10.4f} | {min_nz:<15.6f}\n")
                    
            surf = ax.plot_surface(A2, A3, Z, cmap="plasma", edgecolor='none', alpha=0.9, antialiased=True)
            
            ax.set_title(r"$\alpha_0={}, \alpha_1={}$".format(a0, a1))
            ax.set_xlabel(r"$\alpha_2$")
            ax.set_ylabel(r"$\alpha_3$")
            ax.set_zlabel(r"$\min |\lambda|$")
            ax.view_init(elev=25, azim=-135)
            
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
            cbar.formatter = ScalarFormatter()
            cbar.formatter.set_scientific(False)
            cbar.formatter.set_useOffset(False)
            cbar.update_ticks()
            
        f.write("=" * len(header.strip()) + "\n")
            
    plt.tight_layout()
    plt.savefig(CHAOS_DIR / "stability_analysis_3d.pdf")
    plt.savefig(CHAOS_DIR / "stability_analysis_3d.png", dpi=300)
    plt.close()

# =========================================================================
# 3. Bifurcation & Lyapunov for Alpha (Extended: [0.5, 6.5])
# =========================================================================
def plot_bifurcation_and_lyapunov_alpha():
    print("[3/5] Computing Extended Alpha Sweep ([0.5, 6.5]) for Bifurcation & LEs...")
    ivs, params = get_baseline_params()
    
    a0_range = np.linspace(0.5, 6.5, 450)
    bifurcation_x1 = []
    
    le1, le2, le3, le4 = [], [], [], []
    le_sum = []
    d_ky = []
    
    length = 1500
    burn_in = 600
    
    for a0 in a0_range:
        p_copy = params.copy()
        p_copy["alpha"][0] = a0
        
        X = ivs.copy()
        for _ in range(burn_in):
            X = map_step(X, p_copy)
            
        points = []
        Q = np.eye(6)
        LE = np.zeros(6)
        
        for _ in range(length):
            X = map_step(X, p_copy)
            points.append(X[0])
            
            J = jacobian(X, p_copy)
            Q, R = np.linalg.qr(J @ Q)
            LE += np.log(np.abs(np.diag(R)) + 1e-15)
            
        bifurcation_x1.append(points)
        
        LE /= length
        LE = np.sort(LE)[::-1]
        
        le1.append(LE[0]); le2.append(LE[1]); le3.append(LE[2]); le4.append(LE[3])
        le_sum.append(np.sum(LE))
        
        cumsum = np.cumsum(LE)
        j = np.where(cumsum >= 0)[0]
        if len(j) == 0:
            dky = 0.0
        elif j[-1] == 5:
            dky = 6.0
        else:
            k = j[-1]
            dky = (k + 1) + cumsum[k] / (np.abs(LE[k + 1]) + 1e-15)
        d_ky.append(dky)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    for i, a0 in enumerate(a0_range):
        axes[0].plot([a0] * length, bifurcation_x1[i], ',k', alpha=0.03)
    axes[0].axvspan(2.0, 5.0, color='gray', alpha=0.15, label="Standard Region [2.0, 5.0]")
    axes[0].axvline(4.1, color='green', linestyle='--', linewidth=1.5, label=r"Optimal $\alpha_0 \geq 4.1$")
    axes[0].set_title(r"(a) Bifurcation Diagram for $x_1$ ($\alpha_0 \in [0.5, 6.5]$)")
    axes[0].set_xlabel(r"Control Parameter $\alpha_0$")
    axes[0].set_ylabel(r"$x_1$")
    axes[0].set_xlim([0.5, 6.5])
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc="upper left")
    
    axes[1].plot(a0_range, le1, label=r"$\lambda_1$", color="#C0392B", lw=1.2)
    axes[1].plot(a0_range, le2, label=r"$\lambda_2$", color="#E67E22", lw=1.2)
    axes[1].plot(a0_range, le3, label=r"$\lambda_3$", color="#27AE60", lw=1.2)
    axes[1].plot(a0_range, le4, label=r"$\lambda_4$", color="#2980B9", lw=1.2)
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='-')
    axes[1].axvspan(2.0, 5.0, color='gray', alpha=0.15)
    axes[1].axvline(4.1, color='green', linestyle='--', linewidth=1.5)
    axes[1].set_title(r"(b) Four Largest Lyapunov Exponents")
    axes[1].set_xlabel(r"Control Parameter $\alpha_0$")
    axes[1].set_ylabel("Exponent Magnitude")
    axes[1].set_xlim([0.5, 6.5])
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(CHAOS_DIR / "bifurcation_lyapunov.pdf")
    plt.savefig(CHAOS_DIR / "bifurcation_lyapunov.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    axes[0].plot(a0_range, le_sum, color='#2980B9', lw=1.5)
    axes[0].axhline(-4.0, color='#C0392B', linestyle='--', label=r"Target $\sum \lambda \approx -4$")
    axes[0].axvspan(2.0, 5.0, color='gray', alpha=0.15, label="Standard Region [2.0, 5.0]")
    axes[0].set_title(r"(a) Dissipative Phase Space Divergence ($\sum_{i=1}^6 \lambda_i$)")
    axes[0].set_xlabel(r"Control Parameter $\alpha_0$")
    axes[0].set_ylabel(r"Sum of Lyapunov Exponents")
    axes[0].set_xlim([0.5, 6.5])
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower left")
    
    axes[1].plot(a0_range, d_ky, color='#8E44AD', lw=1.5)
    axes[1].axvspan(2.0, 5.0, color='gray', alpha=0.15, label="Standard Region [2.0, 5.0]")
    axes[1].set_title(r"(b) Kaplan-Yorke Dimension ($D_{KY}$)")
    axes[1].set_xlabel(r"Control Parameter $\alpha_0$")
    axes[1].set_ylabel(r"Fractal Dimension $D_{KY}$")
    axes[1].set_xlim([0.5, 6.5])
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(CHAOS_DIR / "dissipation_ky.pdf")
    plt.savefig(CHAOS_DIR / "dissipation_ky.png", dpi=300)
    plt.close()
    
    # Export Table
    table_file = CHAOS_DIR / "alpha_lyapunov_table.txt"
    with open(table_file, "w") as f:
        header = f"{'alpha_0':<10} | {'LE1':<10} | {'LE2':<10} | {'LE3':<10} | {'LE4':<10} | {'Sum(LE)':<12} | {'D_KY':<10}\n"
        f.write("=" * len(header.strip()) + "\n")
        f.write(header)
        f.write("-" * len(header.strip()) + "\n")
        
        # Output every 5th value to keep table concise
        for i in range(0, len(a0_range), 5):
            f.write(f"{a0_range[i]:<10.4f} | {le1[i]:<10.4f} | {le2[i]:<10.4f} | {le3[i]:<10.4f} | {le4[i]:<10.4f} | {le_sum[i]:<12.4f} | {d_ky[i]:<10.4f}\n")
        f.write("=" * len(header.strip()) + "\n")

# =========================================================================
# 4. Extended Analysis for Mu Parameter ([0.1, 5.0])
# =========================================================================
def plot_extended_mu_analysis():
    print("[4/5] Computing Extended Mu Sweep ([0.1, 5.0]) Analysis...")
    ivs, params = get_baseline_params()
    
    m0_range = np.linspace(0.1, 5.0, 350)
    bifurcation_x5 = []
    le1, le2, le3, le4 = [], [], [], []
    le_sum = []
    
    length = 1200
    burn_in = 500
    
    for m0 in m0_range:
        p_copy = params.copy()
        p_copy["mu"][0] = m0
        
        X = ivs.copy()
        for _ in range(burn_in):
            X = map_step(X, p_copy)
            
        points = []
        Q = np.eye(6)
        LE = np.zeros(6)
        
        for _ in range(length):
            X = map_step(X, p_copy)
            points.append(X[4])
            
            J = jacobian(X, p_copy)
            Q, R = np.linalg.qr(J @ Q)
            LE += np.log(np.abs(np.diag(R)) + 1e-15)
            
        bifurcation_x5.append(points)
        
        LE /= length
        LE = np.sort(LE)[::-1]
        
        le1.append(LE[0]); le2.append(LE[1]); le3.append(LE[2]); le4.append(LE[3])
        le_sum.append(np.sum(LE))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    for i, m0 in enumerate(m0_range):
        axes[0].plot([m0] * length, bifurcation_x5[i], ',k', alpha=0.03)
    axes[0].axvline(1.0, color='red', linestyle='--', label=r"Critical Expansion $\mu_0 = 1.0$")
    axes[0].axvspan(2.0, 4.5, color='green', alpha=0.12, label=r"Operational Domain $\mu_0 \in [2.0, 4.5]$")
    axes[0].set_title(r"(a) Bifurcation of Auxiliary Variable $x_5$ vs. $\mu_0$")
    axes[0].set_xlabel(r"Control Parameter $\mu_0$")
    axes[0].set_ylabel(r"$x_5$")
    axes[0].set_xlim([0.1, 5.0])
    axes[0].legend(loc="upper left")
    
    axes[1].plot(m0_range, le1, label=r"$\lambda_1$", color="#C0392B", lw=1.2)
    axes[1].plot(m0_range, le2, label=r"$\lambda_2$", color="#E67E22", lw=1.2)
    axes[1].plot(m0_range, le3, label=r"$\lambda_3$", color="#27AE60", lw=1.2)
    axes[1].plot(m0_range, le4, label=r"$\lambda_4$", color="#2980B9", lw=1.2)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].axvline(1.0, color='red', linestyle='--')
    axes[1].axvspan(2.0, 4.5, color='green', alpha=0.12)
    axes[1].set_title(r"(b) Lyapunov Spectrum vs. $\mu_0$")
    axes[1].set_xlabel(r"Control Parameter $\mu_0$")
    axes[1].set_ylabel("Exponent Magnitude")
    axes[1].set_xlim([0.1, 5.0])
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(CHAOS_DIR / "mu_parameter_analysis.pdf")
    plt.savefig(CHAOS_DIR / "mu_parameter_analysis.png", dpi=300)
    plt.close()
    
    # Export Table
    table_file = CHAOS_DIR / "mu_lyapunov_table.txt"
    with open(table_file, "w") as f:
        header = f"{'mu_0':<10} | {'LE1':<10} | {'LE2':<10} | {'LE3':<10} | {'LE4':<10} | {'Sum(LE)':<12}\n"
        f.write("=" * len(header.strip()) + "\n")
        f.write(header)
        f.write("-" * len(header.strip()) + "\n")
        
        # Output every 5th value
        for i in range(0, len(m0_range), 5):
            f.write(f"{m0_range[i]:<10.4f} | {le1[i]:<10.4f} | {le2[i]:<10.4f} | {le3[i]:<10.4f} | {le4[i]:<10.4f} | {le_sum[i]:<12.4f}\n")
        f.write("=" * len(header.strip()) + "\n")

# =========================================================================
# 5. Sensitivity (Trajectory Separation & Bit Change Probability)
# =========================================================================

def plot_sensitivity():
    print("[5/5] Computing Sensitivity & Bit-Change Analysis...")
    ivs, params = get_baseline_params()
    
    length = 100
    delta = 1e-15
    ivs_pert = ivs.copy()
    ivs_pert[0] += delta
    
    X1, X2 = ivs.copy(), ivs_pert.copy()
    distances = []
    
    for _ in range(length):
        X1 = map_step(X1, params)
        X2 = map_step(X2, params)
        distances.append(np.linalg.norm(X1 - X2))
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    axes[0].semilogy(range(length), distances, color="#C0392B", marker="o", markersize=3, lw=1.0)
    axes[0].set_title(r"(a) Local Exponential Separation ($\Delta = 10^{-15}$)")
    axes[0].set_xlabel("Iteration (n)")
    axes[0].set_ylabel(r"Euclidean Distance $||X_n - X'_n||_2$")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    
    trials = 5000
    pt = np.zeros(length)
    
    for _ in range(trials):
        X1, params1 = get_baseline_params()
        X2 = X1.copy()
        X2[0] += delta
        
        diff_bits = np.zeros(length)
        for t in range(length):
            X1 = map_step(X1, params1)
            X2 = map_step(X2, params1)
            
            B1 = np.mod(np.floor(X1[:4] * 1e14), 256).astype(np.uint8)
            B2 = np.mod(np.floor(X2[:4] * 1e14), 256).astype(np.uint8)
            
            xor_val = np.bitwise_xor(B1, B2)
            hamming_weight = np.unpackbits(xor_val).sum()
            diff_bits[t] = hamming_weight / 32.0
            
        pt += diff_bits
        
    pt /= trials
    
    axes[1].plot(range(length), pt, color="#27AE60", lw=1.2)
    axes[1].axhline(0.5, color="red", linestyle="--", label="Optimal Limit = 0.5")
    axes[1].set_title(r"(b) Bit-Change Probability $P(t)$")
    axes[1].set_xlabel("Iteration (n)")
    axes[1].set_ylabel(r"$P(t)$")
    axes[1].set_ylim([0, 1.0])
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(CHAOS_DIR / "sensitivity.pdf")
    plt.savefig(CHAOS_DIR / "sensitivity.png", dpi=300)
    plt.close()
    
    # Export Combined Table for Sensitivity
    table_file = CHAOS_DIR / "sensitivity_table.txt"
    with open(table_file, "w") as f:
        header = f"{'Iteration (n)':<15} | {'Euclidean Distance':<25} | {'P(t)':<15}\n"
        f.write("=" * len(header.strip()) + "\n")
        f.write(header)
        f.write("-" * len(header.strip()) + "\n")
        for n in range(length):
            f.write(f"{n:<15} | {distances[n]:<25.6g} | {pt[n]:<15.4f}\n")
        f.write("=" * len(header.strip()) + "\n")
        
    print(f"Done. Vector/raster figures and all data tables generated in '{CHAOS_DIR}'")

if __name__ == "__main__":
    plot_phase_portrait()
    plot_stability_analysis()
    plot_bifurcation_and_lyapunov_alpha()
    plot_extended_mu_analysis()
    plot_sensitivity()