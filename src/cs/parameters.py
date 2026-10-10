import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Configuration
BASELINE = {'alpha': 4.1, 'beta': 1.3, 'gamma': 2.1, 'mu': 3.7, 'delta': 2.9}

SCAN_RANGES = {
    'alpha': np.linspace(1.1, 5.0, 150),
    'beta':  np.linspace(1.1, 5.0, 150),
    'gamma': np.linspace(1.1, 5.0, 150),
    'mu':    np.linspace(1.1, 5.0, 150),
    'delta': np.linspace(2.1, 5.0, 150)
}

MIN_LYAPUNOV = 0.05
INTEGER_EXCLUSION_MARGIN = 0.05
OUTPUT_FILE = "cryptographic_parameter_ranges.txt"

def iterate(x, p):
    a, b, g, m, d = p['alpha'], p['beta'], p['gamma'], p['mu'], p['delta']
    nx1 = (a*x[0] + b*x[1] + g*x[4]) % 1.0
    nx2 = (a*x[1] + b*x[2] + g*x[5]) % 1.0
    nx3 = (a*x[2] + b*x[3] + g*x[4]) % 1.0
    nx4 = (a*x[3] + b*x[0] + g*x[5]) % 1.0
    nx5 = (m*x[4] + d*np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[2])) % 1.0
    nx6 = (m*x[5] + d*np.sin(2*np.pi*x[1])*np.sin(2*np.pi*x[3])) % 1.0
    return np.array([nx1, nx2, nx3, nx4, nx5, nx6])

def jacobian(x, p):
    a, b, g, m, d = p['alpha'], p['beta'], p['gamma'], p['mu'], p['delta']
    J = np.zeros((6, 6))
    J[0,0] = a; J[0,1] = b; J[0,4] = g
    J[1,1] = a; J[1,2] = b; J[1,5] = g
    J[2,2] = a; J[2,3] = b; J[2,4] = g
    J[3,3] = a; J[3,0] = b; J[3,5] = g
    J[4,0] = d * 2*np.pi * np.cos(2*np.pi*x[0]) * np.sin(2*np.pi*x[2])
    J[4,2] = d * 2*np.pi * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[2])
    J[4,4] = m
    J[5,1] = d * 2*np.pi * np.cos(2*np.pi*x[1]) * np.sin(2*np.pi*x[3])
    J[5,3] = d * 2*np.pi * np.sin(2*np.pi*x[1]) * np.cos(2*np.pi*x[3])
    J[5,5] = m
    return J

def get_lyapunov_spectrum(p, n_iter=800):
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    Q = np.eye(6)
    logs = np.zeros(6)
    for _ in range(200): # Transient
        x = iterate(x, p)
    for _ in range(n_iter):
        J = jacobian(x, p)
        x = iterate(x, p)
        Q, R = np.linalg.qr(J @ Q)
        logs += np.log(np.abs(np.diag(R)) + 1e-12)
    return logs / n_iter

def is_near_integer(val, margin):
    return abs(val - round(val)) < margin

def analyze_and_plot():
    with open(OUTPUT_FILE, 'w') as f:
        # Write Methodology
        f.write("METHODOLOGY\n")
        f.write("-" * 50 + "\n")
        f.write("This script determines mathematically sound parameter ranges for a 6D modulo-1 chaotic map.\n")
        f.write("1. Baselines: Variables not currently being scanned are held at non-integer values strictly > 1.\n")
        f.write("2. Integer Exclusion: Exact integers (and values within a 0.05 margin) are explicitly rejected, \n")
        f.write("   as integer multipliers in modulo-1 maps cause finite-precision orbit collapse.\n")
        f.write("3. Lyapunov Requirement: To guarantee hyperchaos, all 6 Lyapunov exponents must be > 0.05.\n")
        f.write("Reality Check: These ranges ensure chaotic stretching and folding, which is necessary but not \n")
        f.write("sufficient for total cryptographic security. The final mapped bitstream must still pass NIST STS.\n")
        f.write("-" * 50 + "\n\n")

        for param_name, values in SCAN_RANGES.items():
            print(f"Analyzing {param_name}...")
            
            valid_ranges = []
            in_range = False
            start_val = None
            
            spectra = []
            bifurcation_x = []
            bifurcation_y = []
            
            for v in values:
                test_params = BASELINE.copy()
                test_params[param_name] = v
                
                # Lyapunov Data
                spectrum = get_lyapunov_spectrum(test_params)
                spectra.append(spectrum)
                min_le = np.min(spectrum)
                
                # Range filtering logic
                if is_near_integer(v, INTEGER_EXCLUSION_MARGIN):
                    if in_range:
                        valid_ranges.append((start_val, prev_v))
                        in_range = False
                elif min_le > MIN_LYAPUNOV:
                    if not in_range:
                        start_val = v
                        in_range = True
                else:
                    if in_range:
                        valid_ranges.append((start_val, prev_v))
                        in_range = False
                prev_v = v
                
                # Bifurcation Data (keeping last 50 iterations of x1)
                x_bif = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
                for _ in range(200): x_bif = iterate(x_bif, test_params)
                for _ in range(50):
                    x_bif = iterate(x_bif, test_params)
                    bifurcation_x.append(v)
                    bifurcation_y.append(x_bif[0])
                
            if in_range:
                valid_ranges.append((start_val, prev_v))
            
            # File Output
            f.write(f"Valid continuous ranges for {param_name}:\n")
            for r in valid_ranges:
                if r[1] - r[0] > 0.1: 
                    f.write(f"  [{r[0]:.3f}, {r[1]:.3f}]\n")
            f.write("\n")
            
            # Plot Lyapunov Spectrum
            spectra = np.array(spectra)
            plt.figure(figsize=(8, 5))
            for i in range(6):
                plt.plot(values, spectra[:, i], label=f'LE {i+1}')
            plt.axhline(MIN_LYAPUNOV, color='r', linestyle='--', label='Min Threshold')
            plt.title(f'Lyapunov Spectrum vs {param_name}')
            plt.xlabel(param_name)
            plt.ylabel('Lyapunov Exponent')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'lyapunov_vs_{param_name}.png')
            plt.close()
            
            # Plot Bifurcation Diagram
            plt.figure(figsize=(8, 5))
            plt.plot(bifurcation_x, bifurcation_y, ',k', alpha=0.5)
            plt.title(f'Bifurcation Diagram (x1) vs {param_name}')
            plt.xlabel(param_name)
            plt.ylabel('x1')
            plt.savefig(f'bifurcation_vs_{param_name}.png')
            plt.close()

    print(f"Analysis complete. Results written to {OUTPUT_FILE}.")
    print("Diagrams saved as .png files in the current directory.")

if __name__ == "__main__":
    analyze_and_plot()