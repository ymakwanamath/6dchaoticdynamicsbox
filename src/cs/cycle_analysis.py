import os
import secrets
import numpy as np

from cs.key_schedule import derive_chaotic_parameters


def get_next_state(state: np.ndarray, params: dict) -> np.ndarray:
    x1, x2, x3, x4, x5, x6 = state
    a = params["alpha"]
    b = params["beta"]
    m = params["mu"]
    d = params["delta"]
    g = params["gamma"]
    two_pi = 2.0 * np.pi

    nx1 = (a[0] * x1 + b[0] * x2 + g * x5) % 1.0
    nx2 = (a[1] * x2 + b[1] * x3 + g * x6) % 1.0
    nx3 = (a[2] * x3 + b[2] * x4 + g * x5) % 1.0
    nx4 = (a[3] * x4 + b[3] * x1 + g * x6) % 1.0

    nx5 = (m[0] * x5 + d[0] * np.sin(two_pi * x1) * np.sin(two_pi * x3)) % 1.0
    nx6 = (m[1] * x6 + d[1] * np.sin(two_pi * x2) * np.sin(two_pi * x4)) % 1.0

    return np.array([nx1, nx2, nx3, nx4, nx5, nx6], dtype=np.float64)


def floyd_cycle_detection(initial_state: np.ndarray, params: dict, max_steps: int = 1_000_000) -> tuple[int | None, int | None]:
    tortoise = get_next_state(initial_state, params)
    hare = get_next_state(get_next_state(initial_state, params), params)
    steps = 1

    # Phase 1: Cycle detection using bitwise equality
    while not np.array_equal(tortoise, hare) and steps < max_steps:
        tortoise = get_next_state(tortoise, params)
        hare = get_next_state(get_next_state(hare, params), params)
        steps += 1

    if steps >= max_steps:
        return None, None  # No cycle found within step threshold

    # Phase 2: Identify transient length (mu)
    mu = 0
    tortoise = initial_state
    while not np.array_equal(tortoise, hare):
        tortoise = get_next_state(tortoise, params)
        hare = get_next_state(hare, params)
        mu += 1

    # Phase 3: Identify cycle length (lambda)
    lam = 1
    hare = get_next_state(tortoise, params)
    while not np.array_equal(tortoise, hare):
        hare = get_next_state(hare, params)
        lam += 1

    return mu, lam


def main():
    output_dir = os.path.join("data", "analysis", "chaos")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "periodicity.txt")

    num_trials = 100
    max_steps = 1_000_000

    print(f"Running Floyd's cycle detection ({num_trials} trials, max_steps={max_steps})...")

    detected_cycles = []
    transient_lengths = []
    cycle_lengths = []

    for trial in range(1, num_trials + 1):
        random_seed = secrets.token_bytes(32)
        ivs, params = derive_chaotic_parameters(random_seed)
        init_state = np.array(ivs, dtype=np.float64)

        mu, lam = floyd_cycle_detection(init_state, params, max_steps=max_steps)

        if lam is not None:
            detected_cycles.append(trial)
            transient_lengths.append(mu)
            cycle_lengths.append(lam)
            print(f"Trial {trial}/{num_trials}: Cycle detected! mu={mu}, lambda={lam}")
        else:
            print(f"Trial {trial}/{num_trials}: No cycle detected up to {max_steps} iterations.")

    # Write metrics to periodicity.txt
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Finite-Precision Periodicity Analysis ===\n")
        f.write(f"Total Trials: {num_trials}\n")
        f.write(f"Max Steps Bound: {max_steps}\n")
        f.write(f"Cycles Detected: {len(detected_cycles)} / {num_trials}\n\n")

        if detected_cycles:
            f.write(f"Minimum Cycle Length: {min(cycle_lengths)}\n")
            f.write(f"Maximum Cycle Length: {max(cycle_lengths)}\n")
            f.write(f"Average Cycle Length: {np.mean(cycle_lengths):.2f}\n")
            f.write(f"Average Transient Length: {np.mean(transient_lengths):.2f}\n")
        else:
            f.write(f"Result: All orbits exceeded {max_steps} steps without entering a closed limit cycle.\n")
            f.write("State degradation bound: Periodicity T > max_steps under IEEE-754 float64 arithmetic.\n")

    print(f"\nAnalysis complete. Results written to: {output_file}")


if __name__ == "__main__":
    main()