"""
Grover's Algorithm - Educational Interactive Implementation
===========================================================
Features added:
  - User can choose number of Grover iterations manually
  - Or use the optimal iteration count automatically
  - One iteration can run on every key press
"""

import numpy as np
import math
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1.  HADAMARD GATE
# ─────────────────────────────────────────────
def hadamard_transform(n_qubits: int) -> np.ndarray:
    """
    Return the 2^n × 2^n Hadamard matrix.
    H⊗n |0⟩ produces a uniform superposition over all 2^n states.
    """
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    if n_qubits == 1:
        return H

    result = H.copy()
    for _ in range(n_qubits - 1):
        result = np.kron(result, H)
    return result


def apply_hadamard(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply Hadamard transform to a state vector."""
    H = hadamard_transform(n_qubits)
    return H @ state


# ─────────────────────────────────────────────
# 2.  ORACLE
# ─────────────────────────────────────────────
def oracle(state: np.ndarray, target: int) -> np.ndarray:
    """
    Phase-flip oracle:
        |x⟩ → -|x⟩ if x == target
        |x⟩ →  |x⟩ otherwise
    """
    marked = state.copy()
    marked[target] *= -1
    return marked


# ─────────────────────────────────────────────
# 3.  DIFFUSION OPERATOR
# ─────────────────────────────────────────────
def diffusion(state: np.ndarray) -> np.ndarray:
    """
    Inversion about the mean:
        a_i → 2*mean - a_i
    """
    mean_amplitude = np.mean(state)
    return 2 * mean_amplitude - state


# ─────────────────────────────────────────────
# 4.  GROVER'S ALGORITHM
# ─────────────────────────────────────────────
def grovers_algorithm(
    n_qubits: int,
    target: int,
    iterations: int = None,
    use_optimal: bool = True,
    step_mode: bool = False,
    verbose: bool = True
):
    """
    Run Grover's algorithm on N = 2^n_qubits states.

    Parameters
    ----------
    n_qubits   : int
        Number of qubits.
    target     : int
        Marked state index.
    iterations : int or None
        Number of Grover iterations to apply.
        If None and use_optimal=True, optimal count is used.
    use_optimal : bool
        Whether to use optimal iteration count automatically.
    step_mode  : bool
        If True, each iteration happens after a key press.
    verbose    : bool
        Print detailed output.

    Returns
    -------
    probabilities_history : list of np.ndarray
    measured_state        : int
    final_state           : np.ndarray
    """
    N = 2 ** n_qubits
    assert 0 <= target < N, f"Target must be in [0, {N - 1}]"

    optimal_iters = math.floor((math.pi / 4) * math.sqrt(N))

    if use_optimal or iterations is None:
        total_iterations = optimal_iters
    else:
        total_iterations = iterations

    if verbose:
        print("=" * 60)
        print("         G R O V E R ' S   A L G O R I T H M")
        print("=" * 60)
        print(f"  Qubits             : {n_qubits}")
        print(f"  Search space N     : {N}")
        print(f"  Target state       : |{target:0{n_qubits}b}⟩  (index {target})")
        print(f"  Optimal iterations : {optimal_iters}")
        print(f"  Iterations used    : {total_iterations}")
        print(f"  Step mode          : {step_mode}")
        print("=" * 60)

    # Step 1: Initialize |0...0>
    state = np.zeros(N)
    state[0] = 1.0

    # Step 2: Apply Hadamard
    state = apply_hadamard(state, n_qubits)

    if verbose:
        probs = state ** 2
        print(f"\n[Init] Uniform superposition")
        print(f"       Amplitude of each state = {state[0]:.6f}")
        print(f"       P(target) = {probs[target]:.6f} (= 1/N = {1 / N:.6f})")

    probabilities_history = [state ** 2]

    # Step 3: Grover iterations
    for i in range(1, total_iterations + 1):
        if step_mode:
            cmd = input(f"\nPress ENTER for iteration {i} (or type 'q' to stop): ").strip().lower()
            if cmd == "q":
                print("Execution stopped by user.")
                break

        state = oracle(state, target)
        state = diffusion(state)

        probs = state ** 2
        probabilities_history.append(probs.copy())

        if verbose:
            print(f"\n[Iter {i:02d}] Oracle ✓  Diffusion ✓")
            print(f"  State vector         : {np.round(state, 6)}")
            print(f"  Amplitude of target  : {state[target]:+.6f}")
            print(f"  P(target)            : {probs[target]:.6f}")
            print(f"  P(others, avg)       : {np.mean(np.delete(probs, target)):.6f}")

    # Step 4: Measurement
    measured = int(np.argmax(state ** 2))

    if verbose:
        print("\n" + "=" * 60)
        print(f"  Measurement result  : |{measured:0{n_qubits}b}⟩  (index {measured})")
        print(f"  Correct target      : |{target:0{n_qubits}b}⟩  (index {target})")
        print(f"  Success             : {'✅ YES' if measured == target else '❌ NO'}")
        print("=" * 60)

    return probabilities_history, measured, state


# ─────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────
def plot_grover(probabilities_history: list, target: int, n_qubits: int):
    """
    Two-panel figure:
      Left  – Final probabilities
      Right – Target probability vs. iteration
    """
    N = 2 ** n_qubits
    actual_iterations = len(probabilities_history) - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Grover's Algorithm | {n_qubits} qubits, N={N}, target={target}",
        fontsize=13,
        fontweight="bold"
    )

    # Left plot
    ax = axes[0]
    final_probs = probabilities_history[-1]
    colors = ["crimson" if i == target else "steelblue" for i in range(N)]
    ax.bar(range(N), final_probs, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("State index")
    ax.set_ylabel("Probability")
    ax.set_title(f"Final probabilities (after {actual_iterations} iterations)")
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / N, color="gray", linestyle="--", linewidth=0.8, label="1/N (uniform)")
    ax.legend()

    # Right plot
    ax2 = axes[1]
    target_probs = [ph[target] for ph in probabilities_history]
    ax2.plot(range(len(target_probs)), target_probs, marker="o", color="crimson", linewidth=2, markersize=6)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("P(target state)")
    ax2.set_title("Target probability vs Iteration")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1 / N, color="gray", linestyle="--", linewidth=0.8, label="1/N (uniform)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("grover_results.png", dpi=150, bbox_inches="tight")
    print("\n[Plot saved] → grover_results.png")
    plt.show()


# ─────────────────────────────────────────────
# 6.  CLASSICAL vs QUANTUM COMPARISON
# ─────────────────────────────────────────────
def complexity_comparison():
    print("\n" + "=" * 50)
    print("  Classical vs Quantum Search Complexity")
    print("=" * 50)
    print(f"  {'Qubits':>6}  {'N':>8}  {'Classical O(N)':>14}  {'Quantum O(√N)':>14}")
    print("-" * 50)
    for q in [2, 4, 6, 8, 10, 20]:
        N = 2 ** q
        classical = N
        quantum = math.floor((math.pi / 4) * math.sqrt(N))
        print(f"  {q:>6}  {N:>8}  {classical:>14}  {quantum:>14}")
    print("=" * 50)


# ─────────────────────────────────────────────
# 7.  USER INPUT
# ─────────────────────────────────────────────
def get_user_input():
    n_qubits = int(input("Enter number of qubits: "))
    N = 2 ** n_qubits

    target = int(input(f"Enter target state index (0 to {N - 1}): "))
    while not (0 <= target < N):
        target = int(input(f"Invalid target. Enter target state index (0 to {N - 1}): "))

    choice = input("Use optimal iterations? (y/n): ").strip().lower()

    if choice == "y":
        use_optimal = True
        iterations = None
    else:
        use_optimal = False
        iterations = int(input("Enter number of Grover iterations: "))

    step_choice = input("Run one iteration on each key press? (y/n): ").strip().lower()
    step_mode = (step_choice == "y")

    return n_qubits, target, iterations, use_optimal, step_mode


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    n_qubits, target, iterations, use_optimal, step_mode = get_user_input()

    history, result, final_state = grovers_algorithm(
        n_qubits=n_qubits,
        target=target,
        iterations=iterations,
        use_optimal=use_optimal,
        step_mode=step_mode,
        verbose=True
    )

    plot_grover(history, target, n_qubits)
    complexity_comparison()