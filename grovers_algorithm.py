"""
Grover's Algorithm - Educational Implementation
================================================
A step-by-step simulation of Grover's quantum search algorithm
using NumPy (no quantum framework required).

Concepts demonstrated:
  - Superposition via Hadamard gate
  - Oracle (phase flip on target state)
  - Diffusion (inversion about the mean)
  - Probability amplitude evolution
  - Optimal number of iterations: floor(pi/4 * sqrt(N))
"""

import numpy as np
import math
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 1.  HADAMARD GATE  (creates equal superposition)
# ─────────────────────────────────────────────
def hadamard_transform(n_qubits: int) -> np.ndarray:
    """
    Return the 2^n × 2^n Hadamard matrix.
    H⊗n |0⟩ produces a uniform superposition over all 2^n states.
    """
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    # copy() is used so that any future changes to result don't accidentally modify the original H.
    result = H.copy()
    for _ in range(n_qubits - 1):
        # `np.kron` computes the **Kronecker product** (tensor product)
        result = np.kron(result, H)
    return result


# PARAMETER:
#   state vector (a 1D NumPy array of length N) 
def apply_hadamard(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply Hadamard transform to a state vector."""
    H = hadamard_transform(n_qubits)
    # '@' operator performs matrix multiplication. 
    # This multiplies the 2ⁿ × 2ⁿ Hadamard matrix by the 2ⁿ length state vector, producing a new 2ⁿ length state vector. 
    # Mathematically this is H|ψ⟩.
    return H @ state


# ─────────────────────────────────────────────
# 2.  ORACLE  (marks the target with a phase flip)
# ─────────────────────────────────────────────
def oracle(state: np.ndarray, target: int) -> np.ndarray:
    """
    Phase-flip oracle:  |x⟩ → -|x⟩  if x == target
                        |x⟩ →  |x⟩  otherwise
    In matrix form this is I - 2|target⟩⟨target|
    """
    #|target⟩⟨target| — an outer product that creates a matrix which, 
    # when applied to any state vector, extracts only the target component
    marked = state.copy()
    marked[target] *= -1
    return marked


# ─────────────────────────────────────────────
# 3.  DIFFUSION OPERATOR  (inversion about the mean)
# ─────────────────────────────────────────────
def diffusion(state: np.ndarray) -> np.ndarray:
    """
    Grover diffusion operator:  2|s⟩⟨s| - I
    where |s⟩ is the uniform superposition.
    Geometrically: reflect amplitudes about their mean.
    """
    #np.mean() computes the arithmetic mean of all amplitudes in the state vector — adds them all up and divides by N.
    # After oracle has flipped the target's sign, this mean is slightly below the uniform value of 1/√N because one element is now negative.
    mean_amplitude = np.mean(state)
    return 2 * mean_amplitude - state          # element-wise


# ─────────────────────────────────────────────
# 4.  GROVER'S ALGORITHM
# ─────────────────────────────────────────────
def grovers_algorithm(n_qubits: int, target: int, time: int, verbose: bool = True):
    """
    Run Grover's algorithm on a search space of N = 2^n_qubits states.

    Parameters
    ----------
    n_qubits : int   Number of qubits (search space = 2^n_qubits).
    target   : int   Index of the marked / solution state.
    verbose  : bool  Print step-by-step amplitude information.

    Returns
    -------
    probabilities_history : list of np.ndarray  (one per iteration)
    measured_state        : int
    """
    N = 2 ** n_qubits
    assert 0 <= target < N, f"Target must be in [0, {N-1}]"

    optimal_iters = math.floor(math.pi / 4 * math.sqrt(N))
    expected_optimal_iters=time

    if verbose:
        print("=" * 60)
        print("         G R O V E R ' S   A L G O R I T H M")
        print("=" * 60)
        print(f"  Qubits          : {n_qubits}")
        print(f"  Search space N  : {N}")
        #target — the integer value being formatted (e.g. 11)
        #0 — pad with zeros on the left instead of spaces
        #{n_qubits} — the total width of the output (e.g. 4 digits for 4 qubits)
        #b — convert the integer to binary representation
        print(f"  Target state    : |{target:0{n_qubits}b}⟩  (index {target})")
        print(f"  Target type  : {type(target)} ")
        print(f"  Optimal iters   : {optimal_iters}")
        print("=" * 60)

    # ── Step 1: Initialise |0⟩^⊗n ──────────────────────────────
    state = np.zeros(N)
    state[0] = 1.0

    # ── Step 2: Apply Hadamard → uniform superposition ──────────
    state = apply_hadamard(state, n_qubits)

    if verbose:
        probs = state ** 2
        print(f"\n[Init]  Uniform superposition (all amplitudes = {state[0]:.4f})")
        print(f"        P(target) = {probs[target]:.4f}  (= 1/N = {1/N:.4f})")

    probabilities_history = [state ** 2]

    # ── Step 3: Grover iterations ────────────────────────────────
    for i in range(1, expected_optimal_iters + 1):
        state = oracle(state, target)          # mark target
        state = diffusion(state)               # amplify marked state
        print(f"state after iteration {i}: {state}")
        probs = state ** 2
        probabilities_history.append(probs.copy())

        if verbose:
            print(f"\n[Iter {i:02d}]  Oracle ✓  Diffusion ✓")
            print(f"  Amplitude of target  : {state[target]:+.6f}")
            print(f"  P(target)            : {probs[target]:.6f}")
            print(f"  P(others, avg)       : {np.mean(np.delete(probs, target)):.6f}")

    # ── Step 4: Measurement (simulate collapse) ──────────────────
    measured = int(np.argmax(state ** 2))

    if verbose:
        print("\n" + "=" * 60)
        print(f"  Measurement result  : |{measured:0{n_qubits}b}⟩  (index {measured})")
        print(f"  Correct target      : |{target:0{n_qubits}b}⟩  (index {target})")
        print(f"  Success             : {'✅ YES' if measured == target else '❌ NO'}")
        print("=" * 60)
    #print(f"\n Target probability history {probabilities_history}")
    #print(f"\n Measured state index: {measured}")
    return probabilities_history, measured


# ─────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────
def plot_grover(probabilities_history: list, target: int,time:int, n_qubits: int):
    """
    Two-panel figure:
      Left  – Probability amplitudes across all iterations (target highlighted).
      Right – Target probability vs. iteration number.
    """
    N = 2 ** n_qubits
    #optimal_iters = len(probabilities_history) - 1
    optimal_iters = time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Grover's Algorithm  |  {n_qubits} qubits, N={N}, target={target}",
        fontsize=13, fontweight="bold"
    )
    # ── Left: bar chart of final probabilities ──────────────────
    ax = axes[0]
    final_probs = probabilities_history[-1]
    colours = ["crimson" if i == target else "steelblue" for i in range(N)]
    ax.bar(range(N), final_probs, color=colours, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("State index")
    ax.set_ylabel("Probability")
    ax.set_title(f"Final probabilities  (after {optimal_iters} iterations)")
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(1 / N, color="gray", linestyle="--", linewidth=0.8, label="1/N (uniform)")
    ax.legend()

    # ── Right: target probability over iterations ───────────────
    ax2 = axes[1]
    target_probs = [ph[target] for ph in probabilities_history]
    ax2.plot(range(len(target_probs)), target_probs, marker="o",
             color="crimson", linewidth=2, markersize=6)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("P(target state)")
    ax2.set_title("Target probability vs. Iteration")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1 / N, color="gray", linestyle="--", linewidth=0.8, label="1/N (uniform)")
    ax2.legend()

    plt.tight_layout()
    #plt.savefig("/mnt/user-data/outputs/grover_results.png", dpi=150, bbox_inches="tight")
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
        quantum   = math.floor(math.pi / 4 * math.sqrt(N))
        print(f"  {q:>6}  {N:>8}  {classical:>14}  {quantum:>14}")
    print("=" * 50)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Configuration ──────────────────────────────────────────
    N_QUBITS = 10       # 2^4 = 16 possible states
    TARGET   = 3         # the "needle" we're searching for
    #TIME    = 6
    N = 2 ** N_QUBITS
    TIME = 3*(math.floor(math.pi / 4 * math.sqrt(N)))

    # ── Run algorithm ──────────────────────────────────────────
    history, result = grovers_algorithm(N_QUBITS, TARGET, TIME, verbose=True)

    # ── Visualise ──────────────────────────────────────────────
    plot_grover(history, TARGET, TIME, N_QUBITS)

    # ── Classical vs Quantum speedup table ─────────────────────
    complexity_comparison()