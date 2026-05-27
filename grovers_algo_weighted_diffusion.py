"""
Grover's Algorithm - Educational Interactive Implementation
===========================================================
Features added:
  - User can choose number of Grover iterations manually
  - Or use the optimal iteration count automatically
  - One iteration can run on every key press
  - Weighted (non-uniform) diffusion operator D_φ = 2|φ⟩⟨φ| − I
  - Three weight modes: gaussian prior, hot-block prior, manual
  - Correct optimal iteration count for weighted case
  - Side-by-side comparison plot of weighted vs uniform curves
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
# 3a.  STANDARD DIFFUSION  (unchanged)
# ─────────────────────────────────────────────
def diffusion(state: np.ndarray) -> np.ndarray:
    """
    Standard inversion-about-mean (uniform diffusion).

    This is the matrix form of  D = 2|ψ⟩⟨ψ| − I
    where |ψ⟩ = (1/√N)·(1,1,...,1) is the uniform superposition.

    Closed-form equivalent:
        a_i  →  2·mean(a) − a_i
    """
    mean_amplitude = np.mean(state)
    return 2 * mean_amplitude - state


# ─────────────────────────────────────────────
# 3b.  BIASED STATE PREPARATION
# ─────────────────────────────────────────────
def prepare_biased_state(weights: np.ndarray) -> np.ndarray:
    """
    Build the pivot state |φ⟩ from a raw weight vector.

    The weight vector encodes a probability distribution:
        p(x)  ∝  weights[x]      (non-negative, need not sum to 1)

    The amplitude of each basis state is:
        φ_x  =  √p(x)  =  √( weights[x] / Σ_x weights[x] )

    This guarantees  ⟨φ|φ⟩ = 1  and that  |⟨x*|φ⟩|² = p(x*)
    is the initial probability mass on the target.

    Parameters
    ----------
    weights : np.ndarray
        Non-negative weight for every basis state (need not be normalised).

    Returns
    -------
    phi : np.ndarray
        Unit-norm amplitude vector representing |φ⟩.
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative.")
    total = weights.sum()
    if total == 0:
        raise ValueError("Weight vector must not be all-zero.")
    probs = weights / total        # normalise → probability distribution
    phi   = np.sqrt(probs)         # amplitudes:  φ_x = √p(x)
    return phi


# ─────────────────────────────────────────────
# 3c.  WEIGHTED DIFFUSION   D_φ = 2|φ⟩⟨φ| − I
# ─────────────────────────────────────────────
def weighted_diffusion(state: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Non-uniform diffusion: reflection about the arbitrary pivot |φ⟩.

    Matrix form:    D_φ = 2|φ⟩⟨φ| − I
    Applied to |ψ⟩:
        D_φ|ψ⟩ = 2·⟨φ|ψ⟩·|φ⟩ − |ψ⟩

    where  overlap = ⟨φ|ψ⟩ = Σ_x φ_x · ψ_x   (real dot product).

    Special case — uniform φ = 1/√N · (1,...,1):
        overlap = (1/√N)·Σ_x ψ_x = √N · mean(ψ)
        D_φ|ψ⟩ = 2·(√N·mean)·(1/√N) − |ψ⟩ = 2·mean − |ψ⟩
    This is exactly the standard diffusion(), confirming correctness.

    Parameters
    ----------
    state : np.ndarray  — current amplitude vector |ψ⟩
    phi   : np.ndarray  — unit-norm pivot state |φ⟩

    Returns
    -------
    np.ndarray — amplitude vector after reflection about |φ⟩
    """
    overlap = np.dot(phi, state)       # ⟨φ|ψ⟩  (scalar)
    return 2.0 * overlap * phi - state


# ─────────────────────────────────────────────
# 3d.  WEIGHT PRESETS
# ─────────────────────────────────────────────
def make_gaussian_weights(N: int, center: int, sigma: float) -> np.ndarray:
    """
    Gaussian prior centred at `center` with std-dev `sigma` (in index units).
    Useful when you believe the target is near a known region of the database.

        w(x) = exp( −(x − center)² / (2·σ²) )
    """
    indices = np.arange(N)
    return np.exp(-0.5 * ((indices - center) / sigma) ** 2)


def make_block_weights(N: int, hot_start: int, hot_end: int,
                       hot_weight: float = 10.0) -> np.ndarray:
    """
    Flat prior with a 'hot block': indices [hot_start, hot_end) get
    weight `hot_weight`; all others get weight 1.
    Useful when the target is believed to be in a specific subspace.
    """
    weights = np.ones(N)
    weights[hot_start:hot_end] = hot_weight
    return weights


# ─────────────────────────────────────────────
# 4.  GROVER'S ALGORITHM  (standard + weighted)
# ─────────────────────────────────────────────
def grovers_algorithm(
    n_qubits: int,
    target: int,
    iterations: int = None,
    use_optimal: bool = True,
    step_mode: bool = False,
    verbose: bool = True,
    weights: np.ndarray = None,
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
        Number of Grover iterations.
        If None and use_optimal=True, optimal count is computed automatically.
    use_optimal : bool
        Whether to compute and use the theoretically optimal iteration count.
    step_mode  : bool
        If True, each iteration happens after a key press.
    verbose    : bool
        Print detailed output.
    weights    : np.ndarray or None
        Optional non-negative weight vector of length N = 2^n_qubits.
          None   → standard uniform Grover (original behaviour, unchanged).
          array  → weighted diffusion D_φ = 2|φ⟩⟨φ| − I is used,
                   where |φ⟩ is built from `weights` via prepare_biased_state().

    Returns
    -------
    probabilities_history : list of np.ndarray
    measured_state        : int
    final_state           : np.ndarray
    phi                   : np.ndarray  (the pivot state used)
    """
    N = 2 ** n_qubits
    assert 0 <= target < N, f"Target must be in [0, {N - 1}]"

    # ── Build pivot state |φ⟩ ──────────────────────────────────────────
    if weights is not None:
        if len(weights) != N:
            raise ValueError(f"weights must have length N={N}, got {len(weights)}.")
        phi = prepare_biased_state(weights)
        mode_label = "weighted (non-uniform)"
    else:
        phi = np.ones(N) / np.sqrt(N)    # uniform: φ_x = 1/√N
        mode_label = "uniform (standard Grover)"

    # ── Optimal iteration count ────────────────────────────────────────
    # General formula:  t* = floor( π / (4·arcsin(√p₀)) )
    # p₀ = |⟨x*|φ⟩|² = φ[target]²   is the initial probability on the target.
    #
    # Derivation: the algorithm rotates the state in the 2D subspace
    # {|x*⟩, |φ_⊥⟩} by angle 2θ per iteration, starting at angle θ
    # from the bad subspace. Peak success at angle π/2, so we need
    # (2k+1)θ = π/2  →  k = (π/4θ) − 1/2, rounded to nearest integer.
    #
    # For uniform φ: p₀ = 1/N → θ = arcsin(1/√N) ≈ 1/√N → t* ≈ (π/4)√N  ✓
    p0    = float(phi[target] ** 2)
    theta = math.asin(math.sqrt(p0))
    optimal_iters = max(1, math.floor(math.pi / (4.0 * theta)))

    if use_optimal or iterations is None:
        total_iterations = optimal_iters
    else:
        total_iterations = iterations

    if verbose:
        print("=" * 62)
        print("         G R O V E R ' S   A L G O R I T H M")
        print("=" * 62)
        print(f"  Qubits             : {n_qubits}")
        print(f"  Search space N     : {N}")
        print(f"  Target state       : |{target:0{n_qubits}b}⟩  (index {target})")
        print(f"  Diffusion mode     : {mode_label}")
        print(f"  Initial P(target)  : {p0:.6f}  [= φ[target]²]")
        print(f"  Rotation angle θ   : {theta:.6f} rad")
        print(f"  Optimal iterations : {optimal_iters}  [= floor(π / 4θ)]")
        print(f"  Iterations used    : {total_iterations}")
        print(f"  Step mode          : {step_mode}")
        print("=" * 62)

    # ── Initialise: start from |φ⟩ ────────────────────────────────────
    # Both uniform and weighted Grover begin from their pivot state |φ⟩.
    # For uniform: |φ⟩ = H⊗n|0⟩.
    # For weighted: |φ⟩ is constructed from the weight vector.
    state = phi.copy()

    if verbose:
        probs = state ** 2
        print(f"\n[Init] Starting state |φ⟩")
        print(f"       P(target) = {probs[target]:.6f}  (initial prior weight)")
        print(f"       Max P(any) = {probs.max():.6f}  at index {int(probs.argmax())}")

    probabilities_history = [state ** 2]

    # ── Grover iterations ──────────────────────────────────────────────
    for i in range(1, total_iterations + 1):
        if step_mode:
            cmd = input(
                f"\nPress ENTER for iteration {i} (or 'q' to stop): "
            ).strip().lower()
            if cmd == "q":
                print("Execution stopped by user.")
                break

        # Oracle: phase-flip the marked state
        state = oracle(state, target)

        # Diffusion: reflect about |φ⟩
        if weights is not None:
            state = weighted_diffusion(state, phi)
        else:
            state = diffusion(state)

        probs = state ** 2
        probabilities_history.append(probs.copy())

        if verbose:
            print(f"\n[Iter {i:02d}] Oracle ✓  Diffusion ({mode_label}) ✓")
            print(f"  State vector         : {np.round(state, 4)}")
            print(f"  Amplitude of target  : {state[target]:+.6f}")
            print(f"  P(target)            : {probs[target]:.6f}")
            print(f"  P(others, avg)       : {np.mean(np.delete(probs, target)):.6f}")

    # ── Measurement ───────────────────────────────────────────────────
    measured = int(np.argmax(state ** 2))

    if verbose:
        final_prob = (state ** 2)[target]
        print("\n" + "=" * 62)
        print(f"  Measurement result  : |{measured:0{n_qubits}b}⟩  (index {measured})")
        print(f"  Correct target      : |{target:0{n_qubits}b}⟩  (index {target})")
        print(f"  P(target) at meas.  : {final_prob:.6f}")
        print(f"  Success             : {'YES' if measured == target else 'NO'}")
        print("=" * 62)

    return probabilities_history, measured, state, phi


# ─────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────
def plot_grover(
    probabilities_history: list,
    target: int,
    n_qubits: int,
    phi: np.ndarray = None,
    label: str = "weighted",
    compare_uniform: bool = True,
):
    """
    Three-panel figure:
      Left   – Pivot state |φ⟩² (the prior / initial distribution)
      Centre – Final measurement probabilities after all iterations
      Right  – P(target) vs iteration, with uniform Grover overlay

    Parameters
    ----------
    probabilities_history : list of np.ndarray
    target                : int
    n_qubits              : int
    phi                   : np.ndarray  — pivot state (for left panel)
    label                 : str         — legend label for the weighted curve
    compare_uniform       : bool        — overlay standard Grover on right panel
    """
    N = 2 ** n_qubits
    actual_iterations = len(probabilities_history) - 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Grover's Algorithm  |  {n_qubits} qubits, N={N}, target={target}",
        fontsize=13,
        fontweight="bold",
    )

    # ── Left: pivot state |φ⟩² ────────────────────────────────────────
    ax0 = axes[0]
    if phi is not None:
        phi_probs = phi ** 2
        colors0 = ["crimson" if i == target else "mediumpurple" for i in range(N)]
        ax0.bar(range(N), phi_probs, color=colors0, edgecolor="black", linewidth=0.4)
        ax0.axhline(1 / N, color="gray", linestyle="--", linewidth=0.8,
                    label="1/N uniform")
        ax0.set_title("Prior  |φ⟩²  (initial distribution)")
    else:
        ax0.set_title("No prior  (uniform)")
    ax0.set_xlabel("State index")
    ax0.set_ylabel("Probability")
    ax0.set_xlim(-0.5, N - 0.5)
    ax0.set_ylim(0, min(1.05, max((phi ** 2).max() * 1.15, 2 / N)))
    ax0.legend(fontsize=8)

    # ── Centre: final probabilities ────────────────────────────────────
    ax1 = axes[1]
    final_probs = probabilities_history[-1]
    colors1 = ["crimson" if i == target else "steelblue" for i in range(N)]
    ax1.bar(range(N), final_probs, color=colors1, edgecolor="black", linewidth=0.4)
    ax1.set_xlabel("State index")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"Final probabilities  (after {actual_iterations} iterations)")
    ax1.set_xlim(-0.5, N - 0.5)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(1 / N, color="gray", linestyle="--", linewidth=0.8, label="1/N uniform")
    ax1.legend(fontsize=8)

    # ── Right: P(target) vs iteration ─────────────────────────────────
    ax2 = axes[2]
    target_probs = [ph[target] for ph in probabilities_history]
    ax2.plot(
        range(len(target_probs)),
        target_probs,
        marker="o",
        color="crimson",
        linewidth=2,
        markersize=6,
        label=label,
    )

    if compare_uniform:
        # Closed-form uniform Grover curve:  P(target, k) = sin²((2k+1)·θ_u)
        theta_u = math.asin(1.0 / math.sqrt(N))
        uniform_iters = max(1, math.floor(math.pi / (4.0 * theta_u)))
        uniform_curve = [
            math.sin((2 * k + 1) * theta_u) ** 2
            for k in range(uniform_iters + 1)
        ]
        ax2.plot(
            range(len(uniform_curve)),
            uniform_curve,
            marker="s",
            color="steelblue",
            linewidth=2,
            markersize=5,
            linestyle="--",
            label="uniform (standard)",
        )

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("P(target state)")
    ax2.set_title("Target probability vs iteration")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1 / N, color="gray", linestyle="--", linewidth=0.8, label="1/N baseline")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("grover_results.png", dpi=150, bbox_inches="tight")
    print("\n[Plot saved] → grover_results.png")
    plt.show()


# ─────────────────────────────────────────────
# 6.  CLASSICAL vs QUANTUM COMPARISON
# ─────────────────────────────────────────────
def complexity_comparison(phi: np.ndarray = None, target: int = None):
    """
    Print classical vs quantum complexity table.
    If phi and target are provided, also prints the weighted iteration count.
    """
    has_weighted = (phi is not None and target is not None)

    header = f"  {'Qubits':>6}  {'N':>8}  {'Classical O(N)':>14}  {'Grover O(√N)':>13}"
    if has_weighted:
        header += f"  {'Weighted':>10}"
    print("\n" + "=" * (len(header) + 2))
    print("  Classical vs Quantum Search Complexity")
    print("=" * (len(header) + 2))
    print(header)
    print("-" * (len(header) + 2))

    for q in [2, 4, 6, 8, 10, 20]:
        N    = 2 ** q
        cls  = N
        grov = math.floor((math.pi / 4) * math.sqrt(N))
        row  = f"  {q:>6}  {N:>8}  {cls:>14}  {grov:>13}"
        if has_weighted and len(phi) == N:
            p0 = float(phi[target] ** 2)
            theta = math.asin(math.sqrt(p0))
            wt = max(1, math.floor(math.pi / (4.0 * theta)))
            row += f"  {wt:>10}"
        print(row)

    print("=" * (len(header) + 2))


# ─────────────────────────────────────────────
# 7.  USER INPUT
# ─────────────────────────────────────────────
def get_user_input():
    n_qubits = int(input("Enter number of qubits: "))
    N = 2 ** n_qubits

    target = int(input(f"Enter target state index (0 to {N - 1}): "))
    while not (0 <= target < N):
        target = int(input(f"Invalid. Enter target state index (0 to {N - 1}): "))

    # ── Diffusion mode ─────────────────────────────────────────────────
    print("\nDiffusion mode:")
    print("  [1] Standard uniform Grover (original behaviour)")
    print("  [2] Weighted — Gaussian prior around a centre index")
    print("  [3] Weighted — Hot-block prior (elevated subspace)")
    print("  [4] Weighted — Manual weights (you enter each value)")
    mode = input("Choose mode (1/2/3/4): ").strip()

    weights      = None
    weight_label = "uniform"

    if mode == "2":
        center = int(input(f"  Gaussian centre index (0 to {N-1}): "))
        sigma  = float(input(f"  Sigma / std-dev in index units (e.g. {max(1, N//8)}): "))
        weights      = make_gaussian_weights(N, center, sigma)
        weight_label = f"gaussian(c={center}, σ={sigma})"

    elif mode == "3":
        print(f"  Indices in [start, end) receive elevated weight.")
        hot_start  = int(input(f"  Hot block start (0–{N-1}): "))
        hot_end    = int(input(f"  Hot block end   (1–{N}):   "))
        hot_weight = float(input(f"  Weight multiplier (e.g. 10): "))
        weights      = make_block_weights(N, hot_start, hot_end, hot_weight)
        weight_label = f"block([{hot_start},{hot_end}), ×{hot_weight})"

    elif mode == "4":
        print(f"  Enter {N} non-negative weights (space-separated):")
        raw          = input("  Weights: ").strip().split()
        weights      = np.array([float(v) for v in raw])
        weight_label = "manual"

    # ── Iteration count ────────────────────────────────────────────────
    choice = input("\nUse optimal iterations? (y/n): ").strip().lower()
    if choice == "y":
        use_optimal = True
        iterations  = None
    else:
        use_optimal = False
        iterations  = int(input("Enter number of Grover iterations: "))

    step_choice = input("Run one iteration per key press? (y/n): ").strip().lower()
    step_mode   = (step_choice == "y")

    return n_qubits, target, iterations, use_optimal, step_mode, weights, weight_label


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    (n_qubits, target, iterations,
     use_optimal, step_mode, weights, weight_label) = get_user_input()

    history, result, final_state, phi = grovers_algorithm(
        n_qubits=n_qubits,
        target=target,
        iterations=iterations,
        use_optimal=use_optimal,
        step_mode=step_mode,
        verbose=True,
        weights=weights,
    )

    plot_grover(
        history,
        target,
        n_qubits,
        phi=phi,
        label=weight_label,
        compare_uniform=(weights is not None),
    )

    complexity_comparison(phi=phi, target=target)