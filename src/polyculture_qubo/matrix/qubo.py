"""QUBO matrix construction for polyculture species selection.

Constructs the Q matrix that encodes the full optimization problem:

    Minimize: x^T Q x

Where x_i ∈ {0, 1} indicates whether species i is selected.

The Q matrix combines three objective components and constraint penalties:

    Q_ij = -α * J_ij  -  β * N_ij  -  γ * D_ij  +  2λ   (off-diagonal, i < j)
    Q_ii = -α * h_i   +  λ(1 - 2k)                       (diagonal)

Where:
    J_ij = pairwise interaction coefficient (LER-derived)
    N_ij = pairwise nitrogen balance bonus (fixer + non-fixer = 1.0)
    D_ij = functional diversity bonus
    h_i  = linear species bias (economic value)
    k    = target number of species to select
    λ    = penalty strength for species count constraint
    α, β, γ = objective component weights (default 0.7, 0.2, 0.1)

When a confidence matrix is provided, N_ij is scaled by (1 - confidence_ij)
to avoid double-counting nitrogen synergy already captured in LER data.

Sign convention: QUBO minimizes, so beneficial terms are negated.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polyculture_qubo.species import CANDIDATE_SPECIES


@dataclass
class QUBOConfig:
    """Configuration for QUBO construction."""

    # Objective weights (should sum to ~1.0)
    alpha: float = 0.7  # LER/interaction weight (dominant)
    beta: float = 0.2  # Nitrogen balance weight
    gamma: float = 0.1  # Functional diversity weight

    # Constraint parameters
    target_species: int = 4  # Target number of species to select (k)
    penalty_strength: float | None = None  # λ — auto-derived if None

    # Species pool (None = use all candidates)
    species_keys: list[str] | None = None


def compute_penalty_strength(
    j_matrix: pd.DataFrame,
    diversity_matrix: pd.DataFrame,
    biases: pd.DataFrame,
    config: "QUBOConfig",
    species_keys: list[str],
    margin: float = 1.5,
) -> float:
    """Derive minimum penalty strength λ that guarantees constraint enforcement.

    The penalty must be large enough that no single-variable flip violating
    the constraint can improve the objective. Specifically, λ must exceed the
    maximum objective benefit achievable by adding or removing one species.

    For adding species i to a set of k already-selected species:
      benefit_i = α·h_i + β·N_i + Σ_j(α·J_ij + γ·D_ij)  (over selected j)

    Upper bound: use max h_i, max N_i, and (k) times the max pairwise term.

    Args:
        margin: Safety multiplier (>1.0) to ensure strict enforcement.
    """
    j = j_matrix.loc[species_keys, species_keys].values
    d = diversity_matrix.loc[species_keys, species_keys].values
    h = biases.loc[species_keys, "bias"].values

    # Max diagonal objective benefit from selecting one species
    max_linear = config.alpha * np.max(np.abs(h))

    # Max pairwise objective benefit from one pair.
    # Compute each component's max separately then sum, rather than
    # adding beta to every matrix element before taking the max.
    max_interaction = config.alpha * np.max(np.abs(j))
    max_n_balance = config.beta * 1.0  # max N-balance score (fixer + non-fixer)
    max_diversity = config.gamma * np.max(np.abs(d))
    max_pairwise = max_interaction + max_n_balance + max_diversity

    # Worst case: adding one species that has max linear benefit
    # plus k pairwise benefits with already-selected species
    max_flip_benefit = max_linear + config.target_species * max_pairwise

    return margin * max_flip_benefit


def build_nitrogen_matrix(species_keys: list[str]) -> np.ndarray:
    """Build pairwise nitrogen balance matrix.

    The nitrogen balance term rewards selecting complementary fixer/non-fixer
    pairs. A fixer paired with a non-fixer is the most beneficial combination:
    the fixer provides N, the non-fixer benefits from it.

    Pairwise scores:
      - fixer + non-fixer: 1.0 (maximum synergy)
      - fixer + fixer: 0.2 (diminishing returns on N-fixation)
      - non-fixer + non-fixer: 0.0 (no N benefit)
    """
    n = len(species_keys)
    is_fixer = np.array([CANDIDATE_SPECIES[k].is_nitrogen_fixer for k in species_keys])
    n_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if is_fixer[i] != is_fixer[j]:
                n_matrix[i, j] = 1.0  # Complementary: fixer + non-fixer
            elif is_fixer[i] and is_fixer[j]:
                n_matrix[i, j] = 0.2  # Both fixers: diminishing returns
            # Both non-fixers: 0.0

    return n_matrix


def build_qubo_matrix(
    j_matrix: pd.DataFrame,
    diversity_matrix: pd.DataFrame,
    biases: pd.DataFrame,
    config: QUBOConfig | None = None,
    confidence_matrix: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the QUBO Q matrix from preprocessed components.

    Args:
        j_matrix: Pairwise interaction matrix (J_ij), species-indexed DataFrame
        diversity_matrix: Pairwise diversity scores, species-indexed DataFrame
        biases: Linear biases (h_i), species-indexed DataFrame with 'bias' column
        config: QUBO configuration parameters
        confidence_matrix: Pairwise confidence scores. When provided, the
            N-balance term is scaled by (1 - confidence) to avoid double-counting
            with LER data that already captures nitrogen fixation benefits.

    Returns:
        (Q, species_keys): Q matrix as numpy array and ordered species key list
    """
    if config is None:
        config = QUBOConfig()

    species_keys = config.species_keys or sorted(CANDIDATE_SPECIES.keys())
    n = len(species_keys)
    k = config.target_species

    if config.penalty_strength is not None:
        lam = config.penalty_strength
    else:
        lam = compute_penalty_strength(
            j_matrix, diversity_matrix, biases, config, species_keys
        )

    # Extract numpy arrays for selected species
    j = j_matrix.loc[species_keys, species_keys].values.copy()
    d = diversity_matrix.loc[species_keys, species_keys].values.copy()
    h = biases.loc[species_keys, "bias"].values.copy()
    n_mat = build_nitrogen_matrix(species_keys)

    # Scale N-balance by (1 - confidence) to avoid double-counting.
    # LER data already captures cereal-legume nitrogen synergy, so the
    # explicit N-balance term should only contribute for pairs where we
    # lack empirical LER evidence.
    if confidence_matrix is not None:
        conf = confidence_matrix.loc[species_keys, species_keys].values
        # Upper triangle only (n_mat is already upper triangular)
        for i in range(n):
            for jj in range(i + 1, n):
                n_mat[i, jj] *= 1.0 - conf[i, jj]

    # Build Q matrix (upper triangular form).
    #
    # Energy is computed as: E = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j
    # This is the standard QUBO convention where Q is upper triangular
    # and each off-diagonal pair is counted once.
    q = np.zeros((n, n))

    # --- Off-diagonal terms (i < j, each pair counted once) ---
    for i in range(n):
        for jj in range(i + 1, n):
            q[i, jj] = (
                -config.alpha * j[i, jj]  # Interaction: complementary pairs → negative Q (good)
                - config.beta * n_mat[i, jj]  # N-balance: fixer+non-fixer pairs → negative Q (good)
                - config.gamma * d[i, jj]  # Diversity: different species → negative Q (good)
                + 2 * lam  # Count constraint: penalizes selecting extra pairs
            )

    # --- Diagonal terms ---
    for i in range(n):
        q[i, i] = (
            -config.alpha * h[i]  # Species value: good species → negative Q (good)
            + lam * (1 - 2 * k)  # Count constraint: encourages selecting up to k species
        )

    return q, species_keys


def qubo_energy(q: np.ndarray, x: np.ndarray) -> float:
    """Compute the QUBO energy for a given binary solution vector.

    Uses upper triangular convention: E = Σ_i Q_ii x_i + Σ_{i<j} Q_ij x_i x_j
    """
    energy = 0.0
    n = len(x)
    for i in range(n):
        if x[i]:
            energy += q[i, i]
            for j in range(i + 1, n):
                if x[j]:
                    energy += q[i, j]
    return energy


def solution_to_species(x: np.ndarray, species_keys: list[str]) -> list[str]:
    """Convert a binary solution vector to a list of selected species."""
    return [species_keys[i] for i in range(len(x)) if x[i] == 1]


def evaluate_solution(
    x: np.ndarray,
    species_keys: list[str],
    q: np.ndarray,
    j_matrix: pd.DataFrame,
    diversity_matrix: pd.DataFrame,
    config: QUBOConfig | None = None,
    confidence_matrix: pd.DataFrame | None = None,
) -> dict:
    """Evaluate a solution with detailed breakdown of objective components.

    Args:
        confidence_matrix: If provided, N-balance scores are scaled by
            (1 - confidence) to match the actual QUBO construction.

    Returns a dict with energy, selected species, constraint satisfaction,
    and per-component scores.
    """
    if config is None:
        config = QUBOConfig()

    selected = solution_to_species(x, species_keys)
    n_selected = len(selected)
    energy = qubo_energy(q, x)

    # LER score: sum of pairwise J_ij for selected pairs
    ler_score = 0.0
    for i, si in enumerate(selected):
        for sj in selected[i + 1 :]:
            ler_score += j_matrix.loc[si, sj]

    # Diversity score: sum of pairwise diversity for selected pairs
    div_score = 0.0
    for i, si in enumerate(selected):
        for sj in selected[i + 1 :]:
            div_score += diversity_matrix.loc[si, sj]

    # Nitrogen balance (pairwise), with optional confidence scaling
    n_fixers = sum(1 for s in selected if CANDIDATE_SPECIES[s].is_nitrogen_fixer)
    n_non_fixers = n_selected - n_fixers
    n_mat = build_nitrogen_matrix(species_keys)
    if confidence_matrix is not None:
        conf = confidence_matrix.loc[species_keys, species_keys].values
        n = len(species_keys)
        for i in range(n):
            for jj in range(i + 1, n):
                n_mat[i, jj] *= 1.0 - conf[i, jj]
    n_score = 0.0
    for i, si in enumerate(selected):
        idx_i = species_keys.index(si)
        for sj in selected[i + 1 :]:
            idx_j = species_keys.index(sj)
            ii, jj = min(idx_i, idx_j), max(idx_i, idx_j)
            n_score += n_mat[ii, jj]

    # Cash crop count
    n_cash = sum(1 for s in selected if CANDIDATE_SPECIES[s].is_cash_crop)

    return {
        "energy": energy,
        "selected_species": selected,
        "n_selected": n_selected,
        "target_species": config.target_species,
        "constraint_satisfied": n_selected == config.target_species,
        "ler_score": ler_score,
        "diversity_score": div_score,
        "n_balance_score": n_score,
        "n_fixers": n_fixers,
        "n_non_fixers": n_non_fixers,
        "n_cash_crops": n_cash,
        "has_cash_crop": n_cash > 0,
    }


def normalize_qubo(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize a QUBO matrix so its entries lie in [-1, 1].

    QAOA performance depends on the energy scale of the cost Hamiltonian.
    The variational parameter γ in exp(-iγ H_C) has an optimal range that
    scales inversely with the spectral width of H_C. When Q entries span
    orders of magnitude (as ours do — objective terms O(0.01-0.5) vs.
    penalty terms O(several)), the parameter landscape becomes difficult
    to navigate.

    Dividing by the max absolute entry puts all Q coefficients in [-1, 1],
    giving γ a consistent optimal range across problem instances.

    Args:
        q: Upper triangular QUBO matrix (N x N)

    Returns:
        (q_normalized, scale): Normalized matrix and scale factor.
            To recover original energies: E_original = scale * E_normalized
    """
    scale = np.max(np.abs(q))
    if scale == 0:
        return q.copy(), 1.0
    return q / scale, scale


def qubo_to_ising(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert upper-triangular QUBO matrix to Ising model (h, J, offset).

    QAOA circuits operate on Ising spin variables s_i ∈ {-1, +1}, not
    binary variables x_i ∈ {0, 1}. This function performs the standard
    substitution x_i = (1 - s_i) / 2 and collects terms.

    The Ising Hamiltonian is:
        H = offset + Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j

    The substitution gives:
        J_ij = Q_ij / 4           (coupling between spins i and j)
        h_i  = -Q_ii/2 - (1/4) Σ_{j≠i} Q̃_ij   (local field on spin i)
        offset = (1/2) Σ_i Q_ii + (1/4) Σ_{i<j} Q_ij

    where Q̃_ij is the upper-triangular entry for the pair (i, j).

    Args:
        q: Upper triangular QUBO matrix (N x N)

    Returns:
        (h, J, offset): h is N-vector of local fields, J is upper-triangular
            N x N coupling matrix, offset is a constant energy shift.
    """
    n = q.shape[0]

    # Ising couplings: J_ij = Q_ij / 4 for i < j
    j_ising = np.triu(q, k=1) / 4.0

    # Constant offset
    offset = 0.5 * np.trace(q) + 0.25 * np.sum(np.triu(q, k=1))

    # Local fields: h_i = -Q_ii/2 - (1/4) * sum of Q entries involving i
    h = np.zeros(n)
    for i in range(n):
        h[i] = -q[i, i] / 2.0
        # Q entries where i is first index (i < j): q[i, j]
        h[i] -= 0.25 * np.sum(q[i, i + 1 :])
        # Q entries where i is second index (j < i): q[j, i]
        h[i] -= 0.25 * np.sum(q[:i, i])

    return h, j_ising, offset


def print_solution(eval_result: dict) -> None:
    """Print a human-readable summary of an evaluated solution."""
    r = eval_result
    status = "YES" if r["constraint_satisfied"] else "NO"
    print(f"Energy: {r['energy']:.4f}")
    print(f"Selected ({r['n_selected']}/{r['target_species']}): {', '.join(r['selected_species'])}")
    print(f"Constraint satisfied: {status}")
    print(f"LER score: {r['ler_score']:.4f}")
    print(f"N-balance score: {r['n_balance_score']:.4f}")
    print(f"Diversity score: {r['diversity_score']:.4f}")
    print(f"N-fixers: {r['n_fixers']}, Non-fixers: {r['n_non_fixers']}")
    print(f"Cash crops: {r['n_cash_crops']}, Has cash crop: {r['has_cash_crop']}")
