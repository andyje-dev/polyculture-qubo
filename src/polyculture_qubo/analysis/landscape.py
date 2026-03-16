"""Energy landscape characterization for the polyculture QUBO.

Analyzes the structure of the solution space using the exact solver's
complete enumeration: degeneracy, spectral gap, energy distribution,
local minima density, and frustration patterns. These metrics characterize
problem hardness and inform quantum advantage projections.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from polyculture_qubo.matrix.qubo import qubo_energy
from polyculture_qubo.solvers.result import SolverResult


@dataclass
class LandscapeMetrics:
    """Energy landscape characterization metrics."""

    # Basic stats
    k: int
    n_solutions: int
    best_energy: float
    worst_energy: float
    mean_energy: float
    std_energy: float
    skewness: float
    kurtosis: float

    # Gap structure
    spectral_gap: float  # E₁ - E₀
    gap_ratio: float  # spectral_gap / energy_range

    # Degeneracy (near-optimal solution counts)
    n_within_1pct: int
    n_within_5pct: int
    n_within_10pct: int

    # Local minima
    n_local_minima: int
    local_minima_fraction: float

    # Sorted energies (low end) for gap structure plots
    sorted_energies_bottom: np.ndarray  # lowest 50 energies

    # Objective-only decomposition (penalty subtracted)
    # For feasible solutions (exactly k selected), the penalty contribution
    # is constant: -k²λ. Subtracting it reveals the agronomic signal.
    penalty_offset: float  # the constant -k²λ
    obj_best: float
    obj_worst: float
    obj_mean: float
    obj_std: float
    obj_range: float
    obj_spectral_gap: float
    obj_n_within_1pct: int
    obj_n_within_5pct: int
    obj_n_within_10pct: int
    obj_sorted_bottom: np.ndarray  # lowest 50 objective-only energies


def analyze_landscape(
    exact_result: SolverResult,
    q: np.ndarray,
    penalty_strength: float | None = None,
) -> LandscapeMetrics:
    """Characterize the energy landscape from an exact solver result.

    Args:
        exact_result: Result from ExactSolver with collect_all=True.
        q: The QUBO matrix used (needed for local minima analysis).
        penalty_strength: λ value used in QUBO construction. When provided,
            enables objective-only decomposition by subtracting the constant
            penalty offset (-k²λ) from all feasible solution energies.

    Returns:
        LandscapeMetrics with full characterization.
    """
    energies = exact_result.all_energies
    solutions = exact_result.all_solutions
    k = exact_result.metadata["k"]
    n = exact_result.metadata["n"]

    sorted_e = np.sort(energies)
    best = sorted_e[0]
    worst = sorted_e[-1]
    energy_range = worst - best

    # Spectral gap
    spectral_gap = sorted_e[1] - sorted_e[0] if len(sorted_e) > 1 else 0.0
    gap_ratio = spectral_gap / energy_range if energy_range > 0 else 0.0

    # Degeneracy: count solutions within ε of optimum
    # "Within X%" means energy within X% of |best| above the best
    threshold_1 = best + 0.01 * abs(best)
    threshold_5 = best + 0.05 * abs(best)
    threshold_10 = best + 0.10 * abs(best)

    n_within_1 = int(np.sum(energies <= threshold_1))
    n_within_5 = int(np.sum(energies <= threshold_5))
    n_within_10 = int(np.sum(energies <= threshold_10))

    # Local minima analysis
    n_local_minima = _count_local_minima(solutions, energies, q, n, k)

    # Bottom 50 sorted energies for gap structure plot
    bottom_n = min(50, len(sorted_e))

    # --- Objective-only decomposition ---
    # For all feasible solutions (exactly k selected), the constraint penalty
    # (sum_i x_i - k)² = 0, but the λ terms in Q still contribute a constant:
    #   penalty_offset = k * λ(1-2k) + C(k,2) * 2λ = -k²λ
    # Subtracting this reveals the agronomic signal alone.
    if penalty_strength is not None:
        penalty_offset = -(k ** 2) * penalty_strength
        obj_energies = energies - penalty_offset
        obj_sorted = np.sort(obj_energies)
        obj_best = obj_sorted[0]
        obj_worst = obj_sorted[-1]
        obj_range = obj_worst - obj_best
        obj_gap = obj_sorted[1] - obj_sorted[0] if len(obj_sorted) > 1 else 0.0

        obj_t1 = obj_best + 0.01 * abs(obj_best) if obj_best != 0 else 0.01 * obj_range
        obj_t5 = obj_best + 0.05 * abs(obj_best) if obj_best != 0 else 0.05 * obj_range
        obj_t10 = obj_best + 0.10 * abs(obj_best) if obj_best != 0 else 0.10 * obj_range
    else:
        # No penalty info — use zeros as placeholder
        penalty_offset = 0.0
        obj_energies = energies
        obj_sorted = sorted_e
        obj_best = best
        obj_worst = worst
        obj_range = energy_range
        obj_gap = spectral_gap
        obj_t1 = threshold_1
        obj_t5 = threshold_5
        obj_t10 = threshold_10

    return LandscapeMetrics(
        k=k,
        n_solutions=len(energies),
        best_energy=best,
        worst_energy=worst,
        mean_energy=float(np.mean(energies)),
        std_energy=float(np.std(energies)),
        skewness=float(stats.skew(energies)),
        kurtosis=float(stats.kurtosis(energies)),
        spectral_gap=spectral_gap,
        gap_ratio=gap_ratio,
        n_within_1pct=n_within_1,
        n_within_5pct=n_within_5,
        n_within_10pct=n_within_10,
        n_local_minima=n_local_minima,
        local_minima_fraction=n_local_minima / len(energies),
        sorted_energies_bottom=sorted_e[:bottom_n],
        # Objective-only
        penalty_offset=penalty_offset,
        obj_best=float(obj_best),
        obj_worst=float(obj_worst),
        obj_mean=float(np.mean(obj_energies)),
        obj_std=float(np.std(obj_energies)),
        obj_range=float(obj_range),
        obj_spectral_gap=float(obj_gap),
        obj_n_within_1pct=int(np.sum(obj_energies <= obj_t1)),
        obj_n_within_5pct=int(np.sum(obj_energies <= obj_t5)),
        obj_n_within_10pct=int(np.sum(obj_energies <= obj_t10)),
        obj_sorted_bottom=obj_sorted[:bottom_n],
    )


def _count_local_minima(
    solutions: list[np.ndarray],
    energies: np.ndarray,
    q: np.ndarray,
    n: int,
    k: int,
) -> int:
    """Count solutions that are local minima under constraint-preserving swaps.

    A solution is a local minimum if every single swap (deselect one species,
    select another) results in equal or higher energy.
    """
    count = 0
    for sol, energy in zip(solutions, energies):
        if _is_local_minimum(sol, energy, q, n, k):
            count += 1
    return count


def _is_local_minimum(
    x: np.ndarray, energy: float, q: np.ndarray, n: int, k: int
) -> bool:
    """Check if a solution is a local minimum under swap moves."""
    selected = [i for i in range(n) if x[i] == 1]
    unselected = [i for i in range(n) if x[i] == 0]

    for i_out in selected:
        for j_in in unselected:
            # Compute swap delta inline (avoid import cycle)
            x_new = x.copy()
            x_new[i_out] = 0
            x_new[j_in] = 1
            new_energy = qubo_energy(q, x_new)
            if new_energy < energy - 1e-12:
                return False
    return True


def frustration_analysis(
    optimal_solution: np.ndarray,
    species_keys: list[str],
    j_matrix: pd.DataFrame,
) -> dict:
    """Identify frustrated interactions in the optimal solution.

    A pair is "frustrated" when:
    - Both species are selected but J_ij < 0 (competitive pair forced together)
    - One is selected, one isn't, but J_ij > 0 (complementary pair split apart)

    Returns:
        Dict with 'selected_competitive' and 'split_complementary' lists.
    """
    selected = [i for i, v in enumerate(optimal_solution) if v == 1]
    unselected = [i for i, v in enumerate(optimal_solution) if v == 0]

    selected_competitive = []
    for idx_a in range(len(selected)):
        for idx_b in range(idx_a + 1, len(selected)):
            i, j = selected[idx_a], selected[idx_b]
            si, sj = species_keys[i], species_keys[j]
            j_val = j_matrix.loc[si, sj]
            if j_val < -0.01:  # Meaningfully competitive (not just prior noise)
                selected_competitive.append((si, sj, float(j_val)))

    split_complementary = []
    for i_sel in selected:
        for j_unsel in unselected:
            si, sj = species_keys[i_sel], species_keys[j_unsel]
            j_val = j_matrix.loc[si, sj]
            if j_val > 0.05:  # Meaningfully complementary
                split_complementary.append((si, sj, float(j_val)))

    return {
        "selected_competitive": sorted(selected_competitive, key=lambda x: x[2]),
        "split_complementary": sorted(split_complementary, key=lambda x: -x[2]),
    }


def print_landscape_metrics(metrics: LandscapeMetrics) -> None:
    """Print a human-readable summary of landscape metrics."""
    m = metrics
    print(f"\n{'='*60}")
    print(f"  ENERGY LANDSCAPE — k = {m.k}")
    print(f"{'='*60}")
    print(f"  Total feasible solutions: {m.n_solutions}")
    print(f"  Energy range: [{m.best_energy:.4f}, {m.worst_energy:.4f}]")
    print(f"  Mean ± std: {m.mean_energy:.4f} ± {m.std_energy:.4f}")
    print(f"  Skewness: {m.skewness:.4f}, Kurtosis: {m.kurtosis:.4f}")
    print(f"\n  Spectral gap (E1 - E0): {m.spectral_gap:.6f}")
    print(f"  Gap ratio (gap / range): {m.gap_ratio:.6f}")
    print(f"\n  Degeneracy (total energy):")
    print(f"    Within 1% of optimum: {m.n_within_1pct}/{m.n_solutions}")
    print(f"    Within 5% of optimum: {m.n_within_5pct}/{m.n_solutions}")
    print(f"    Within 10% of optimum: {m.n_within_10pct}/{m.n_solutions}")
    print(f"\n  Local minima: {m.n_local_minima}/{m.n_solutions} "
          f"({m.local_minima_fraction:.1%})")

    if m.penalty_offset != 0:
        print(f"\n  --- Objective-Only Decomposition (penalty subtracted) ---")
        print(f"  Penalty offset (constant for all feasible): {m.penalty_offset:.4f}")
        print(f"  Objective range: [{m.obj_best:.4f}, {m.obj_worst:.4f}]")
        print(f"  Objective mean ± std: {m.obj_mean:.4f} ± {m.obj_std:.4f}")
        print(f"  Objective spectral gap: {m.obj_spectral_gap:.6f}")
        total_range_pct = (m.obj_range / abs(m.penalty_offset) * 100
                           if m.penalty_offset != 0 else 0)
        print(f"  Objective range as % of penalty: {total_range_pct:.2f}%")
        print(f"  Degeneracy (objective only):")
        print(f"    Within 1% of optimum: {m.obj_n_within_1pct}/{m.n_solutions}")
        print(f"    Within 5% of optimum: {m.obj_n_within_5pct}/{m.n_solutions}")
        print(f"    Within 10% of optimum: {m.obj_n_within_10pct}/{m.n_solutions}")
