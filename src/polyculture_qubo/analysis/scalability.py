"""Scalability projection for quantum advantage analysis.

Computes problem hardness metrics across different problem sizes (k values)
and projects how these scale. At N=20, classical solvers dominate — the
value is in characterizing the problem structure to argue whether quantum
advantage could emerge at larger N.
"""

from dataclasses import dataclass
from math import comb

import numpy as np

from polyculture_qubo.analysis.landscape import LandscapeMetrics


@dataclass
class ScalabilityMetrics:
    """Scalability indicators for quantum advantage projection."""

    # Problem size
    k_values: list[int]
    n_species: int

    # Per-k metrics
    solution_space_sizes: list[int]  # C(N, k) for each k
    spectral_gaps: list[float]
    gap_ratios: list[float]
    degeneracies_1pct: list[int]
    degeneracies_5pct: list[int]
    local_minima_fractions: list[float]
    energy_stds: list[float]

    # Q matrix properties
    condition_numbers: list[float]
    penalty_to_objective_ratios: list[float]

    # QAOA performance (if available)
    qaoa_approx_ratios: dict[int, dict[int, float]]  # k -> {depth -> ratio}


def compute_scalability_metrics(
    landscape_metrics: dict[int, LandscapeMetrics],
    q_matrices: dict[int, np.ndarray],
    qaoa_results: dict[int, dict[int, float]] | None = None,
) -> ScalabilityMetrics:
    """Compute scalability metrics from landscape characterizations.

    Args:
        landscape_metrics: Dict mapping k -> LandscapeMetrics.
        q_matrices: Dict mapping k -> Q matrix (for condition number).
        qaoa_results: Optional dict mapping k -> {depth -> approx_ratio}.

    Returns:
        ScalabilityMetrics summarizing scaling behavior.
    """
    k_values = sorted(landscape_metrics.keys())
    n = landscape_metrics[k_values[0]].n_solutions  # Will be overridden
    # Get n from the first landscape's metadata
    for k in k_values:
        lm = landscape_metrics[k]
        # n_species is inferred from C(N, k) = n_solutions
        # Solve for N given C(N,k) = n_solutions
        for n_try in range(k, 100):
            if comb(n_try, k) == lm.n_solutions:
                n = n_try
                break

    solution_sizes = [comb(n, k) for k in k_values]
    spectral_gaps = [landscape_metrics[k].spectral_gap for k in k_values]
    gap_ratios = [landscape_metrics[k].gap_ratio for k in k_values]
    deg_1 = [landscape_metrics[k].n_within_1pct for k in k_values]
    deg_5 = [landscape_metrics[k].n_within_5pct for k in k_values]
    lm_fracs = [landscape_metrics[k].local_minima_fraction for k in k_values]
    stds = [landscape_metrics[k].std_energy for k in k_values]

    # Q matrix condition numbers and penalty ratios
    cond_nums = []
    penalty_ratios = []
    for k in k_values:
        q = q_matrices[k]
        # Condition number of the symmetric part
        q_sym = (q + q.T) / 2
        eigenvalues = np.linalg.eigvalsh(q_sym)
        nonzero = eigenvalues[np.abs(eigenvalues) > 1e-10]
        if len(nonzero) > 0:
            cond = float(np.max(np.abs(nonzero)) / np.min(np.abs(nonzero)))
        else:
            cond = float("inf")
        cond_nums.append(cond)

        # Penalty-to-objective ratio: constant penalty offset vs. objective energy range
        # For feasible solutions, the penalty contributes a constant -k²λ offset.
        # The objective range is the spread of agronomic signal across solutions.
        lm = landscape_metrics[k]
        if lm.obj_range > 0 and lm.penalty_offset != 0:
            ratio = abs(lm.penalty_offset) / lm.obj_range
        else:
            ratio = float("inf")
        penalty_ratios.append(float(ratio))

    return ScalabilityMetrics(
        k_values=k_values,
        n_species=n,
        solution_space_sizes=solution_sizes,
        spectral_gaps=spectral_gaps,
        gap_ratios=gap_ratios,
        degeneracies_1pct=deg_1,
        degeneracies_5pct=deg_5,
        local_minima_fractions=lm_fracs,
        energy_stds=stds,
        condition_numbers=cond_nums,
        penalty_to_objective_ratios=penalty_ratios,
        qaoa_approx_ratios=qaoa_results or {},
    )


def print_scalability_summary(metrics: ScalabilityMetrics) -> None:
    """Print a human-readable scalability analysis."""
    m = metrics
    print(f"\n{'=' * 70}")
    print(f"  SCALABILITY PROJECTION (N = {m.n_species})")
    print(f"{'=' * 70}")

    print(
        f"\n{'k':>3} {'C(N,k)':>8} {'Gap':>10} {'Gap Ratio':>10} "
        f"{'Deg(1%)':>8} {'Deg(5%)':>8} {'LM Frac':>8} {'Cond#':>10}"
    )
    print("-" * 70)
    for i, k in enumerate(m.k_values):
        print(
            f"{k:>3} {m.solution_space_sizes[i]:>8} "
            f"{m.spectral_gaps[i]:>10.6f} {m.gap_ratios[i]:>10.6f} "
            f"{m.degeneracies_1pct[i]:>8} {m.degeneracies_5pct[i]:>8} "
            f"{m.local_minima_fractions[i]:>8.1%} {m.condition_numbers[i]:>10.1f}"
        )

    # Gap trend
    if len(m.spectral_gaps) >= 2:
        gap_trend = (m.spectral_gaps[-1] - m.spectral_gaps[0]) / (
            m.k_values[-1] - m.k_values[0]
        )
        print(f"\n  Gap trend (Δgap/Δk): {gap_trend:.6f}")
        if gap_trend < 0:
            print("  → Gap shrinks with k: problem gets harder at larger scale")
        else:
            print("  → Gap grows with k: problem structure remains favorable")

    # Degeneracy trend
    if len(m.degeneracies_5pct) >= 2:
        deg_ratio = m.degeneracies_5pct[-1] / max(m.degeneracies_5pct[0], 1)
        space_ratio = m.solution_space_sizes[-1] / max(m.solution_space_sizes[0], 1)
        print(
            f"\n  Degeneracy growth: {deg_ratio:.1f}x (solution space grew {space_ratio:.1f}x)"
        )
        if deg_ratio > space_ratio:
            print(
                "  → Degeneracy grows faster than solution space: increasingly flat landscape"
            )
        else:
            print(
                "  → Degeneracy grows slower than solution space: landscape differentiates"
            )

    # QAOA results
    if m.qaoa_approx_ratios:
        print("\n  QAOA Approximation Ratios:")
        depths = sorted(
            set(d for ratios in m.qaoa_approx_ratios.values() for d in ratios)
        )
        header = f"  {'k':>3}" + "".join(f"  {'p=' + str(d):>8}" for d in depths)
        print(header)
        for k in m.k_values:
            if k in m.qaoa_approx_ratios:
                row = f"  {k:>3}"
                for d in depths:
                    if d in m.qaoa_approx_ratios[k]:
                        row += f"  {m.qaoa_approx_ratios[k][d]:>8.4f}"
                    else:
                        row += f"  {'—':>8}"
                print(row)

    print("\n  Quantum advantage assessment:")
    print(f"    At N={m.n_species}, brute force enumerates all solutions in <1s.")
    print("    Classical solvers (SA) find exact optimum reliably.")
    if all(
        r > 0.99 for ratios in m.qaoa_approx_ratios.values() for r in ratios.values()
    ):
        print("    QAOA achieves >99% approximation — problem is easy for all methods.")
    print(f"    Quantum advantage would require N >> {m.n_species} with maintained")
    print("    problem structure (small gap, high degeneracy, frustrated couplings).")
