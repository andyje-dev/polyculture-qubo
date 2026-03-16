"""Sensitivity analysis for the polyculture QUBO.

Tests how the optimal solution changes when:
1. Objective weights (α, β, γ) are varied
2. Observed species pairs are removed from the interaction matrix (data masking)
"""

from dataclasses import dataclass

import pandas as pd

from polyculture_qubo.matrix.interaction import (
    build_diversity_matrix,
    build_interaction_matrix,
    compute_linear_biases,
)
from polyculture_qubo.matrix.qubo import QUBOConfig, build_qubo_matrix
from polyculture_qubo.solvers.exact import ExactSolver


@dataclass
class WeightSweepResult:
    """Result of a single weight configuration."""

    alpha: float
    beta: float
    gamma: float
    best_energy: float
    selected_species: list[str]


def weight_sweep(
    j_matrix: pd.DataFrame,
    diversity_matrix: pd.DataFrame,
    biases: pd.DataFrame,
    confidence_matrix: pd.DataFrame,
    target_species: int = 4,
    n_points: int = 21,
) -> list[WeightSweepResult]:
    """Sweep objective weights and track optimal solution at each point.

    Varies α, β, γ on a grid where α + β + γ = 1.0, with α ≥ 0.3
    (LER should always dominate). Uses barycentric coordinates on the
    simplex.

    Args:
        n_points: Resolution of the grid along each axis.

    Returns:
        List of WeightSweepResult for each grid point.
    """
    results = []
    step = 1.0 / (n_points - 1)

    for i in range(n_points):
        alpha = i * step
        if alpha < 0.3:
            continue  # LER must dominate
        for j in range(n_points - i):
            beta = j * step
            gamma = 1.0 - alpha - beta
            if gamma < -1e-10:
                continue
            gamma = max(0.0, gamma)

            config = QUBOConfig(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                target_species=target_species,
            )
            q, species_keys = build_qubo_matrix(
                j_matrix, diversity_matrix, biases, config, confidence_matrix
            )
            result = ExactSolver().solve(
                q, target_species=target_species, collect_all=False
            )

            selected = [
                species_keys[idx]
                for idx in range(len(species_keys))
                if result.best_solution[idx] == 1
            ]
            results.append(
                WeightSweepResult(
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    best_energy=result.best_energy,
                    selected_species=sorted(selected),
                )
            )

    return results


def analyze_weight_sweep(results: list[WeightSweepResult]) -> dict:
    """Compute summary statistics from a weight sweep.

    Returns:
        Dict with solution stability metrics and species frequency.
    """
    # How many distinct solutions?
    unique_solutions = set()
    for r in results:
        unique_solutions.add(tuple(r.selected_species))

    # Most common solution
    from collections import Counter

    solution_counts = Counter(tuple(r.selected_species) for r in results)
    most_common = solution_counts.most_common(1)[0]

    # Species frequency: how often does each species appear across configs?
    species_freq: dict[str, int] = {}
    for r in results:
        for sp in r.selected_species:
            species_freq[sp] = species_freq.get(sp, 0) + 1

    total = len(results)

    return {
        "n_configs": total,
        "n_unique_solutions": len(unique_solutions),
        "stability": most_common[1] / total,
        "most_common_solution": list(most_common[0]),
        "most_common_count": most_common[1],
        "species_frequency": {
            sp: count / total
            for sp, count in sorted(species_freq.items(), key=lambda x: -x[1])
        },
        "all_unique_solutions": [
            {"solution": list(sol), "count": count}
            for sol, count in solution_counts.most_common()
        ],
    }


@dataclass
class MaskingResult:
    """Result of masking one observed pair from the interaction matrix."""

    masked_pair: tuple[str, str]
    masked_ler: float
    masked_j: float
    original_solution: list[str]
    masked_solution: list[str]
    solution_changed: bool
    original_energy: float
    masked_energy: float


def data_masking_analysis(
    ler_stats: pd.DataFrame,
    companion_data: pd.DataFrame | None,
    bischoff_pairs: pd.DataFrame | None,
    config: QUBOConfig | None = None,
) -> list[MaskingResult]:
    """Systematically mask each observed pair and measure impact.

    For each of the 31 observed pairs, removes it from the LER data,
    rebuilds the QUBO (the pair falls back to its prior), and solves.
    Identifies which pairs are most influential on the optimal solution.

    Returns:
        List of MaskingResult sorted by impact (changed solutions first).
    """
    if config is None:
        config = QUBOConfig()

    d_matrix = build_diversity_matrix()
    biases_dict = compute_linear_biases()
    biases = pd.DataFrame.from_dict(biases_dict, orient="index", columns=["bias"])

    # Full solution
    j_full, c_full = build_interaction_matrix(ler_stats, companion_data, bischoff_pairs)
    q_full, species_keys = build_qubo_matrix(j_full, d_matrix, biases, config, c_full)
    full_result = ExactSolver().solve(
        q_full, target_species=config.target_species, collect_all=False
    )
    full_selected = sorted(
        species_keys[i]
        for i in range(len(species_keys))
        if full_result.best_solution[i] == 1
    )

    results = []
    for _, row in ler_stats.iterrows():
        sp_a, sp_b = row["species_a"], row["species_b"]

        # Mask this pair
        mask = ~(
            ((ler_stats["species_a"] == sp_a) & (ler_stats["species_b"] == sp_b))
            | ((ler_stats["species_a"] == sp_b) & (ler_stats["species_b"] == sp_a))
        )
        ler_masked = ler_stats[mask]

        j_masked, c_masked = build_interaction_matrix(
            ler_masked, companion_data, bischoff_pairs
        )
        q_masked, _ = build_qubo_matrix(j_masked, d_matrix, biases, config, c_masked)
        masked_result = ExactSolver().solve(
            q_masked, target_species=config.target_species, collect_all=False
        )
        masked_selected = sorted(
            species_keys[i]
            for i in range(len(species_keys))
            if masked_result.best_solution[i] == 1
        )

        results.append(
            MaskingResult(
                masked_pair=(sp_a, sp_b),
                masked_ler=row["ler_mean"],
                masked_j=float(j_full.loc[sp_a, sp_b]),
                original_solution=full_selected,
                masked_solution=masked_selected,
                solution_changed=full_selected != masked_selected,
                original_energy=full_result.best_energy,
                masked_energy=masked_result.best_energy,
            )
        )

    # Sort: changed solutions first, then by LER impact
    results.sort(key=lambda r: (not r.solution_changed, -abs(r.masked_j)))
    return results


@dataclass
class CrossWeightMaskingResult:
    """Summary of data masking across multiple weight configurations."""

    masked_pair: tuple[str, str]
    masked_ler: float
    n_configs_tested: int
    n_configs_changed: int  # How many weight configs had solution change


def cross_weight_data_masking(
    ler_stats: pd.DataFrame,
    companion_data: pd.DataFrame | None,
    bischoff_pairs: pd.DataFrame | None,
    target_species: int = 4,
    weight_configs: list[tuple[float, float, float]] | None = None,
) -> list[CrossWeightMaskingResult]:
    """Test data masking robustness across multiple weight configurations.

    For each observed pair, removes it and checks whether the optimal solution
    changes under several representative weight configs (not just the default).
    This tests whether "load-bearing" pairs are weight-dependent.

    Args:
        weight_configs: List of (alpha, beta, gamma) tuples. If None, uses
            5 representative configs spanning the weight simplex.

    Returns:
        List of CrossWeightMaskingResult sorted by impact.
    """
    if weight_configs is None:
        weight_configs = [
            (0.7, 0.2, 0.1),  # Default
            (0.5, 0.3, 0.2),  # More N-balance and diversity
            (0.9, 0.05, 0.05),  # LER-dominated
            (0.4, 0.5, 0.1),  # N-balance heavy
            (0.3, 0.2, 0.5),  # Diversity heavy
        ]

    d_matrix = build_diversity_matrix()
    biases_dict = compute_linear_biases()
    biases = pd.DataFrame.from_dict(biases_dict, orient="index", columns=["bias"])

    results = []
    for _, row in ler_stats.iterrows():
        sp_a, sp_b = row["species_a"], row["species_b"]

        # Mask this pair
        mask = ~(
            ((ler_stats["species_a"] == sp_a) & (ler_stats["species_b"] == sp_b))
            | ((ler_stats["species_a"] == sp_b) & (ler_stats["species_b"] == sp_a))
        )
        ler_masked = ler_stats[mask]

        n_changed = 0
        for alpha, beta, gamma in weight_configs:
            config = QUBOConfig(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                target_species=target_species,
            )

            # Full solution for this weight config
            j_full, c_full = build_interaction_matrix(
                ler_stats, companion_data, bischoff_pairs
            )
            q_full, species_keys = build_qubo_matrix(
                j_full, d_matrix, biases, config, c_full
            )
            full_result = ExactSolver().solve(
                q_full, target_species=target_species, collect_all=False
            )
            full_selected = sorted(
                species_keys[i]
                for i in range(len(species_keys))
                if full_result.best_solution[i] == 1
            )

            # Masked solution
            j_masked, c_masked = build_interaction_matrix(
                ler_masked, companion_data, bischoff_pairs
            )
            q_masked, _ = build_qubo_matrix(
                j_masked, d_matrix, biases, config, c_masked
            )
            masked_result = ExactSolver().solve(
                q_masked, target_species=target_species, collect_all=False
            )
            masked_selected = sorted(
                species_keys[i]
                for i in range(len(species_keys))
                if masked_result.best_solution[i] == 1
            )

            if full_selected != masked_selected:
                n_changed += 1

        results.append(
            CrossWeightMaskingResult(
                masked_pair=(sp_a, sp_b),
                masked_ler=row["ler_mean"],
                n_configs_tested=len(weight_configs),
                n_configs_changed=n_changed,
            )
        )

    results.sort(key=lambda r: (-r.n_configs_changed, -r.masked_ler))
    return results


def print_sensitivity_summary(
    sweep_analysis: dict,
    masking_results: list[MaskingResult],
    cross_weight_masking: list[CrossWeightMaskingResult] | None = None,
) -> None:
    """Print a human-readable sensitivity analysis summary."""
    print(f"\n{'=' * 70}")
    print("  SENSITIVITY ANALYSIS")
    print(f"{'=' * 70}")

    sa = sweep_analysis
    print(f"\n--- Weight Sweep ({sa['n_configs']} configurations) ---")
    print(f"  Unique optimal solutions: {sa['n_unique_solutions']}")
    print(f"  Solution stability: {sa['stability']:.1%}")
    print(
        f"  Most common: {sa['most_common_solution']} "
        f"({sa['most_common_count']}/{sa['n_configs']})"
    )

    print("\n  Species frequency across all configs:")
    for sp, freq in sa["species_frequency"].items():
        bar = "█" * int(freq * 40)
        print(f"    {sp:<20s} {freq:>5.1%} {bar}")

    if sa["n_unique_solutions"] > 1:
        print("\n  All unique solutions:")
        for entry in sa["all_unique_solutions"]:
            print(f"    {entry['solution']} — {entry['count']} configs")

    print(f"\n--- Data Masking ({len(masking_results)} observed pairs) ---")
    n_changed = sum(1 for r in masking_results if r.solution_changed)
    print(
        f"  Pairs whose removal changes the solution: {n_changed}/{len(masking_results)}"
    )

    if n_changed > 0:
        print("\n  Influential pairs (removal changes optimal solution):")
        for r in masking_results:
            if r.solution_changed:
                dropped = set(r.original_solution) - set(r.masked_solution)
                added = set(r.masked_solution) - set(r.original_solution)
                print(
                    f"    {r.masked_pair[0]}+{r.masked_pair[1]} "
                    f"(LER={r.masked_ler:.2f}, J={r.masked_j:.4f}): "
                    f"dropped {dropped}, added {added}"
                )

    print("\n  Top 5 pairs by interaction strength:")
    top_5 = sorted(masking_results, key=lambda r: -abs(r.masked_j))[:5]
    for r in top_5:
        status = "CHANGES solution" if r.solution_changed else "stable"
        print(
            f"    {r.masked_pair[0]}+{r.masked_pair[1]}: "
            f"J={r.masked_j:.4f}, LER={r.masked_ler:.2f} [{status}]"
        )

    if cross_weight_masking is not None:
        print("\n--- Cross-Weight Data Masking ---")
        n_configs = (
            cross_weight_masking[0].n_configs_tested if cross_weight_masking else 0
        )
        any_changed = [r for r in cross_weight_masking if r.n_configs_changed > 0]
        print(
            f"  Tested {len(cross_weight_masking)} pairs across {n_configs} weight configs"
        )
        print(
            f"  Pairs influential in ≥1 config: {len(any_changed)}/{len(cross_weight_masking)}"
        )
        if any_changed:
            print("  Load-bearing pairs by weight robustness:")
            for r in any_changed:
                print(
                    f"    {r.masked_pair[0]}+{r.masked_pair[1]} "
                    f"(LER={r.masked_ler:.2f}): changes solution in "
                    f"{r.n_configs_changed}/{r.n_configs_tested} weight configs"
                )
