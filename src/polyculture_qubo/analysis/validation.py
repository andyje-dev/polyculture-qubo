"""Retrospective validation of QUBO solutions against empirical data.

Validates that the solver selects species pairs with strong empirical LER
evidence, and tests robustness via leave-one-out cross-validation on
observed pairs.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from polyculture_qubo.matrix.interaction import (
    build_interaction_matrix,
)
from polyculture_qubo.matrix.qubo import (
    QUBOConfig,
    build_qubo_matrix,
)
from polyculture_qubo.solvers.exact import ExactSolver
from polyculture_qubo.species import CANDIDATE_SPECIES


def pairwise_contribution_table(
    selected_species: list[str],
    j_matrix: pd.DataFrame,
    confidence_matrix: pd.DataFrame,
    diversity_matrix: pd.DataFrame,
    ler_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Break down the contribution of each selected pair to the objective.

    For each pair in the optimal solution, shows the interaction coefficient,
    confidence, diversity bonus, nitrogen bonus, and the raw LER data
    (if observed).

    Args:
        selected_species: List of species keys in the solution.
        j_matrix: Interaction matrix.
        confidence_matrix: Confidence matrix.
        diversity_matrix: Diversity matrix.
        ler_stats: Pairwise LER statistics from data/processed/.

    Returns:
        DataFrame with one row per selected pair.
    """
    rows = []
    for sp_a, sp_b in combinations(sorted(selected_species), 2):
        j_val = j_matrix.loc[sp_a, sp_b]
        conf = confidence_matrix.loc[sp_a, sp_b]
        div = diversity_matrix.loc[sp_a, sp_b]

        # Nitrogen balance
        a_fixer = CANDIDATE_SPECIES[sp_a].is_nitrogen_fixer
        b_fixer = CANDIDATE_SPECIES[sp_b].is_nitrogen_fixer
        if a_fixer != b_fixer:
            n_bal = 1.0
        elif a_fixer and b_fixer:
            n_bal = 0.2
        else:
            n_bal = 0.0

        # Look up raw LER data
        match = ler_stats[
            ((ler_stats["species_a"] == sp_a) & (ler_stats["species_b"] == sp_b))
            | ((ler_stats["species_a"] == sp_b) & (ler_stats["species_b"] == sp_a))
        ]
        if len(match) > 0:
            row = match.iloc[0]
            ler_mean = row["ler_mean"]
            ler_std = row["ler_std"]
            n_obs = int(row["n_obs"])
            n_articles = int(row["n_articles"])
            data_source = "LER"
        else:
            ler_mean = None
            ler_std = None
            n_obs = 0
            n_articles = 0
            data_source = "prior"

        rows.append(
            {
                "species_a": sp_a,
                "species_b": sp_b,
                "j_coefficient": j_val,
                "confidence": conf,
                "diversity": div,
                "n_balance": n_bal,
                "ler_mean": ler_mean,
                "ler_std": ler_std,
                "n_obs": n_obs,
                "n_articles": n_articles,
                "data_source": data_source,
            }
        )

    return pd.DataFrame(rows)


def pairwise_ranking_check(
    selected_species: list[str],
    ler_stats: pd.DataFrame,
) -> dict:
    """Check how the solver's selected pairs rank among all observed pairs.

    Compares the mean LER of selected pairs against the full distribution
    of observed LERs to assess whether the solver is picking empirically
    strong combinations.

    Returns:
        Dict with ranking statistics.
    """
    # All observed LER values, sorted descending
    all_lers = ler_stats["ler_mean"].sort_values(ascending=False).values
    n_total = len(all_lers)

    # LER values for pairs in the selected solution
    selected_lers = []
    selected_pairs = []
    for sp_a, sp_b in combinations(sorted(selected_species), 2):
        match = ler_stats[
            ((ler_stats["species_a"] == sp_a) & (ler_stats["species_b"] == sp_b))
            | ((ler_stats["species_a"] == sp_b) & (ler_stats["species_b"] == sp_a))
        ]
        if len(match) > 0:
            ler = match.iloc[0]["ler_mean"]
            selected_lers.append(ler)
            selected_pairs.append((sp_a, sp_b, ler))

    if not selected_lers:
        return {
            "n_observed_pairs_in_solution": 0,
            "n_total_observed_pairs": n_total,
            "message": "No selected pairs have observed LER data.",
        }

    # Percentile ranking: what fraction of observed pairs have lower LER?
    ranks = []
    for ler in selected_lers:
        rank = np.sum(all_lers <= ler) / n_total
        ranks.append(rank)

    # Permutation test: is the mean selected LER significantly better than random?
    observed_mean_ler = float(np.mean(selected_lers))
    n_selected = len(selected_lers)
    n_permutations = 10_000
    rng = np.random.default_rng(42)
    count_as_good = 0
    for _ in range(n_permutations):
        sample = rng.choice(all_lers, size=n_selected, replace=False)
        if np.mean(sample) >= observed_mean_ler:
            count_as_good += 1
    p_value = (count_as_good + 1) / (n_permutations + 1)  # +1 for continuity correction

    return {
        "n_observed_pairs_in_solution": len(selected_lers),
        "n_total_pairs_in_solution": len(list(combinations(selected_species, 2))),
        "n_total_observed_pairs": n_total,
        "selected_pairs": selected_pairs,
        "mean_ler_of_selected": observed_mean_ler,
        "mean_ler_of_all_observed": float(np.mean(all_lers)),
        "mean_percentile_rank": float(np.mean(ranks)),
        "permutation_p_value": p_value,
        "individual_ranks": list(
            zip(
                [p[0] + "+" + p[1] for p in selected_pairs],
                [float(r) for r in ranks],
            )
        ),
    }


@dataclass
class LOOResult:
    """Result of leave-one-out cross-validation for one held-out pair."""

    held_out_pair: tuple[str, str]
    held_out_ler: float
    pair_in_full_solution: bool
    pair_in_loo_solution: bool
    consistent: bool  # Did holding out data change the pair's status correctly?
    full_solution: list[str]
    loo_solution: list[str]
    full_energy: float
    loo_energy: float


def leave_one_out_cv(
    ler_stats: pd.DataFrame,
    companion_data: pd.DataFrame | None,
    bischoff_pairs: pd.DataFrame | None,
    config: QUBOConfig | None = None,
) -> list[LOOResult]:
    """Leave-one-out cross-validation on observed species pairs.

    For each observed pair, removes it from the LER data, rebuilds the
    interaction matrix (the pair falls back to its prior), rebuilds the
    QUBO, solves exactly, and checks whether the pair's inclusion in the
    solution is consistent with its held-out LER value.

    A "consistent" result means:
    - High-LER pair (> 1.0): consistent if the pair is selected in the full
      solution, OR if the pair was selected in the full solution and removing
      its data changed the solution (demonstrating the pair's importance)
    - Neutral pair (0.95–1.05): always consistent (no strong signal)
    - Low-LER pair (< 1.0): consistent if the pair is NOT in the full solution

    Args:
        ler_stats: Full pairwise LER statistics.
        companion_data: Companion planting data (for prior rebuilding).
        bischoff_pairs: Bischoff studied pairs (for prior rebuilding).
        config: QUBO configuration. Default: QUBOConfig().

    Returns:
        List of LOOResult for each held-out pair.
    """
    from polyculture_qubo.matrix.interaction import (
        build_diversity_matrix,
        compute_linear_biases,
    )

    if config is None:
        config = QUBOConfig()

    # Full solution (no holdout)
    j_full, c_full = build_interaction_matrix(ler_stats, companion_data, bischoff_pairs)
    d_matrix = build_diversity_matrix()
    biases_dict = compute_linear_biases()
    biases = pd.DataFrame.from_dict(biases_dict, orient="index", columns=["bias"])
    q_full, species_keys = build_qubo_matrix(j_full, d_matrix, biases, config, c_full)
    full_result = ExactSolver().solve(
        q_full, target_species=config.target_species, collect_all=False
    )
    full_selected = set(
        species_keys[i]
        for i in range(len(species_keys))
        if full_result.best_solution[i] == 1
    )

    results = []
    for _, row in ler_stats.iterrows():
        sp_a, sp_b = row["species_a"], row["species_b"]

        # Hold out this pair
        mask = ~(
            ((ler_stats["species_a"] == sp_a) & (ler_stats["species_b"] == sp_b))
            | ((ler_stats["species_a"] == sp_b) & (ler_stats["species_b"] == sp_a))
        )
        ler_loo = ler_stats[mask]

        # Rebuild with held-out data
        j_loo, c_loo = build_interaction_matrix(ler_loo, companion_data, bischoff_pairs)
        q_loo, _ = build_qubo_matrix(j_loo, d_matrix, biases, config, c_loo)
        loo_result = ExactSolver().solve(
            q_loo, target_species=config.target_species, collect_all=False
        )
        loo_selected = set(
            species_keys[i]
            for i in range(len(species_keys))
            if loo_result.best_solution[i] == 1
        )

        pair_in_full = sp_a in full_selected and sp_b in full_selected
        pair_in_loo = sp_a in loo_selected and sp_b in loo_selected
        ler_val = row["ler_mean"]

        # Consistency check
        if 0.95 <= ler_val <= 1.05:
            # Neutral pair: no strong directional signal, always consistent
            consistent = True
        elif ler_val > 1.0:
            # High-LER pair: consistent if pair is in the full solution,
            # or if pair was in full solution and its removal mattered
            consistent = pair_in_full
        else:
            # Low-LER pair: consistent if it was NOT in the full solution
            consistent = not pair_in_full

        results.append(
            LOOResult(
                held_out_pair=(sp_a, sp_b),
                held_out_ler=ler_val,
                pair_in_full_solution=pair_in_full,
                pair_in_loo_solution=pair_in_loo,
                consistent=consistent,
                full_solution=sorted(full_selected),
                loo_solution=sorted(loo_selected),
                full_energy=full_result.best_energy,
                loo_energy=loo_result.best_energy,
            )
        )

    return results


def print_validation_summary(
    contribution_table: pd.DataFrame,
    ranking: dict,
    loo_results: list[LOOResult] | None = None,
) -> None:
    """Print a human-readable validation summary."""
    print(f"\n{'=' * 70}")
    print("  RETROSPECTIVE VALIDATION")
    print(f"{'=' * 70}")

    print("\n--- Pairwise Contribution Table ---")
    print(contribution_table.to_string(index=False, float_format="%.4f"))

    print("\n--- Pairwise Ranking ---")
    print(
        f"  Observed pairs in solution: {ranking['n_observed_pairs_in_solution']}"
        f"/{ranking.get('n_total_pairs_in_solution', '?')}"
    )
    if ranking["n_observed_pairs_in_solution"] > 0:
        print(f"  Mean LER of selected pairs: {ranking['mean_ler_of_selected']:.4f}")
        print(
            f"  Mean LER of all observed pairs: {ranking['mean_ler_of_all_observed']:.4f}"
        )
        print(f"  Mean percentile rank: {ranking['mean_percentile_rank']:.1%}")
        if "permutation_p_value" in ranking:
            p = ranking["permutation_p_value"]
            sig = (
                "***"
                if p < 0.001
                else "**"
                if p < 0.01
                else "*"
                if p < 0.05
                else "n.s."
            )
            print(f"  Permutation test p-value: {p:.4f} ({sig})")
        print("  Individual pair ranks:")
        for pair_name, rank in ranking["individual_ranks"]:
            print(f"    {pair_name}: top {(1 - rank):.0%}")

    if loo_results is not None:
        print(f"\n--- Leave-One-Out Cross-Validation ({len(loo_results)} pairs) ---")
        n_consistent = sum(1 for r in loo_results if r.consistent)
        print(f"  Consistent: {n_consistent}/{len(loo_results)}")
        n_changed = sum(
            1 for r in loo_results if set(r.full_solution) != set(r.loo_solution)
        )
        print(f"  Solution changed when held out: {n_changed}/{len(loo_results)}")

        # Show pairs where solution changed
        for r in loo_results:
            if set(r.full_solution) != set(r.loo_solution):
                dropped = set(r.full_solution) - set(r.loo_solution)
                added = set(r.loo_solution) - set(r.full_solution)
                print(
                    f"    Held out {r.held_out_pair[0]}+{r.held_out_pair[1]} "
                    f"(LER={r.held_out_ler:.2f}): "
                    f"dropped {dropped}, added {added}"
                )
