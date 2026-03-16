"""Filter candidate species pool based on data coverage and constraints.

Per CLAUDE.md: "Before implementation, filter the candidate pool to retain only
species with ≥3 pairwise interaction observations in the combined dataset."
"""

import pandas as pd

from regen_qubo.species import CANDIDATE_SPECIES


def filter_species_by_coverage(
    ler_stats: pd.DataFrame,
    min_pairwise_obs: int = 3,
    min_pairs_per_species: int = 2,
) -> list[str]:
    """Select species that have sufficient pairwise interaction data.

    Args:
        ler_stats: Output of compute_pairwise_ler_stats()
        min_pairwise_obs: Minimum observations per pair to count as "covered"
        min_pairs_per_species: Minimum number of covered pairs a species must
            participate in to be retained

    Returns:
        Sorted list of species keys that pass the filter
    """
    # Pairs meeting the observation threshold
    good_pairs = ler_stats[ler_stats["n_obs"] >= min_pairwise_obs]

    # Count how many good pairs each species participates in
    pair_counts: dict[str, int] = {}
    for _, row in good_pairs.iterrows():
        for sp in [row["species_a"], row["species_b"]]:
            pair_counts[sp] = pair_counts.get(sp, 0) + 1

    # Filter to species meeting the minimum pairs threshold
    selected = [sp for sp, count in pair_counts.items() if count >= min_pairs_per_species]

    return sorted(selected)


def print_coverage_report(
    ler_stats: pd.DataFrame,
    selected_species: list[str],
) -> None:
    """Print a human-readable report of species data coverage."""
    print(f"\n=== Species Pool Filter Report ===")
    print(f"Total candidates: {len(CANDIDATE_SPECIES)}")
    print(f"Selected (sufficient data): {len(selected_species)}")
    print()

    # Per-species coverage
    pair_counts: dict[str, int] = {}
    obs_counts: dict[str, int] = {}
    for _, row in ler_stats.iterrows():
        for sp in [row["species_a"], row["species_b"]]:
            pair_counts[sp] = pair_counts.get(sp, 0) + 1
            obs_counts[sp] = obs_counts.get(sp, 0) + int(row["n_obs"])

    print(f"{'Species':<20} {'Pairs':>5} {'Obs':>6} {'Selected':>8}")
    print("-" * 42)
    for key in sorted(CANDIDATE_SPECIES.keys()):
        sp = CANDIDATE_SPECIES[key]
        n_pairs = pair_counts.get(key, 0)
        n_obs = obs_counts.get(key, 0)
        sel = "  yes" if key in selected_species else "  no"
        print(f"{sp.common_name:<20} {n_pairs:>5} {n_obs:>6} {sel:>8}")

    # Coverage matrix for selected species
    print(f"\n=== Pair coverage for selected species ===")
    good_pairs = ler_stats[ler_stats["n_obs"] >= 3]
    selected_pairs = good_pairs[
        good_pairs["species_a"].isin(selected_species)
        & good_pairs["species_b"].isin(selected_species)
    ]
    n_selected = len(selected_species)
    n_possible = n_selected * (n_selected - 1) // 2
    print(f"Observed pairs: {len(selected_pairs)} / {n_possible}")
    print(f"Coverage: {len(selected_pairs) / n_possible * 100:.1f}%")
