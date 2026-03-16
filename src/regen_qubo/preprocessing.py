"""Main preprocessing pipeline: raw data → interaction matrices → saved outputs.

Run as: python -m regen_qubo.preprocessing
"""

import numpy as np
import pandas as pd

from regen_qubo.data_loader import (
    RAW_DIR,
    PROCESSED_DIR,
    aggregate_bischoff_pairs,
    compute_pairwise_ler_stats,
    filter_to_candidate_pairs,
    load_bischoff_pairs,
    load_companion_plants,
    load_paut_dataset,
)
from regen_qubo.interaction_matrix import (
    build_diversity_matrix,
    build_interaction_matrix,
    compute_linear_biases,
)
from regen_qubo.species import CANDIDATE_SPECIES


def run_pipeline() -> dict[str, pd.DataFrame]:
    """Execute the full preprocessing pipeline.

    Returns dict with keys: ler_stats, j_matrix, confidence, diversity, biases
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load Paut et al. dataset ---
    print("Loading Paut et al. intercropping dataset...")
    paut = load_paut_dataset()
    print(f"  Total observations: {len(paut)}")
    print(f"  Observations with LER: {paut['ler_total'].notna().sum()}")

    paut_filtered = filter_to_candidate_pairs(paut)
    print(f"  Observations matching candidate species: {len(paut_filtered)}")
    print(f"  With LER data: {paut_filtered['ler_total'].notna().sum()}")

    # --- Load companion planting data (optional) ---
    companion_data = None
    companion_path = RAW_DIR / "companion_plants.csv"
    if companion_path.exists():
        print("\nLoading companion planting dataset...")
        companion_data = load_companion_plants()
        companion_filtered = filter_to_candidate_pairs(companion_data)
        print(f"  Total edges: {len(companion_data)}")
        print(f"  Edges matching candidate species: {len(companion_filtered)}")
        companion_data = companion_filtered
    else:
        print("\nCompanion planting dataset not found (optional), skipping.")

    # --- Load Bischoff et al. data (optional) ---
    bischoff_agg = None
    bischoff_path = RAW_DIR / "bischoff_species_pairs.csv"
    if bischoff_path.exists():
        print("\nLoading Bischoff et al. species pair data...")
        bischoff = load_bischoff_pairs()
        bischoff_filtered = filter_to_candidate_pairs(bischoff)
        print(f"  Total directed pairs: {len(bischoff)}")
        print(f"  Pairs matching candidate species: {len(bischoff_filtered)}")
        bischoff_agg = aggregate_bischoff_pairs(bischoff_filtered)
        print(f"  Unique undirected pairs: {len(bischoff_agg)}")
        total_exps = bischoff_agg["n_exps_total"].sum()
        print(f"  Total experiments in matching pairs: {total_exps}")
    else:
        print("\nBischoff dataset not found (optional), skipping.")

    # --- Compute pairwise LER statistics ---
    print("\nComputing pairwise LER statistics...")
    ler_stats = compute_pairwise_ler_stats(paut_filtered)
    print(f"  Unique species pairs with LER data: {len(ler_stats)}")
    print(f"  Pairs with ≥3 observations: {(ler_stats['n_obs'] >= 3).sum()}")
    print(f"  Pairs with ≥5 observations: {(ler_stats['n_obs'] >= 5).sum()}")

    # --- Build interaction matrix ---
    print("\nBuilding interaction matrix (J_ij)...")
    j_matrix, confidence = build_interaction_matrix(
        ler_stats,
        companion_data=companion_data,
        bischoff_pairs=bischoff_agg,
    )

    # Count non-default entries
    n_species = len(CANDIDATE_SPECIES)
    n_pairs = n_species * (n_species - 1) // 2
    n_observed = (confidence.values[np.triu_indices(n_species, k=1)] > 0).sum()
    print(f"  Species in pool: {n_species}")
    print(f"  Total possible pairs: {n_pairs}")
    print(f"  Pairs with data: {n_observed}")
    print(f"  Pairs using prior: {n_pairs - n_observed}")

    # --- Build diversity matrix ---
    print("\nBuilding functional diversity matrix...")
    diversity = build_diversity_matrix()

    # --- Compute linear biases ---
    print("Computing linear biases (h_i)...")
    biases = compute_linear_biases()
    biases_df = pd.DataFrame.from_dict(biases, orient="index", columns=["bias"])

    # --- Save outputs ---
    print("\nSaving processed data...")
    ler_stats.to_csv(PROCESSED_DIR / "pairwise_ler_stats.csv", index=False)
    j_matrix.to_csv(PROCESSED_DIR / "interaction_matrix.csv")
    confidence.to_csv(PROCESSED_DIR / "confidence_matrix.csv")
    diversity.to_csv(PROCESSED_DIR / "diversity_matrix.csv")
    biases_df.to_csv(PROCESSED_DIR / "linear_biases.csv")

    print(f"  Saved to {PROCESSED_DIR}/")

    # --- Summary ---
    print("\n=== Pipeline Summary ===")
    print(f"Candidate species: {n_species}")
    print(f"Species with LER pair data: {_count_species_with_data(ler_stats)}")
    print(f"Observed pairs: {n_observed} / {n_pairs}")
    _print_top_interactions(j_matrix, confidence)

    return {
        "ler_stats": ler_stats,
        "j_matrix": j_matrix,
        "confidence": confidence,
        "diversity": diversity,
        "biases": biases_df,
    }


def _count_species_with_data(ler_stats: pd.DataFrame) -> int:
    all_species = set(ler_stats["species_a"]) | set(ler_stats["species_b"])
    return len(all_species)


def _print_top_interactions(
    j_matrix: pd.DataFrame, confidence: pd.DataFrame
) -> None:
    species = j_matrix.index.tolist()
    n = len(species)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if confidence.iloc[i, j] > 0:
                pairs.append(
                    (species[i], species[j], j_matrix.iloc[i, j], confidence.iloc[i, j])
                )

    if not pairs:
        print("No observed interactions.")
        return

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 10 complementary pairs (highest J_ij):")
    for a, b, j_val, conf in pairs[:10]:
        print(f"  {a:15s} + {b:15s}  J={j_val:+.4f}  conf={conf:.3f}")

    print("\nTop 10 competitive pairs (lowest J_ij):")
    for a, b, j_val, conf in pairs[-10:]:
        print(f"  {a:15s} + {b:15s}  J={j_val:+.4f}  conf={conf:.3f}")


if __name__ == "__main__":
    run_pipeline()
