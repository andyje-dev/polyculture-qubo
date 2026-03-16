"""Build the species interaction matrix (J_ij) for QUBO construction.

The interaction coefficient J_ij encodes how complementary species i and j are
when intercropped together. Positive J_ij = complementary (rewards co-selection),
negative J_ij = competitive (penalizes co-selection).

Coefficient derivation:
  1. From LER data: J_ij = confidence_weight * (mean_LER - 1.0)
     - LER > 1.0 → positive interaction (land-use efficient)
     - LER < 1.0 → negative interaction (competitive)

  2. From companion planting data: used as a weak prior when LER data unavailable

  3. For unknown pairs: mildly negative prior (species are more likely
     to compete than complement without specific trait complementarity)
"""

import numpy as np
import pandas as pd

from polyculture_qubo.species import CANDIDATE_SPECIES, Species


def compute_confidence_weight(
    n_obs: int,
    ler_std: float,
    n_articles: int,
    max_obs: int = 50,
) -> float:
    """Compute a confidence weight in [0, 1] for a species pair interaction coefficient.

    Factors:
      - More observations → higher confidence (saturates around max_obs)
      - Lower variance → higher confidence
      - More independent articles → higher confidence (replication bonus)
    """
    # Observation count factor: sqrt saturation
    obs_factor = min(1.0, np.sqrt(n_obs / max_obs))

    # Variance factor: penalize high uncertainty
    # Use inverse coefficient of variation, capped
    if ler_std > 0:
        precision = 1.0 / (1.0 + ler_std)
    else:
        precision = 1.0
    var_factor = precision

    # Replication factor: bonus for multiple independent studies
    rep_factor = min(1.0, np.sqrt(n_articles / 5.0))

    return obs_factor * var_factor * rep_factor


def build_interaction_matrix(
    ler_stats: pd.DataFrame,
    companion_data: pd.DataFrame | None = None,
    bischoff_pairs: pd.DataFrame | None = None,
    unknown_pair_prior: float = -0.05,
    studied_pair_prior: float = -0.01,
    companion_weight: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the J_ij interaction matrix from LER statistics and companion data.

    Args:
        ler_stats: Output of compute_pairwise_ler_stats() with columns:
            species_a, species_b, ler_mean, ler_std, n_obs, n_articles
        companion_data: Optional companion planting data with columns:
            crop_1, crop_2, interaction (+1/-1)
        bischoff_pairs: Optional Bischoff et al. aggregated pair data with columns:
            species_a, species_b, n_exps_total. For pairs without LER data,
            uses a less negative prior (studied but no yield data available).
        unknown_pair_prior: Default J_ij for pairs with no data (mildly negative)
        studied_pair_prior: J_ij for pairs studied in Bischoff but without LER data
            (less negative than unknown — researchers chose to study these pairs)
        companion_weight: Weight for companion planting data as prior

    Returns:
        (J_matrix, confidence_matrix): Both as DataFrames indexed by species keys
    """
    species_keys = sorted(CANDIDATE_SPECIES.keys())
    n = len(species_keys)
    key_to_idx = {k: i for i, k in enumerate(species_keys)}

    j_matrix = np.full((n, n), unknown_pair_prior)
    confidence = np.zeros((n, n))

    # Fill diagonal with zeros (no self-interaction)
    np.fill_diagonal(j_matrix, 0.0)
    np.fill_diagonal(confidence, 1.0)

    # Layer 1: Apply Bischoff studied-pair prior (weakest signal)
    if bischoff_pairs is not None:
        _apply_bischoff_prior(
            j_matrix, confidence, bischoff_pairs, key_to_idx, studied_pair_prior
        )

    # Layer 2: Apply companion planting data as a weak prior
    if companion_data is not None:
        _apply_companion_prior(
            j_matrix, confidence, companion_data, key_to_idx, companion_weight
        )

    # Layer 3: Override with LER-derived coefficients (strongest signal)
    for _, row in ler_stats.iterrows():
        a, b = row["species_a"], row["species_b"]
        if a not in key_to_idx or b not in key_to_idx:
            continue

        i, j = key_to_idx[a], key_to_idx[b]

        # J_ij = confidence * (LER - 1.0)
        # LER > 1 → positive J (complementary), LER < 1 → negative J (competitive)
        conf = compute_confidence_weight(
            n_obs=int(row["n_obs"]),
            ler_std=float(row["ler_std"]),
            n_articles=int(row["n_articles"]),
        )
        j_val = conf * (row["ler_mean"] - 1.0)

        j_matrix[i, j] = j_val
        j_matrix[j, i] = j_val
        confidence[i, j] = conf
        confidence[j, i] = conf

    j_df = pd.DataFrame(j_matrix, index=species_keys, columns=species_keys)
    conf_df = pd.DataFrame(confidence, index=species_keys, columns=species_keys)

    return j_df, conf_df


def _apply_bischoff_prior(
    j_matrix: np.ndarray,
    confidence: np.ndarray,
    bischoff_pairs: pd.DataFrame,
    key_to_idx: dict[str, int],
    studied_pair_prior: float,
) -> None:
    """Apply Bischoff studied-pair prior to the interaction matrix.

    For pairs that appear in the Bischoff dataset but have no LER data,
    use a less negative prior than completely unknown pairs. The rationale:
    researchers chose to study these pairs, suggesting they were considered
    potentially complementary.
    """
    for _, row in bischoff_pairs.iterrows():
        a, b = row["species_a"], row["species_b"]
        if a not in key_to_idx or b not in key_to_idx:
            continue
        if a == b:
            continue

        i, j = key_to_idx[a], key_to_idx[b]

        # Scale prior slightly by experiment count (more studied = slightly more confidence)
        n_exps = int(row["n_exps_total"])
        conf = min(0.15, 0.05 * np.sqrt(n_exps / 5.0))

        j_matrix[i, j] = studied_pair_prior
        j_matrix[j, i] = studied_pair_prior
        confidence[i, j] = conf
        confidence[j, i] = conf


def _apply_companion_prior(
    j_matrix: np.ndarray,
    confidence: np.ndarray,
    companion_data: pd.DataFrame,
    key_to_idx: dict[str, int],
    weight: float,
) -> None:
    """Apply companion planting data as a weak prior to the interaction matrix."""
    valid = companion_data.dropna(subset=["crop_1", "crop_2", "interaction"]).copy()

    # Normalize to undirected pairs (alphabetical order) before aggregating,
    # so (maize, bean) and (bean, maize) are treated as the same pair.
    valid["species_a"] = np.minimum(valid["crop_1"], valid["crop_2"])
    valid["species_b"] = np.maximum(valid["crop_1"], valid["crop_2"])

    for (a, b), group in valid.groupby(["species_a", "species_b"]):
        if a not in key_to_idx or b not in key_to_idx:
            continue
        if a == b:
            continue

        i, j = key_to_idx[a], key_to_idx[b]
        mean_interaction = group["interaction"].mean()

        # Only apply as prior if we haven't seen LER data for this pair
        # (LER data will overwrite in build_interaction_matrix)
        j_matrix[i, j] = weight * mean_interaction
        j_matrix[j, i] = weight * mean_interaction
        confidence[i, j] = 0.1  # Low confidence for companion-only data
        confidence[j, i] = 0.1


def compute_linear_biases(
    species_keys: list[str] | None = None,
    yield_weight: float = 0.3,
) -> dict[str, float]:
    """Compute linear bias h_i for each species.

    Encodes individual species value based on economic importance.
    N-fixation is handled separately as a pairwise term in the QUBO
    (fixer + non-fixer pairs get a bonus).
    """
    if species_keys is None:
        species_keys = sorted(CANDIDATE_SPECIES.keys())

    biases: dict[str, float] = {}
    for key in species_keys:
        sp = CANDIDATE_SPECIES[key]
        biases[key] = yield_weight * (1.0 if sp.is_cash_crop else 0.3)

    return biases


def compute_diversity_bonus(sp_a: Species, sp_b: Species) -> float:
    """Compute functional diversity bonus for a species pair.

    Based on trait complementarity using quantitative trait differences:
      - Root depth difference → bonus (reduced belowground competition)
      - Height difference → bonus (light partitioning)
      - Different functional groups → bonus

    All components normalized to [0, 1] range, then weighted.
    """
    # Normalization constants (approximate ranges across candidate pool)
    max_height_diff = 2.0  # meters (lettuce 0.25 vs maize 2.2)
    max_root_diff = 1.0  # meters (lettuce 0.4 vs sorghum 1.5)

    height_diff = abs(sp_a.height_m - sp_b.height_m)
    root_diff = abs(sp_a.root_depth_m - sp_b.root_depth_m)

    height_score = min(1.0, height_diff / max_height_diff)
    root_score = min(1.0, root_diff / max_root_diff)
    group_score = 1.0 if sp_a.functional_group != sp_b.functional_group else 0.0

    return 0.3 * root_score + 0.3 * height_score + 0.4 * group_score


def build_diversity_matrix(species_keys: list[str] | None = None) -> pd.DataFrame:
    """Build pairwise functional diversity matrix."""
    if species_keys is None:
        species_keys = sorted(CANDIDATE_SPECIES.keys())

    n = len(species_keys)
    div_matrix = np.zeros((n, n))

    for i, key_i in enumerate(species_keys):
        sp_i = CANDIDATE_SPECIES[key_i]
        for j, key_j in enumerate(species_keys):
            if i >= j:
                continue
            sp_j = CANDIDATE_SPECIES[key_j]
            div_matrix[i, j] = compute_diversity_bonus(sp_i, sp_j)
            div_matrix[j, i] = div_matrix[i, j]

    return pd.DataFrame(div_matrix, index=species_keys, columns=species_keys)
