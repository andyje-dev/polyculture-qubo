"""Tests for data loading and preprocessing pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polyculture_qubo.species import CANDIDATE_SPECIES, normalize_species_name
from polyculture_qubo.data import (
    aggregate_bischoff_pairs,
    compute_pairwise_ler_stats,
    filter_to_candidate_pairs,
    load_bischoff_pairs,
    load_paut_dataset,
)
from polyculture_qubo.matrix import (
    build_diversity_matrix,
    build_interaction_matrix,
    compute_confidence_weight,
    compute_linear_biases,
)

RAW_DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "paut_database.csv"
BISCHOFF_DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "bischoff_species_pairs.csv"
SKIP_NO_DATA = pytest.mark.skipif(
    not RAW_DATA.exists(), reason="Raw data not downloaded"
)
SKIP_NO_BISCHOFF = pytest.mark.skipif(
    not BISCHOFF_DATA.exists(), reason="Bischoff data not extracted"
)


class TestSpeciesNormalization:
    def test_exact_match(self):
        assert normalize_species_name("Maize") == "maize"
        assert normalize_species_name("Cowpea") == "cowpea"

    def test_case_insensitive(self):
        assert normalize_species_name("MAIZE") == "maize"
        assert normalize_species_name("maize") == "maize"

    def test_aliases(self):
        assert normalize_species_name("Faba bean") == "fava_bean"
        assert normalize_species_name("Fava bean") == "fava_bean"
        assert normalize_species_name("Broad bean") == "fava_bean"
        assert normalize_species_name("Corn") == "maize"

    def test_unknown_species(self):
        assert normalize_species_name("Dragonfruit") is None
        assert normalize_species_name("Banana") is None

    def test_whitespace_stripped(self):
        assert normalize_species_name("  Maize  ") == "maize"

    def test_all_candidates_self_resolve(self):
        for key, sp in CANDIDATE_SPECIES.items():
            resolved = normalize_species_name(sp.common_name)
            assert resolved == key, f"{sp.common_name} should resolve to {key}, got {resolved}"


class TestConfidenceWeight:
    def test_higher_obs_higher_confidence(self):
        w1 = compute_confidence_weight(n_obs=2, ler_std=0.2, n_articles=1)
        w2 = compute_confidence_weight(n_obs=20, ler_std=0.2, n_articles=1)
        assert w2 > w1

    def test_lower_std_higher_confidence(self):
        w1 = compute_confidence_weight(n_obs=10, ler_std=0.5, n_articles=2)
        w2 = compute_confidence_weight(n_obs=10, ler_std=0.1, n_articles=2)
        assert w2 > w1

    def test_more_articles_higher_confidence(self):
        w1 = compute_confidence_weight(n_obs=10, ler_std=0.2, n_articles=1)
        w2 = compute_confidence_weight(n_obs=10, ler_std=0.2, n_articles=5)
        assert w2 > w1

    def test_range_zero_to_one(self):
        w = compute_confidence_weight(n_obs=100, ler_std=0.01, n_articles=10)
        assert 0.0 <= w <= 1.0
        w = compute_confidence_weight(n_obs=1, ler_std=1.0, n_articles=1)
        assert 0.0 <= w <= 1.0


class TestInteractionMatrix:
    def test_symmetry(self):
        # Create minimal LER stats
        stats = pd.DataFrame(
            {
                "species_a": ["cowpea"],
                "species_b": ["maize"],
                "ler_mean": [1.3],
                "ler_std": [0.2],
                "n_obs": [10],
                "n_articles": [3],
                "pct_above_1": [0.8],
            }
        )
        j, conf = build_interaction_matrix(stats)
        assert j.loc["cowpea", "maize"] == j.loc["maize", "cowpea"]
        assert conf.loc["cowpea", "maize"] == conf.loc["maize", "cowpea"]

    def test_diagonal_zero(self):
        stats = pd.DataFrame(columns=["species_a", "species_b", "ler_mean", "ler_std", "n_obs", "n_articles", "pct_above_1"])
        j, _ = build_interaction_matrix(stats)
        for sp in j.index:
            assert j.loc[sp, sp] == 0.0

    def test_complementary_positive(self):
        stats = pd.DataFrame(
            {
                "species_a": ["cowpea"],
                "species_b": ["maize"],
                "ler_mean": [1.5],  # LER > 1 → complementary
                "ler_std": [0.1],
                "n_obs": [20],
                "n_articles": [5],
                "pct_above_1": [0.9],
            }
        )
        j, _ = build_interaction_matrix(stats)
        assert j.loc["cowpea", "maize"] > 0

    def test_competitive_negative(self):
        stats = pd.DataFrame(
            {
                "species_a": ["cowpea"],
                "species_b": ["maize"],
                "ler_mean": [0.7],  # LER < 1 → competitive
                "ler_std": [0.1],
                "n_obs": [20],
                "n_articles": [5],
                "pct_above_1": [0.1],
            }
        )
        j, _ = build_interaction_matrix(stats)
        assert j.loc["cowpea", "maize"] < 0

    def test_unknown_pairs_get_prior(self):
        stats = pd.DataFrame(columns=["species_a", "species_b", "ler_mean", "ler_std", "n_obs", "n_articles", "pct_above_1"])
        prior = -0.03
        j, _ = build_interaction_matrix(stats, unknown_pair_prior=prior)
        # Check an arbitrary non-diagonal pair
        assert j.loc["barley", "sunflower"] == prior

    def test_matrix_shape(self):
        stats = pd.DataFrame(columns=["species_a", "species_b", "ler_mean", "ler_std", "n_obs", "n_articles", "pct_above_1"])
        j, conf = build_interaction_matrix(stats)
        n = len(CANDIDATE_SPECIES)
        assert j.shape == (n, n)
        assert conf.shape == (n, n)


class TestDiversityMatrix:
    def test_symmetry(self):
        div = build_diversity_matrix()
        np.testing.assert_array_equal(div.values, div.values.T)

    def test_diagonal_zero(self):
        div = build_diversity_matrix()
        np.testing.assert_array_equal(np.diag(div.values), 0.0)

    def test_different_groups_higher(self):
        div = build_diversity_matrix()
        # Maize (cereal, tall, deep) vs Pea (legume, low, shallow) should have high diversity
        # Maize vs Wheat (both cereals) should have lower diversity
        assert div.loc["maize", "pea"] > div.loc["maize", "wheat"]


class TestLinearBiases:
    def test_cash_crop_higher(self):
        biases = compute_linear_biases()
        # Cash crop should have higher bias than non-cash crop
        assert biases["maize"] > biases["crimson_clover"]

    def test_cash_crops_equal(self):
        biases = compute_linear_biases()
        # N-fixation no longer in linear biases — all cash crops equal
        assert biases["cowpea"] == biases["wheat"]


@SKIP_NO_DATA
class TestFullPipeline:
    def test_load_paut(self):
        df = load_paut_dataset()
        assert len(df) > 2000
        assert "ler_total" in df.columns
        assert "crop_1" in df.columns

    def test_filter_candidates(self):
        df = load_paut_dataset()
        filtered = filter_to_candidate_pairs(df)
        assert len(filtered) < len(df)
        assert filtered["crop_1"].notna().all()
        assert filtered["crop_2"].notna().all()

    def test_pairwise_stats(self):
        df = load_paut_dataset()
        filtered = filter_to_candidate_pairs(df)
        stats = compute_pairwise_ler_stats(filtered)
        assert len(stats) > 0
        assert "ler_mean" in stats.columns
        assert "n_obs" in stats.columns
        # Maize + bean should be the most studied pair
        top = stats.iloc[0]
        assert {top["species_a"], top["species_b"]} == {"common_bean", "maize"}


@SKIP_NO_BISCHOFF
class TestBischoffExtraction:
    def test_load_bischoff(self):
        df = load_bischoff_pairs()
        assert len(df) >= 270
        assert "crop_1" in df.columns
        assert "n_exps" in df.columns

    def test_filter_and_aggregate(self):
        df = load_bischoff_pairs()
        filtered = filter_to_candidate_pairs(df)
        assert len(filtered) > 0
        agg = aggregate_bischoff_pairs(filtered)
        assert "species_a" in agg.columns
        assert "n_exps_total" in agg.columns
        # All pairs should be undirected (a < b alphabetically)
        assert (agg["species_a"] < agg["species_b"]).all()

    def test_bischoff_boosts_coverage(self):
        """Bischoff data should increase pair coverage vs LER-only."""
        stats = pd.DataFrame(
            columns=["species_a", "species_b", "ler_mean", "ler_std",
                      "n_obs", "n_articles", "pct_above_1"]
        )
        # Without Bischoff: only unknown prior
        j_no, conf_no = build_interaction_matrix(stats)
        n = len(CANDIDATE_SPECIES)
        n_no = (conf_no.values[np.triu_indices(n, k=1)] > 0).sum()

        # With Bischoff: studied pairs get a prior
        bischoff = load_bischoff_pairs()
        bischoff_filt = filter_to_candidate_pairs(bischoff)
        bischoff_agg = aggregate_bischoff_pairs(bischoff_filt)
        j_with, conf_with = build_interaction_matrix(stats, bischoff_pairs=bischoff_agg)
        n_with = (conf_with.values[np.triu_indices(n, k=1)] > 0).sum()

        assert n_with > n_no
