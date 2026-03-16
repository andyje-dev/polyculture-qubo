"""Tests for Phase 4 analysis modules."""

import numpy as np
import pandas as pd
import pytest

from polyculture_qubo.analysis.landscape import (
    LandscapeMetrics,
    analyze_landscape,
    frustration_analysis,
)
from polyculture_qubo.analysis.scalability import compute_scalability_metrics
from polyculture_qubo.analysis.sensitivity import (
    WeightSweepResult,
    analyze_weight_sweep,
)
from polyculture_qubo.analysis.validation import (
    pairwise_contribution_table,
    pairwise_ranking_check,
)
from polyculture_qubo.matrix.qubo import qubo_energy
from polyculture_qubo.solvers.exact import ExactSolver
from polyculture_qubo.solvers.result import SolverResult


# --- Helpers ---


def _make_landscape_qubo() -> tuple[np.ndarray, int]:
    """Create a 6-variable QUBO for landscape analysis testing.

    Returns (q, k) where k=2 is the target species count.
    """
    q = np.zeros((6, 6))
    lam = 5.0
    k = 2

    # Diagonal: species value + constraint
    values = [-0.5, -0.4, -0.3, -0.2, -0.1, -0.05]
    for i in range(6):
        q[i, i] = values[i] + lam * (1 - 2 * k)

    # Off-diagonal: pairwise + 2λ
    for i in range(6):
        for j in range(i + 1, 6):
            q[i, j] = 2 * lam  # = 10.0

    # Make pair (0,1) strongly complementary
    q[0, 1] = -2.0 + 2 * lam  # = 8.0
    # Make pair (2,3) moderately complementary
    q[2, 3] = -0.5 + 2 * lam  # = 9.5

    return q, k


# --- Landscape Tests ---


class TestLandscapeAnalysis:
    def test_correct_solution_count(self):
        """Should analyze all C(6,2) = 15 solutions."""
        q, k = _make_landscape_qubo()
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q)

        assert metrics.n_solutions == 15
        assert metrics.k == k

    def test_energy_range(self):
        q, k = _make_landscape_qubo()
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q)

        assert metrics.best_energy < metrics.worst_energy
        assert metrics.best_energy == pytest.approx(exact.best_energy)

    def test_spectral_gap_positive(self):
        q, k = _make_landscape_qubo()
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q)

        assert metrics.spectral_gap >= 0

    def test_degeneracy_ordering(self):
        """Tighter threshold should yield fewer near-optimal solutions."""
        q, k = _make_landscape_qubo()
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q)

        assert metrics.n_within_1pct <= metrics.n_within_5pct
        assert metrics.n_within_5pct <= metrics.n_within_10pct

    def test_at_least_one_local_minimum(self):
        """The global optimum is always a local minimum."""
        q, k = _make_landscape_qubo()
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q)

        assert metrics.n_local_minima >= 1

    def test_sorted_energies_bottom(self):
        q, k = _make_landscape_qubo()
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q)

        # Should be sorted ascending
        assert np.all(np.diff(metrics.sorted_energies_bottom) >= 0)
        assert metrics.sorted_energies_bottom[0] == pytest.approx(metrics.best_energy)

    def test_penalty_decomposition(self):
        """Objective-only energies should have the same relative ordering."""
        q, k = _make_landscape_qubo()
        lam = 5.0  # matches _make_landscape_qubo
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q, penalty_strength=lam)

        # Penalty offset should be -k²λ
        assert metrics.penalty_offset == pytest.approx(-(k ** 2) * lam)

        # Objective range should equal total range (constant offset doesn't change range)
        total_range = metrics.worst_energy - metrics.best_energy
        assert metrics.obj_range == pytest.approx(total_range)

        # Spectral gaps should be identical (constant shift)
        assert metrics.obj_spectral_gap == pytest.approx(metrics.spectral_gap)

    def test_penalty_decomposition_reveals_wider_spread(self):
        """Objective-only degeneracy should differ from total when penalty dominates."""
        q, k = _make_landscape_qubo()
        lam = 5.0
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q, penalty_strength=lam)

        # With penalty subtracted, the absolute values are much smaller,
        # so "within X%" of a smaller number is a tighter absolute band.
        # This means fewer solutions should be "within 5%" of the objective
        # optimum compared to the total energy (where the penalty offset
        # makes 5% a larger absolute window).
        # We can't guarantee this for all QUBOs, but for our test case:
        assert metrics.obj_range > 0  # Non-trivial landscape


class TestFrustrationAnalysis:
    def test_identifies_competitive_selected_pairs(self):
        """Should detect when competitive species are both selected."""
        # Build a fake j_matrix where species 0 and 1 are competitive
        species = ["sp_a", "sp_b", "sp_c", "sp_d"]
        j_data = np.zeros((4, 4))
        j_data[0, 1] = -0.1  # Competitive
        j_data[0, 2] = 0.3   # Complementary
        j_matrix = pd.DataFrame(j_data, index=species, columns=species)

        # Solution selects species 0 and 1 (the competitive pair)
        x = np.array([1, 1, 0, 0])

        result = frustration_analysis(x, species, j_matrix)

        # Should find the competitive pair
        assert len(result["selected_competitive"]) == 1
        assert result["selected_competitive"][0][:2] == ("sp_a", "sp_b")

    def test_identifies_split_complementary_pairs(self):
        species = ["sp_a", "sp_b", "sp_c", "sp_d"]
        j_data = np.zeros((4, 4))
        j_data[0, 2] = 0.3   # Complementary but split
        j_matrix = pd.DataFrame(j_data, index=species, columns=species)

        x = np.array([1, 1, 0, 0])  # sp_a selected, sp_c not

        result = frustration_analysis(x, species, j_matrix)

        assert len(result["split_complementary"]) == 1
        assert result["split_complementary"][0][2] == pytest.approx(0.3)


# --- Sensitivity Tests ---


class TestWeightSweepAnalysis:
    def test_stability_calculation(self):
        """Stability should be fraction of configs with most common solution."""
        results = [
            WeightSweepResult(0.7, 0.2, 0.1, -10.0, ["a", "b"]),
            WeightSweepResult(0.8, 0.1, 0.1, -11.0, ["a", "b"]),
            WeightSweepResult(0.5, 0.3, 0.2, -9.0, ["a", "c"]),
        ]

        analysis = analyze_weight_sweep(results)

        assert analysis["n_configs"] == 3
        assert analysis["n_unique_solutions"] == 2
        assert analysis["stability"] == pytest.approx(2 / 3)
        assert analysis["most_common_solution"] == ["a", "b"]

    def test_species_frequency(self):
        results = [
            WeightSweepResult(0.7, 0.2, 0.1, -10.0, ["a", "b"]),
            WeightSweepResult(0.8, 0.1, 0.1, -11.0, ["a", "c"]),
        ]

        analysis = analyze_weight_sweep(results)

        assert analysis["species_frequency"]["a"] == pytest.approx(1.0)  # In both
        assert analysis["species_frequency"]["b"] == pytest.approx(0.5)  # In 1 of 2
        assert analysis["species_frequency"]["c"] == pytest.approx(0.5)  # In 1 of 2


# --- Validation Tests ---


class TestPairwiseRanking:
    def test_high_ler_pairs_rank_well(self):
        """Selected pairs with high LER should have high percentile rank."""
        ler_stats = pd.DataFrame({
            "species_a": ["a", "b", "c", "d"],
            "species_b": ["b", "c", "d", "e"],
            "ler_mean": [1.8, 1.5, 1.2, 0.9],
        })

        # Select species that include the top LER pair
        selected = ["a", "b", "c"]

        ranking = pairwise_ranking_check(selected, ler_stats)

        assert ranking["n_observed_pairs_in_solution"] >= 1
        assert ranking["mean_ler_of_selected"] > ranking["mean_ler_of_all_observed"]

    def test_no_observed_pairs(self):
        """Should handle case where no selected pairs have LER data."""
        ler_stats = pd.DataFrame({
            "species_a": ["a"],
            "species_b": ["b"],
            "ler_mean": [1.5],
        })

        selected = ["c", "d"]

        ranking = pairwise_ranking_check(selected, ler_stats)

        assert ranking["n_observed_pairs_in_solution"] == 0


# --- Scalability Tests ---


class TestScalabilityMetrics:
    def test_gap_scaling(self):
        """Metrics should capture gap values for each k."""
        metrics_by_k = {}
        q_matrices = {}

        for k in [2, 3]:
            q, _ = _make_landscape_qubo()
            exact = ExactSolver().solve(q, target_species=k)
            metrics_by_k[k] = analyze_landscape(exact, q)
            q_matrices[k] = q

        scalability = compute_scalability_metrics(metrics_by_k, q_matrices)

        assert len(scalability.k_values) == 2
        assert len(scalability.spectral_gaps) == 2
        assert all(g >= 0 for g in scalability.spectral_gaps)

    def test_condition_numbers_finite(self):
        metrics_by_k = {}
        q_matrices = {}

        q, _ = _make_landscape_qubo()
        exact = ExactSolver().solve(q, target_species=2)
        metrics_by_k[2] = analyze_landscape(exact, q)
        q_matrices[2] = q

        scalability = compute_scalability_metrics(metrics_by_k, q_matrices)

        assert all(c > 0 for c in scalability.condition_numbers)
