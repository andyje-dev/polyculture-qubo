"""Tests for offset-invariant solution quality metrics."""

import itertools

import numpy as np
import pytest

from polyculture_qubo.matrix.qubo import qubo_energy
from polyculture_qubo.solvers.metrics import (
    beta_quality,
    evaluate_quality,
    random_baseline,
    sample_feasible_energies,
)


def _make_small_qubo() -> np.ndarray:
    """5-variable QUBO with k=2, matching the fixture used in test_solvers."""
    q = np.zeros((5, 5))
    q[0, 0] = -0.1 - 15
    q[1, 1] = -0.5 - 15
    q[2, 2] = -0.1 - 15
    q[3, 3] = -0.5 - 15
    q[4, 4] = -0.1 - 15
    for i in range(5):
        for j in range(i + 1, 5):
            q[i, j] = 10.0
    q[1, 3] = 8.0
    return q


def _make_generic_qubo(n: int = 8, seed: int = 0) -> np.ndarray:
    """Upper-triangular QUBO with distinct feasible energies (no rank ties)."""
    rng = np.random.default_rng(seed)
    return np.triu(rng.normal(size=(n, n)))


def _enumerate(q: np.ndarray, k: int) -> np.ndarray:
    n = q.shape[0]
    out = []
    for combo in itertools.combinations(range(n), k):
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1
        out.append(qubo_energy(q, x))
    return np.array(out)


def _shift_energy(q: np.ndarray, k: int, c: float) -> np.ndarray:
    """Return a QUBO whose feasible energies are all shifted by exactly c.

    Adding c/k to every diagonal entry adds c to any solution selecting
    exactly k variables. This is the transformation the QUBO problem is
    invariant under and the naive energy ratio is not.
    """
    shifted = q.copy()
    np.fill_diagonal(shifted, np.diag(shifted) + c / k)
    return shifted


class TestSampleFeasibleEnergies:
    def test_sample_count_and_validity(self):
        q = _make_small_qubo()
        energies = sample_feasible_energies(q, target_species=2, n_samples=500, seed=1)
        assert len(energies) == 500
        # Every sampled energy must be an energy of some feasible solution.
        feasible = set(np.round(_enumerate(q, 2), 9))
        assert set(np.round(energies, 9)).issubset(feasible)

    def test_matches_scalar_energy_routine(self):
        """Vectorized energies must equal the canonical qubo_energy."""
        rng = np.random.default_rng(0)
        q = np.triu(rng.normal(size=(8, 8)))
        energies = sample_feasible_energies(q, target_species=3, n_samples=200, seed=3)
        feasible = np.sort(_enumerate(q, 3))
        for e in energies:
            assert np.min(np.abs(feasible - e)) < 1e-9

    def test_sampling_is_seeded(self):
        q = _make_small_qubo()
        a = sample_feasible_energies(q, 2, n_samples=100, seed=7)
        b = sample_feasible_energies(q, 2, n_samples=100, seed=7)
        assert np.array_equal(a, b)

    def test_rejects_bad_arguments(self):
        q = _make_small_qubo()
        with pytest.raises(ValueError):
            sample_feasible_energies(q, target_species=0, n_samples=10)
        with pytest.raises(ValueError):
            sample_feasible_energies(q, target_species=99, n_samples=10)
        with pytest.raises(ValueError):
            sample_feasible_energies(q, target_species=2, n_samples=0)


class TestRandomBaseline:
    def test_exhaustive_matches_enumeration(self):
        q = _make_small_qubo()
        base = random_baseline(q, target_species=2, method="exhaustive")
        expected = _enumerate(q, 2)
        assert base.exhaustive
        assert base.stderr == 0.0
        assert base.mean_energy == pytest.approx(expected.mean())
        assert base.n_samples == len(expected)

    def test_sampled_agrees_with_exhaustive(self):
        """The sampled estimator must be unbiased for the exhaustive mean."""
        q = _make_small_qubo()
        exact = random_baseline(q, 2, method="exhaustive")
        sampled = random_baseline(q, 2, n_samples=50_000, seed=11, method="sample")
        assert not sampled.exhaustive
        assert sampled.stderr > 0
        # Within 4 standard errors is a very loose bound for an unbiased estimator.
        assert abs(sampled.mean_energy - exact.mean_energy) < 4 * sampled.stderr

    def test_auto_enumerates_small_feasible_sets(self):
        q = _make_small_qubo()
        assert random_baseline(q, 2, n_samples=10_000, method="auto").exhaustive

    def test_auto_samples_when_feasible_set_is_large(self):
        q = _make_small_qubo()
        # C(5, 2) = 10, so a budget of 5 forces the sampling path.
        assert not random_baseline(q, 2, n_samples=5, method="auto").exhaustive

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError):
            random_baseline(_make_small_qubo(), 2, method="nonsense")


class TestBetaQuality:
    def test_optimum_scores_one(self):
        assert beta_quality(-10.0, -10.0, -5.0) == pytest.approx(1.0)

    def test_random_scores_zero(self):
        assert beta_quality(-5.0, -10.0, -5.0) == pytest.approx(0.0)

    def test_worse_than_random_is_negative(self):
        assert beta_quality(-2.0, -10.0, -5.0) < 0

    def test_degenerate_landscape_is_nan(self):
        assert np.isnan(beta_quality(-5.0, -5.0, -5.0))


class TestOffsetInvariance:
    """The core property: the problem is offset-invariant, so the metric must be."""

    def test_beta_is_invariant_under_energy_shift(self):
        q = _make_small_qubo()
        k = 2
        shifted = _shift_energy(q, k, c=-1000.0)

        e_orig = _enumerate(q, k)
        e_shift = _enumerate(shifted, k)
        # Confirm the shift did what the docstring claims.
        assert np.allclose(np.sort(e_shift) - np.sort(e_orig), -1000.0)

        chosen = np.sort(e_orig)[2]  # some mid-quality solution
        q1 = evaluate_quality(
            q, k, float(chosen), float(e_orig.min()), all_feasible_energies=e_orig
        )
        q2 = evaluate_quality(
            shifted,
            k,
            float(chosen - 1000.0),
            float(e_shift.min()),
            all_feasible_energies=e_shift,
        )
        assert q1.beta == pytest.approx(q2.beta)
        assert q1.rank == q2.rank

    def test_total_energy_ratio_is_not_invariant(self):
        """Regression guard: documents why the raw ratio was replaced.

        The same solution on the same problem scores differently once a
        constant is added to every feasible energy. This is the defect the
        beta metric exists to fix.
        """
        q = _make_small_qubo()
        k = 2
        shifted = _shift_energy(q, k, c=-1000.0)
        e_orig = _enumerate(q, k)
        e_shift = _enumerate(shifted, k)
        chosen = np.sort(e_orig)[2]

        r1 = evaluate_quality(
            q, k, float(chosen), float(e_orig.min()), all_feasible_energies=e_orig
        ).total_energy_ratio
        r2 = evaluate_quality(
            shifted,
            k,
            float(chosen - 1000.0),
            float(e_shift.min()),
            all_feasible_energies=e_shift,
        ).total_energy_ratio
        assert r1 != pytest.approx(r2)
        # The shifted ratio is compressed toward 1, which is the reported bug.
        assert abs(r2 - 1.0) < abs(r1 - 1.0)


class TestEvaluateQuality:
    def test_rank_and_optimum(self):
        q = _make_small_qubo()
        energies = _enumerate(q, 2)
        best = float(energies.min())
        qual = evaluate_quality(q, 2, best, best, all_feasible_energies=energies)
        assert qual.rank == 1
        assert qual.beta == pytest.approx(1.0)
        assert qual.percentile_better == pytest.approx(0.0)
        assert qual.n_feasible == len(energies)

    def test_worst_solution_ranks_last(self):
        q = _make_generic_qubo()
        energies = _enumerate(q, 3)
        assert len(set(np.round(energies, 9))) == len(energies), (
            "fixture must be tie-free"
        )
        worst = float(energies.max())
        qual = evaluate_quality(
            q, 3, worst, float(energies.min()), all_feasible_energies=energies
        )
        assert qual.rank == len(energies)
        assert qual.beta < 0

    def test_ties_use_competition_ranking(self):
        """Equal-energy solutions share a rank: 1 + (number strictly better).

        This is what "5th of 4,845" means in the report — it counts solutions
        that beat this one, not the position in an arbitrary sort order.
        """
        q = _make_small_qubo()
        energies = _enumerate(q, 2)
        worst = float(energies.max())
        n_strictly_better = int(np.sum(energies < worst - 1e-9))
        qual = evaluate_quality(
            q, 2, worst, float(energies.min()), all_feasible_energies=energies
        )
        assert qual.rank is not None
        assert qual.rank == n_strictly_better + 1
        assert qual.rank < len(energies)  # ties compress the rank

    def test_ratio_floor_is_reported(self):
        q = _make_small_qubo()
        energies = _enumerate(q, 2)
        qual = evaluate_quality(
            q,
            2,
            float(energies.min()),
            float(energies.min()),
            all_feasible_energies=energies,
        )
        assert qual.total_energy_ratio_floor == pytest.approx(
            energies.max() / energies.min()
        )

    def test_objective_ratio_strips_penalty_offset(self):
        q = _make_small_qubo()
        k, lam = 2, 5.0
        offset = -(k**2) * lam
        energies = _enumerate(q, k)
        chosen = float(np.sort(energies)[2])
        qual = evaluate_quality(
            q,
            k,
            chosen,
            float(energies.min()),
            all_feasible_energies=energies,
            penalty_offset=offset,
        )
        expected = (chosen - offset) / (energies.min() - offset)
        assert qual.objective_ratio is not None
        assert qual.objective_ratio == pytest.approx(expected)
        # Stripping the constant must widen the metric's spread.
        assert abs(qual.objective_ratio - 1.0) > abs(qual.total_energy_ratio - 1.0)

    def test_percentile_without_enumeration(self):
        """Without a full enumeration, rank is unavailable but percentile is estimated."""
        q = _make_generic_qubo()
        energies = _enumerate(q, 3)
        median = float(np.median(energies))
        exact_pct = float(np.sum(energies < median - 1e-9) / len(energies))
        qual = evaluate_quality(q, 3, median, float(energies.min()), n_samples=20_000)
        assert qual.rank is None
        assert qual.n_feasible is None
        assert qual.percentile_better == pytest.approx(exact_pct, abs=0.02)

    def test_beta_stderr_zero_for_exhaustive_baseline(self):
        q = _make_small_qubo()
        energies = _enumerate(q, 2)
        base = random_baseline(q, 2, method="exhaustive")
        qual = evaluate_quality(
            q,
            2,
            float(np.sort(energies)[2]),
            float(energies.min()),
            baseline=base,
            all_feasible_energies=energies,
        )
        assert qual.beta_stderr == 0.0

    def test_beta_stderr_positive_for_sampled_baseline(self):
        q = _make_small_qubo()
        energies = _enumerate(q, 2)
        base = random_baseline(q, 2, n_samples=2_000, seed=5, method="sample")
        qual = evaluate_quality(
            q,
            2,
            float(np.sort(energies)[2]),
            float(energies.min()),
            baseline=base,
            all_feasible_energies=energies,
        )
        assert qual.beta_stderr > 0
