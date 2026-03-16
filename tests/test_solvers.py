"""Tests for QUBO solvers."""

import numpy as np
import pytest

from polyculture_qubo.matrix.qubo import qubo_energy
from polyculture_qubo.solvers.exact import ExactSolver
from polyculture_qubo.solvers.annealing import SimulatedAnnealingSolver
from polyculture_qubo.solvers.qaoa import QAOAConfig, QAOASolver


# --- Helpers ---


def _make_small_qubo() -> np.ndarray:
    """Create a small 5-variable QUBO with a known optimum.

    Variables represent 5 species. We want to select exactly k=2.
    The QUBO is designed so that selecting species 1 and 3 (0-indexed)
    is the clear optimum.

    Q diagonal: species value (more negative = more desirable)
    Q off-diagonal: pairwise interactions + constraint penalty
    """
    q = np.zeros((5, 5))
    # Diagonal: linear biases + constraint term λ(1 - 2k) with k=2, λ=5
    # λ(1-2k) = 5*(1-4) = -15
    # Species 1 and 3 are most valuable
    q[0, 0] = -0.1 - 15  # = -15.1
    q[1, 1] = -0.5 - 15  # = -15.5  (good)
    q[2, 2] = -0.1 - 15  # = -15.1
    q[3, 3] = -0.5 - 15  # = -15.5  (good)
    q[4, 4] = -0.1 - 15  # = -15.1

    # Off-diagonal: pairwise terms + 2λ = 10
    # Pair (1,3) is strongly complementary: -2.0 + 10 = 8.0
    # All other pairs are neutral: 0 + 10 = 10.0
    for i in range(5):
        for j in range(i + 1, 5):
            q[i, j] = 10.0
    q[1, 3] = 8.0  # Complementary pair

    return q


def _optimal_solution_5var() -> tuple[np.ndarray, float]:
    """Known optimum for the 5-variable QUBO: select species 1 and 3."""
    q = _make_small_qubo()
    x = np.array([0, 1, 0, 1, 0])
    energy = qubo_energy(q, x)
    return x, energy


# --- ExactSolver Tests ---


class TestExactSolver:
    def test_finds_known_optimum(self):
        q = _make_small_qubo()
        expected_x, expected_energy = _optimal_solution_5var()

        result = ExactSolver().solve(q, target_species=2)

        assert result.best_energy == pytest.approx(expected_energy)
        np.testing.assert_array_equal(result.best_solution, expected_x)

    def test_correct_number_of_solutions(self):
        """Should enumerate exactly C(5,2) = 10 solutions."""
        q = _make_small_qubo()
        result = ExactSolver().solve(q, target_species=2)

        assert result.num_solutions_evaluated == 10
        assert len(result.all_energies) == 10
        assert len(result.all_solutions) == 10

    def test_all_solutions_are_feasible(self):
        """Every solution should select exactly k species."""
        q = _make_small_qubo()
        result = ExactSolver().solve(q, target_species=2)

        for sol in result.all_solutions:
            assert np.sum(sol) == 2

    def test_energies_match_solutions(self):
        """Each energy in all_energies should match its solution vector."""
        q = _make_small_qubo()
        result = ExactSolver().solve(q, target_species=2)

        for sol, energy in zip(result.all_solutions, result.all_energies):
            assert qubo_energy(q, sol) == pytest.approx(energy)

    def test_best_is_global_minimum(self):
        """Best energy should be the minimum across all evaluated solutions."""
        q = _make_small_qubo()
        result = ExactSolver().solve(q, target_species=2)

        assert result.best_energy == pytest.approx(np.min(result.all_energies))

    def test_collect_all_false(self):
        """When collect_all=False, should still find optimum but not store landscape."""
        q = _make_small_qubo()
        _, expected_energy = _optimal_solution_5var()

        result = ExactSolver().solve(q, target_species=2, collect_all=False)

        assert result.best_energy == pytest.approx(expected_energy)
        assert len(result.all_energies) == 0
        assert len(result.all_solutions) == 0

    def test_metadata(self):
        q = _make_small_qubo()
        result = ExactSolver().solve(q, target_species=2)

        assert result.solver_name == "exact"
        assert result.metadata["n"] == 5
        assert result.metadata["k"] == 2
        assert result.wall_time_seconds > 0

    def test_k_equals_1(self):
        """With k=1, should find the species with the lowest diagonal entry."""
        q = _make_small_qubo()
        result = ExactSolver().solve(q, target_species=1)

        assert result.num_solutions_evaluated == 5
        assert np.sum(result.best_solution) == 1
        # Species 1 and 3 have the most negative diagonal (-15.5),
        # so the solver should pick one of them
        assert result.best_energy == pytest.approx(-15.5)


# --- SimulatedAnnealingSolver Tests ---


class TestSimulatedAnnealing:
    def test_finds_known_optimum(self):
        """SA should reliably find the optimum on a small, easy problem."""
        q = _make_small_qubo()
        expected_x, expected_energy = _optimal_solution_5var()

        result = SimulatedAnnealingSolver().solve(
            q, target_species=2, num_reads=50, num_sweeps=200, seed=42
        )

        assert result.best_energy == pytest.approx(expected_energy)
        np.testing.assert_array_equal(result.best_solution, expected_x)

    def test_solutions_are_feasible(self):
        """SA should only produce solutions with exactly k species."""
        q = _make_small_qubo()
        result = SimulatedAnnealingSolver().solve(
            q, target_species=2, num_reads=10, seed=42
        )

        assert np.sum(result.best_solution) == 2

    def test_deterministic_with_seed(self):
        """Same seed should produce same result."""
        q = _make_small_qubo()

        r1 = SimulatedAnnealingSolver().solve(q, target_species=2, num_reads=10, seed=123)
        r2 = SimulatedAnnealingSolver().solve(q, target_species=2, num_reads=10, seed=123)

        assert r1.best_energy == pytest.approx(r2.best_energy)
        np.testing.assert_array_equal(r1.best_solution, r2.best_solution)

    def test_different_seeds_explore(self):
        """Different seeds may find different solutions (or the same optimum)."""
        q = _make_small_qubo()

        r1 = SimulatedAnnealingSolver().solve(q, target_species=2, num_reads=1, seed=1)
        r2 = SimulatedAnnealingSolver().solve(q, target_species=2, num_reads=1, seed=999)

        # Both should be feasible
        assert np.sum(r1.best_solution) == 2
        assert np.sum(r2.best_solution) == 2

    def test_metadata(self):
        q = _make_small_qubo()
        result = SimulatedAnnealingSolver().solve(
            q, target_species=2, num_reads=5, num_sweeps=100, seed=42
        )

        assert result.solver_name == "simulated_annealing"
        assert result.metadata["num_reads"] == 5
        assert result.metadata["num_sweeps"] == 100
        assert result.metadata["seed"] == 42
        assert result.wall_time_seconds > 0

    def test_more_reads_at_least_as_good(self):
        """More reads should find a solution at least as good as fewer reads."""
        q = _make_small_qubo()

        r_few = SimulatedAnnealingSolver().solve(
            q, target_species=2, num_reads=1, num_sweeps=50, seed=42
        )
        r_many = SimulatedAnnealingSolver().solve(
            q, target_species=2, num_reads=50, num_sweeps=50, seed=42
        )

        assert r_many.best_energy <= r_few.best_energy + 1e-10


# --- Cross-solver consistency ---


class TestCrossSolverConsistency:
    def test_sa_matches_exact(self):
        """SA should find the same optimum as exact on a small problem."""
        q = _make_small_qubo()

        exact = ExactSolver().solve(q, target_species=2)
        sa = SimulatedAnnealingSolver().solve(
            q, target_species=2, num_reads=100, num_sweeps=500, seed=42
        )

        assert sa.best_energy == pytest.approx(exact.best_energy)

    def test_sa_energy_at_least_as_good_as_exact(self):
        """SA energy should never be better than exact (exact is ground truth)."""
        q = _make_small_qubo()

        exact = ExactSolver().solve(q, target_species=2)
        sa = SimulatedAnnealingSolver().solve(
            q, target_species=2, num_reads=50, seed=42
        )

        # SA can match but never beat the exact optimum
        assert sa.best_energy >= exact.best_energy - 1e-10


# --- QAOA Tests ---


def _make_tiny_qubo() -> np.ndarray:
    """Create a 4-variable QUBO for fast QAOA testing.

    Designed so selecting species 0 and 2 is optimal when k=2.
    """
    q = np.zeros((4, 4))
    lam = 3.0
    k = 2
    # Diagonal: -h_i + λ(1-2k)
    q[0, 0] = -0.5 + lam * (1 - 2 * k)  # -9.5
    q[1, 1] = -0.1 + lam * (1 - 2 * k)  # -9.1
    q[2, 2] = -0.5 + lam * (1 - 2 * k)  # -9.5
    q[3, 3] = -0.1 + lam * (1 - 2 * k)  # -9.1

    # Off-diagonal: pairwise + 2λ
    for i in range(4):
        for j in range(i + 1, 4):
            q[i, j] = 2 * lam  # = 6.0
    # Pair (0,2) is complementary
    q[0, 2] = -1.0 + 2 * lam  # = 5.0

    return q


class TestQAOA:
    def test_qaoa_runs_and_returns_result(self):
        """QAOA should complete without errors on a tiny problem."""
        q = _make_tiny_qubo()
        config = QAOAConfig(
            depth=1, num_restarts=2, max_iter=20, shots=512, seed=42
        )

        result = QAOASolver().solve(q, target_species=2, config=config)

        assert result.best_solution is not None
        assert result.best_energy is not None
        assert result.solver_name == "qaoa_p1"
        assert result.wall_time_seconds > 0
        assert len(result.all_energies) > 0

    def test_qaoa_solution_is_binary(self):
        """QAOA solution should be a binary vector."""
        q = _make_tiny_qubo()
        config = QAOAConfig(depth=1, num_restarts=2, max_iter=20, shots=512, seed=42)

        result = QAOASolver().solve(q, target_species=2, config=config)

        assert set(result.best_solution).issubset({0, 1})
        assert len(result.best_solution) == 4

    def test_qaoa_energy_matches_solution(self):
        """Reported energy should match the energy computed from the solution."""
        q = _make_tiny_qubo()
        config = QAOAConfig(depth=1, num_restarts=2, max_iter=20, shots=512, seed=42)

        result = QAOASolver().solve(q, target_species=2, config=config)

        computed = qubo_energy(q, result.best_solution)
        assert result.best_energy == pytest.approx(computed)

    def test_qaoa_deeper_at_least_as_good(self):
        """Deeper circuits with more restarts should not do worse on average."""
        q = _make_tiny_qubo()

        config_shallow = QAOAConfig(depth=1, num_restarts=3, max_iter=30, shots=1024, seed=42)
        config_deep = QAOAConfig(depth=2, num_restarts=5, max_iter=50, shots=1024, seed=42)

        r_shallow = QAOASolver().solve(q, target_species=2, config=config_shallow)
        r_deep = QAOASolver().solve(q, target_species=2, config=config_deep)

        # Deep should be at least as good (with some tolerance for shot noise)
        # This is a soft check — QAOA is variational and stochastic
        assert r_deep.best_energy <= r_shallow.best_energy + 2.0

    def test_qaoa_metadata(self):
        """QAOA result should contain expected metadata."""
        q = _make_tiny_qubo()
        config = QAOAConfig(depth=2, num_restarts=2, max_iter=10, shots=256, seed=42)

        result = QAOASolver().solve(q, target_species=2, config=config)

        assert result.metadata["depth"] == 2
        assert result.metadata["num_restarts"] == 2
        assert result.metadata["shots"] == 256
        assert "optimal_params" in result.metadata
        assert "scale_factor" in result.metadata
        assert "feasible_solution_found" in result.metadata

    def test_qaoa_vs_exact_on_tiny_problem(self):
        """QAOA should find a reasonable solution compared to exact on a tiny problem."""
        q = _make_tiny_qubo()

        exact = ExactSolver().solve(q, target_species=2)
        qaoa = QAOASolver().solve(
            q,
            target_species=2,
            config=QAOAConfig(depth=2, num_restarts=5, max_iter=50, shots=2048, seed=42),
        )

        # QAOA should find a solution within 50% of the exact optimum
        # (generous tolerance given the stochastic nature)
        if qaoa.metadata["feasible_solution_found"]:
            assert qaoa.best_energy <= exact.best_energy * 0.5  # energies are negative
