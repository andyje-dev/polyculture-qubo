"""Tests for QUBO matrix construction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polyculture_qubo.matrix import (
    QUBOConfig,
    build_nitrogen_matrix,
    build_qubo_matrix,
    evaluate_solution,
    normalize_qubo,
    qubo_energy,
    qubo_to_ising,
    solution_to_species,
)
from polyculture_qubo.species import CANDIDATE_SPECIES

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SKIP_NO_DATA = pytest.mark.skipif(
    not (PROCESSED_DIR / "interaction_matrix.csv").exists(),
    reason="Processed data not available",
)


def _load_matrices():
    j = pd.read_csv(PROCESSED_DIR / "interaction_matrix.csv", index_col=0)
    d = pd.read_csv(PROCESSED_DIR / "diversity_matrix.csv", index_col=0)
    b = pd.read_csv(PROCESSED_DIR / "linear_biases.csv", index_col=0)
    c = pd.read_csv(PROCESSED_DIR / "confidence_matrix.csv", index_col=0)
    return j, d, b, c


class TestNitrogenMatrix:
    def test_fixer_nonfixer_highest(self):
        keys = sorted(CANDIDATE_SPECIES.keys())
        n_mat = build_nitrogen_matrix(keys)
        # Cowpea (fixer) + Maize (non-fixer) should score 1.0
        i, j = (
            min(keys.index("cowpea"), keys.index("maize")),
            max(keys.index("cowpea"), keys.index("maize")),
        )
        assert n_mat[i, j] == 1.0

    def test_fixer_fixer_low(self):
        keys = sorted(CANDIDATE_SPECIES.keys())
        n_mat = build_nitrogen_matrix(keys)
        # Cowpea (fixer) + Pea (fixer) should score 0.2
        i, j = (
            min(keys.index("cowpea"), keys.index("pea")),
            max(keys.index("cowpea"), keys.index("pea")),
        )
        assert n_mat[i, j] == 0.2

    def test_nonfixer_nonfixer_zero(self):
        keys = sorted(CANDIDATE_SPECIES.keys())
        n_mat = build_nitrogen_matrix(keys)
        # Maize (non-fixer) + Wheat (non-fixer) should score 0.0
        i, j = (
            min(keys.index("maize"), keys.index("wheat")),
            max(keys.index("maize"), keys.index("wheat")),
        )
        assert n_mat[i, j] == 0.0

    def test_upper_triangular(self):
        keys = sorted(CANDIDATE_SPECIES.keys())
        n_mat = build_nitrogen_matrix(keys)
        # Lower triangle should be all zeros
        assert np.allclose(n_mat, np.triu(n_mat))


class TestQUBOEnergy:
    def test_empty_solution(self):
        q = np.array([[1.0, 2.0], [0.0, 3.0]])
        x = np.array([0, 0])
        assert qubo_energy(q, x) == 0.0

    def test_single_variable(self):
        q = np.array([[5.0, 2.0], [0.0, 3.0]])
        x = np.array([1, 0])
        assert qubo_energy(q, x) == 5.0

    def test_two_variables(self):
        q = np.array([[5.0, 2.0], [0.0, 3.0]])
        x = np.array([1, 1])
        # E = Q_00 + Q_11 + Q_01 = 5 + 3 + 2 = 10
        assert qubo_energy(q, x) == 10.0

    def test_upper_triangular_convention(self):
        # Only upper triangle matters
        q = np.array([[1.0, 4.0], [99.0, 2.0]])
        x = np.array([1, 1])
        # E = 1 + 2 + 4 = 7 (lower triangle ignored)
        assert qubo_energy(q, x) == 7.0


class TestSolutionToSpecies:
    def test_basic(self):
        keys = ["a", "b", "c"]
        x = np.array([1, 0, 1])
        assert solution_to_species(x, keys) == ["a", "c"]

    def test_empty(self):
        keys = ["a", "b"]
        x = np.array([0, 0])
        assert solution_to_species(x, keys) == []


@SKIP_NO_DATA
class TestQUBOConstruction:
    def test_upper_triangular(self):
        j, d, b, c = _load_matrices()
        Q, _ = build_qubo_matrix(j, d, b, confidence_matrix=c)
        assert np.allclose(Q, np.triu(Q))

    def test_shape(self):
        j, d, b, c = _load_matrices()
        Q, keys = build_qubo_matrix(j, d, b, confidence_matrix=c)
        n = len(CANDIDATE_SPECIES)
        assert Q.shape == (n, n)
        assert len(keys) == n

    def test_auto_penalty_enforces_constraint(self):
        """Auto-derived λ should enforce the count constraint."""
        j, d, b, c = _load_matrices()
        config = QUBOConfig(target_species=4)  # penalty_strength=None → auto
        Q, keys = build_qubo_matrix(j, d, b, config, confidence_matrix=c)

        def energy(selected):
            x = np.zeros(len(keys))
            for s in selected:
                x[keys.index(s)] = 1
            return qubo_energy(Q, x)

        e4 = energy(["maize", "common_bean", "cowpea", "pea"])
        e2 = energy(["maize", "common_bean"])
        e5 = energy(["maize", "common_bean", "cowpea", "pea", "lettuce"])
        e0 = energy([])
        e3 = energy(["maize", "common_bean", "cowpea"])

        assert e4 < e0
        assert e4 < e2
        assert e4 < e3
        assert e4 < e5

    def test_complementary_pairs_preferred(self):
        """Solutions with high-LER pairs should have lower energy."""
        j, d, b, c = _load_matrices()
        config = QUBOConfig(target_species=4)
        Q, keys = build_qubo_matrix(j, d, b, config, confidence_matrix=c)

        def energy(selected):
            x = np.zeros(len(keys))
            for s in selected:
                x[keys.index(s)] = 1
            return qubo_energy(Q, x)

        # Maize+bean+cowpea+pea (high LER pairs) should beat all-cereals
        e_good = energy(["maize", "common_bean", "cowpea", "pea"])
        e_bad = energy(["maize", "wheat", "barley", "oat"])
        assert e_good < e_bad

    def test_mixed_fixers_beat_all_fixers(self):
        """With pairwise N-balance, mixed fixer/non-fixer should beat all-fixers."""
        j, d, b, c = _load_matrices()
        # Boost beta to make N-balance dominant — don't pass confidence here
        # so N-balance is unscaled (testing the N-balance term itself)
        config = QUBOConfig(alpha=0.1, beta=0.8, gamma=0.1, target_species=4)
        Q, keys = build_qubo_matrix(j, d, b, config)

        def energy(selected):
            x = np.zeros(len(keys))
            for s in selected:
                x[keys.index(s)] = 1
            return qubo_energy(Q, x)

        # 3 fixers + 1 non-fixer (3 complementary N pairs) should beat 4 fixers (0 complementary)
        e_mixed = energy(["maize", "common_bean", "cowpea", "pea"])
        e_all_fixers = energy(["common_bean", "cowpea", "pea", "fava_bean"])
        assert e_mixed < e_all_fixers

    def test_n_balance_scaled_by_confidence(self):
        """N-balance contribution should be reduced for high-confidence LER pairs."""
        j, d, b, c = _load_matrices()
        config = QUBOConfig(target_species=4, penalty_strength=1.0)

        # Without confidence scaling: full N-balance
        Q_no_conf, keys = build_qubo_matrix(j, d, b, config)
        # With confidence scaling: reduced N-balance for observed pairs
        Q_with_conf, _ = build_qubo_matrix(j, d, b, config, confidence_matrix=c)

        # For a high-confidence pair like cowpea+maize, the N-balance
        # contribution should be smaller with confidence scaling
        i_cowpea = keys.index("cowpea")
        i_maize = keys.index("maize")
        ii, jj = min(i_cowpea, i_maize), max(i_cowpea, i_maize)

        # Q_with_conf should be less negative (higher) because N-balance
        # is reduced for this well-observed pair
        assert Q_with_conf[ii, jj] > Q_no_conf[ii, jj]

    def test_evaluate_solution_fields(self):
        j, d, b, c = _load_matrices()
        config = QUBOConfig(target_species=4)
        Q, keys = build_qubo_matrix(j, d, b, config, confidence_matrix=c)

        x = np.zeros(len(keys))
        for s in ["maize", "common_bean", "cowpea", "pea"]:
            x[keys.index(s)] = 1

        result = evaluate_solution(x, keys, Q, j, d, config)
        assert result["n_selected"] == 4
        assert result["constraint_satisfied"] is True
        assert result["n_fixers"] == 3  # bean, cowpea, pea
        assert result["n_non_fixers"] == 1  # maize
        assert result["has_cash_crop"] is True
        assert result["ler_score"] > 0  # These pairs are complementary
        assert result["n_balance_score"] > 0  # Has fixer+non-fixer pairs

    def test_custom_species_subset(self):
        """Building QUBO with a subset of species should work."""
        j, d, b, c = _load_matrices()
        subset = ["maize", "wheat", "pea", "cowpea", "cabbage"]
        config = QUBOConfig(target_species=3, species_keys=subset)
        Q, keys = build_qubo_matrix(j, d, b, config, confidence_matrix=c)
        assert Q.shape == (5, 5)
        assert keys == subset


class TestNormalizeQUBO:
    def test_entries_in_range(self):
        q = np.array([[5.0, -3.0], [0.0, 2.0]])
        q_norm, scale = normalize_qubo(q)
        assert scale == 5.0
        assert np.max(np.abs(q_norm)) == pytest.approx(1.0)

    def test_preserves_relative_structure(self):
        q = np.array([[10.0, -4.0], [0.0, 6.0]])
        q_norm, scale = normalize_qubo(q)
        # Ratios between entries should be preserved
        assert q_norm[0, 0] / q_norm[1, 1] == pytest.approx(q[0, 0] / q[1, 1])

    def test_energy_roundtrip(self):
        """Normalized energy * scale == original energy."""
        q = np.array([[5.0, -3.0, 1.0], [0.0, 2.0, -4.0], [0.0, 0.0, 7.0]])
        q_norm, scale = normalize_qubo(q)
        x = np.array([1, 0, 1])
        e_orig = qubo_energy(q, x)
        e_norm = qubo_energy(q_norm, x)
        assert e_orig == pytest.approx(scale * e_norm)

    def test_zero_matrix(self):
        q = np.zeros((3, 3))
        q_norm, scale = normalize_qubo(q)
        assert scale == 1.0
        np.testing.assert_array_equal(q_norm, q)

    @SKIP_NO_DATA
    def test_real_qubo_normalized_range(self):
        """Full-size QUBO should normalize to [-1, 1]."""
        j, d, b, c = _load_matrices()
        Q, _ = build_qubo_matrix(j, d, b, confidence_matrix=c)
        Q_norm, scale = normalize_qubo(Q)
        assert np.max(np.abs(Q_norm)) == pytest.approx(1.0)
        assert scale > 0


class TestQUBOToIsing:
    def _ising_energy(self, h, j_ising, offset, s):
        """Compute Ising energy: offset + Σ h_i s_i + Σ_{i<j} J_ij s_i s_j."""
        e = offset + np.dot(h, s)
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                e += j_ising[i, j] * s[i] * s[j]
        return e

    def test_energy_equivalence_2var(self):
        """QUBO and Ising energies must match for all 2-variable assignments."""
        q = np.array([[5.0, -3.0], [0.0, 2.0]])
        h, j_ising, offset = qubo_to_ising(q)

        for x0 in [0, 1]:
            for x1 in [0, 1]:
                x = np.array([x0, x1])
                s = 1 - 2 * x  # x=0 → s=+1, x=1 → s=-1
                e_qubo = qubo_energy(q, x)
                e_ising = self._ising_energy(h, j_ising, offset, s)
                assert e_qubo == pytest.approx(e_ising), (
                    f"Mismatch for x={x}: QUBO={e_qubo}, Ising={e_ising}"
                )

    def test_energy_equivalence_3var(self):
        """QUBO and Ising energies must match for all 3-variable assignments."""
        q = np.array([[2.0, -1.0, 3.0], [0.0, 4.0, -2.0], [0.0, 0.0, 1.0]])
        h, j_ising, offset = qubo_to_ising(q)

        for bits in range(8):
            x = np.array([(bits >> i) & 1 for i in range(3)])
            s = 1 - 2 * x
            e_qubo = qubo_energy(q, x)
            e_ising = self._ising_energy(h, j_ising, offset, s)
            assert e_qubo == pytest.approx(e_ising), (
                f"Mismatch for x={x}: QUBO={e_qubo}, Ising={e_ising}"
            )

    def test_j_upper_triangular(self):
        q = np.array([[1.0, 2.0], [0.0, 3.0]])
        _, j_ising, _ = qubo_to_ising(q)
        assert np.allclose(j_ising, np.triu(j_ising))

    @SKIP_NO_DATA
    def test_real_qubo_ising_roundtrip(self):
        """Full-size QUBO → Ising conversion preserves energies."""
        j, d, b, c = _load_matrices()
        Q, keys = build_qubo_matrix(j, d, b, confidence_matrix=c)
        h, j_ising, offset = qubo_to_ising(Q)

        # Test a few specific solutions
        for species_set in [
            ["maize", "common_bean", "cowpea", "pea"],
            ["wheat", "barley", "oat", "sorghum"],
            ["lettuce", "tomato", "cabbage", "onion"],
        ]:
            x = np.zeros(len(keys))
            for sp in species_set:
                x[keys.index(sp)] = 1
            s = 1 - 2 * x
            e_qubo = qubo_energy(Q, x)
            e_ising = self._ising_energy(h, j_ising, offset, s)
            assert e_qubo == pytest.approx(e_ising, abs=1e-10)
