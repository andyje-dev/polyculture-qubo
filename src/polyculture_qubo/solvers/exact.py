"""Brute-force exact solver for the polyculture QUBO problem.

Enumerates all (N choose k) feasible solutions (those selecting exactly k species)
and evaluates each one, establishing ground truth for benchmarking heuristic and
quantum solvers.

For the default problem size (N=20, k=4), this is 4,845 solutions — trivial
to enumerate in under a second.
"""

import time
from itertools import combinations

import numpy as np

from polyculture_qubo.solvers.result import SolverResult


class ExactSolver:
    """Brute-force enumeration of all feasible QUBO solutions.

    Evaluates every binary vector with exactly k ones (the species count
    constraint) and returns the one with minimum energy.

    Also collects the full energy landscape for analysis: degeneracy,
    gap structure, and local minima density.
    """

    def solve(
        self,
        q: np.ndarray,
        target_species: int,
        collect_all: bool = True,
    ) -> SolverResult:
        """Enumerate all (N choose k) feasible solutions and find the optimum.

        Args:
            q: Upper-triangular QUBO matrix (N x N).
            target_species: Number of species to select (k).
            collect_all: If True, store all solutions and energies for
                landscape analysis. For N=20 this is cheap; disable for
                very large N.

        Returns:
            SolverResult with the global optimum and full energy landscape.
        """
        n = q.shape[0]
        k = target_species

        best_energy = float("inf")
        best_solution = None
        all_energies = []
        all_solutions = []

        start = time.perf_counter()

        for combo in combinations(range(n), k):
            x = np.zeros(n, dtype=int)
            x[list(combo)] = 1

            energy = _qubo_energy_fast(q, combo)

            if collect_all:
                all_energies.append(energy)
                all_solutions.append(x)

            if energy < best_energy:
                best_energy = energy
                best_solution = x.copy()

        elapsed = time.perf_counter() - start
        num_evaluated = len(all_energies) if collect_all else _n_choose_k(n, k)

        if best_solution is None:
            raise ValueError(
                f"No feasible solution exists: cannot select k={k} from n={n} species"
            )

        return SolverResult(
            best_solution=best_solution,
            best_energy=best_energy,
            solver_name="exact",
            num_solutions_evaluated=num_evaluated,
            wall_time_seconds=elapsed,
            all_energies=np.array(all_energies) if collect_all else np.array([]),
            all_solutions=all_solutions if collect_all else [],
            metadata={"n": n, "k": k},
        )


def _qubo_energy_fast(q: np.ndarray, selected_indices: tuple[int, ...]) -> float:
    """Compute QUBO energy using only the selected indices.

    Faster than constructing a full binary vector when k << N.
    """
    energy = 0.0
    for i in selected_indices:
        energy += q[i, i]
    for idx_a in range(len(selected_indices)):
        for idx_b in range(idx_a + 1, len(selected_indices)):
            i, j = selected_indices[idx_a], selected_indices[idx_b]
            if i < j:
                energy += q[i, j]
            else:
                energy += q[j, i]
    return energy


def _n_choose_k(n: int, k: int) -> int:
    """Compute binomial coefficient."""
    from math import comb

    return comb(n, k)
