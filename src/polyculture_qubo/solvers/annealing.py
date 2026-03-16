"""Simulated annealing solver for the polyculture QUBO problem.

Implements a classical heuristic baseline that validates the QUBO formulation
produces sensible results before moving to quantum solvers. Uses single-spin
flip dynamics with a constraint-aware move set: only proposes swaps that
maintain exactly k selected species (deselect one, select another).

This constraint-preserving approach avoids wasting moves on infeasible
solutions and eliminates sensitivity to the penalty strength λ.
"""

import time

import numpy as np

from polyculture_qubo.matrix.qubo import qubo_energy
from polyculture_qubo.solvers.result import SolverResult


class SimulatedAnnealingSolver:
    """Simulated annealing with constraint-preserving swap moves.

    Each move swaps one selected species for one unselected species,
    maintaining exactly k species throughout the anneal. This is equivalent
    to exploring the feasible subspace directly.
    """

    def solve(
        self,
        q: np.ndarray,
        target_species: int,
        num_reads: int = 100,
        num_sweeps: int = 1000,
        beta_start: float = 0.1,
        beta_end: float = 10.0,
        seed: int | None = None,
    ) -> SolverResult:
        """Run simulated annealing with multiple random restarts.

        Args:
            q: Upper-triangular QUBO matrix (N x N).
            target_species: Number of species to select (k).
            num_reads: Number of independent annealing runs (random restarts).
            num_sweeps: Number of sweep iterations per run. Each sweep
                proposes N swap moves (one for each selected species).
            beta_start: Inverse temperature at start (low = hot = exploratory).
            beta_end: Inverse temperature at end (high = cold = greedy).
            seed: Random seed for reproducibility.

        Returns:
            SolverResult with the best solution found across all reads.
        """
        n = q.shape[0]
        k = target_species
        rng = np.random.default_rng(seed)

        best_energy = float("inf")
        best_solution = None
        all_energies = []

        start = time.perf_counter()
        total_evaluated = 0

        # Precompute the annealing schedule (geometric cooling)
        betas = np.geomspace(beta_start, beta_end, num_sweeps)

        for _ in range(num_reads):
            x, energy, evaluated = self._single_run(q, n, k, betas, rng)
            total_evaluated += evaluated
            all_energies.append(energy)

            if energy < best_energy:
                best_energy = energy
                best_solution = x.copy()

        elapsed = time.perf_counter() - start

        return SolverResult(
            best_solution=best_solution,
            best_energy=best_energy,
            solver_name="simulated_annealing",
            num_solutions_evaluated=total_evaluated,
            wall_time_seconds=elapsed,
            all_energies=np.array(all_energies),
            metadata={
                "num_reads": num_reads,
                "num_sweeps": num_sweeps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "seed": seed,
            },
        )

    def _single_run(
        self,
        q: np.ndarray,
        n: int,
        k: int,
        betas: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, int]:
        """Execute one annealing run from a random initial state.

        Returns:
            (best_x, best_energy, num_evaluated)
        """
        # Random initial feasible solution: choose k from N
        selected = set(rng.choice(n, size=k, replace=False).tolist())
        unselected = set(range(n)) - selected

        x = np.zeros(n, dtype=int)
        for i in selected:
            x[i] = 1
        current_energy = qubo_energy(q, x)

        best_x = x.copy()
        best_energy = current_energy
        num_evaluated = 1

        for beta in betas:
            # One sweep: try swapping each selected species with a random unselected one
            selected_list = list(selected)
            rng.shuffle(selected_list)

            for i_out in selected_list:
                # Pick a random species to swap in
                j_in = rng.choice(list(unselected))

                # Compute energy delta for swapping i_out -> j_in
                delta = _swap_delta(q, x, i_out, j_in)
                num_evaluated += 1

                # Metropolis acceptance
                if delta < 0 or rng.random() < np.exp(-beta * delta):
                    # Accept the swap
                    x[i_out] = 0
                    x[j_in] = 1
                    current_energy += delta
                    selected.discard(i_out)
                    selected.add(j_in)
                    unselected.discard(j_in)
                    unselected.add(i_out)

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_x = x.copy()

        return best_x, best_energy, num_evaluated


def _swap_delta(q: np.ndarray, x: np.ndarray, i_out: int, j_in: int) -> float:
    """Compute the energy change from deselecting i_out and selecting j_in.

    ΔE = E(x') - E(x) where x' has x[i_out]=0 and x[j_in]=1.

    This is O(N) instead of O(N²) for a full energy recomputation.
    """
    n = len(x)

    # Contribution of removing i_out (was selected, now deselected)
    delta = -q[i_out, i_out]
    for m in range(n):
        if m == i_out or x[m] == 0:
            continue
        if m == j_in:
            continue  # j_in wasn't selected yet
        i, j = min(i_out, m), max(i_out, m)
        delta -= q[i, j]

    # Contribution of adding j_in (was unselected, now selected)
    delta += q[j_in, j_in]
    for m in range(n):
        if m == j_in or x[m] == 0:
            continue
        if m == i_out:
            continue  # i_out is no longer selected
        i, j = min(j_in, m), max(j_in, m)
        delta += q[i, j]

    # Cross-term: if i_out and j_in are the pair being swapped,
    # there's no coupling between them in the new state (one is in, one is out)
    # and there was no coupling in the old state either (same reason).
    # So no cross-term correction needed.

    return delta
