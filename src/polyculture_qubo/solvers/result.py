"""Solver result type shared across all solver implementations."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SolverResult:
    """Result from a QUBO solver.

    Attributes:
        best_solution: Binary vector of the best solution found.
        best_energy: Energy (objective value) of the best solution.
        solver_name: Name of the solver that produced this result.
        num_solutions_evaluated: Number of candidate solutions evaluated.
        wall_time_seconds: Wall-clock time for the solve.
        all_energies: All energies evaluated (for landscape analysis).
            For exact solver, this is every feasible solution's energy.
            For heuristic solvers, this is the energy at each iteration.
        all_solutions: Corresponding solution vectors for all_energies.
            Only populated when requested (can be large for exact solver).
        metadata: Solver-specific metadata (e.g., annealing schedule params).
    """

    best_solution: np.ndarray
    best_energy: float
    solver_name: str
    num_solutions_evaluated: int
    wall_time_seconds: float
    all_energies: np.ndarray = field(default_factory=lambda: np.array([]))
    all_solutions: list[np.ndarray] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
