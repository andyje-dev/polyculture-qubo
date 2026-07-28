"""QAOA (Quantum Approximate Optimization Algorithm) solver for the QUBO problem.

Implements QAOA using Qiskit with Aer statevector simulation. The algorithm
constructs a parameterized quantum circuit with alternating cost and mixer
unitaries, then uses a classical optimizer to find the variational parameters
that minimize the expected energy.

The QUBO is first converted to an Ising Hamiltonian (spins s_i ∈ {-1, +1})
via the standard substitution x_i = (1 - s_i) / 2. The cost unitary
exp(-iγ H_C) is decomposed into ZZ interactions (for couplings J_ij) and
Z rotations (for local fields h_i). The mixer unitary exp(-iβ H_M) uses
the standard transverse-field mixer Σ X_i.

Circuit depth p controls the trade-off between expressiveness and optimization
difficulty. For this 20-qubit problem on a simulator, p=1 to p=5 is tractable.
"""

import logging
import time
from dataclasses import dataclass
from math import comb

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from polyculture_qubo.matrix.qubo import normalize_qubo, qubo_to_ising
from polyculture_qubo.solvers.result import SolverResult

logger = logging.getLogger(__name__)


@dataclass
class QAOAConfig:
    """Configuration for QAOA solver."""

    depth: int = 1  # Circuit depth p (number of QAOA layers)
    num_restarts: int = 5  # Number of random restarts for optimizer
    max_iter: int = 200  # Max iterations per optimizer restart
    shots: int = 4096  # Number of measurement shots per evaluation
    optimizer: str = "COBYLA"  # Classical optimizer (COBYLA or SPSA)
    seed: int | None = None  # Random seed


class QAOASolver:
    """QAOA solver using Qiskit Aer statevector simulator.

    Constructs a QAOA circuit from the Ising Hamiltonian, optimizes
    variational parameters (γ, β) using a classical optimizer, and
    samples the final state to extract solutions.
    """

    def solve(
        self,
        q: np.ndarray,
        target_species: int,
        config: QAOAConfig | None = None,
    ) -> SolverResult:
        """Run QAOA on the given QUBO matrix.

        The QUBO is normalized and converted to Ising form before
        circuit construction. The best feasible solution (exactly k
        species selected) is returned. If no feasible solution is found
        in the samples, the best overall sample is returned with a note
        in metadata.

        Args:
            q: Upper-triangular QUBO matrix (N x N).
            target_species: Number of species to select (k).
            config: QAOA configuration parameters.

        Returns:
            SolverResult with the best solution found.
        """
        if config is None:
            config = QAOAConfig()

        n = q.shape[0]
        rng = np.random.default_rng(config.seed)

        # Normalize and convert to Ising
        q_norm, scale = normalize_qubo(q)
        h_ising, j_ising, offset = qubo_to_ising(q_norm)

        start = time.perf_counter()

        best_params = None
        best_cost = float("inf")
        total_evals = 0

        for restart in range(config.num_restarts):
            # Random initial parameters: γ ∈ [0, 2π], β ∈ [0, π]
            init_gamma = rng.uniform(0, 2 * np.pi, config.depth)
            init_beta = rng.uniform(0, np.pi, config.depth)
            init_params = np.concatenate([init_gamma, init_beta])

            params, cost, n_evals = _optimize_params(
                h_ising, j_ising, offset, n, config, init_params, rng
            )
            total_evals += n_evals

            if cost < best_cost:
                best_cost = cost
                best_params = params

        # Final sampling with best parameters
        assert best_params is not None, "No restarts completed"
        gamma_opt = best_params[: config.depth]
        beta_opt = best_params[config.depth :]
        counts = _sample_qaoa(
            h_ising, j_ising, n, gamma_opt, beta_opt, config.shots, rng
        )

        elapsed = time.perf_counter() - start

        # Extract best solution from measurement results
        best_energy = float("inf")
        best_solution: np.ndarray = np.zeros(n, dtype=int)
        best_feasible_energy = float("inf")
        best_feasible_solution = None
        all_energies = []
        in_constraint_shots = 0
        total_shots = 0

        from polyculture_qubo.matrix.qubo import qubo_energy

        for bitstring, count in counts.items():
            # Qiskit returns bitstrings in reverse order (qubit 0 is rightmost)
            x = np.array([int(b) for b in reversed(bitstring)], dtype=int)
            energy = qubo_energy(q, x)
            all_energies.extend([energy] * count)
            total_shots += count

            if energy < best_energy:
                best_energy = energy
                best_solution = x

            if np.sum(x) == target_species:
                in_constraint_shots += count
                if energy < best_feasible_energy:
                    best_feasible_energy = energy
                    best_feasible_solution = x

        # Prefer feasible solution
        if best_feasible_solution is not None:
            best_solution = best_feasible_solution
            best_energy = best_feasible_energy
            feasible = True
        else:
            feasible = False
            n_selected = int(np.sum(best_solution))
            logger.warning(
                "QAOA found no feasible solution (target=%d species, best has %d). "
                "The standard transverse-field mixer explores the full 2^N space "
                "and relies on penalty strength to enforce constraints. Consider "
                "increasing penalty margin or circuit depth.",
                target_species,
                n_selected,
            )

        return SolverResult(
            best_solution=best_solution,
            best_energy=best_energy,
            solver_name=f"qaoa_p{config.depth}",
            num_solutions_evaluated=total_evals,
            wall_time_seconds=elapsed,
            all_energies=np.array(all_energies),
            metadata={
                "depth": config.depth,
                "num_restarts": config.num_restarts,
                "shots": config.shots,
                "optimal_params": best_params.tolist(),
                "optimal_cost": best_cost,
                "scale_factor": scale,
                "feasible_solution_found": feasible,
                "num_unique_samples": len(counts),
                # In-constraint (Hamming weight == k) sampling statistics.
                # The transverse-field mixer explores all 2^N states, so the
                # share of shots landing in the feasible subspace measures how
                # much of the constraint structure the circuit actually learned.
                # Compare against uniform_in_constraint_probability = C(n,k)/2^n.
                "in_constraint_shots": in_constraint_shots,
                "total_shots": total_shots,
                "in_constraint_probability": (
                    in_constraint_shots / total_shots if total_shots else 0.0
                ),
                "uniform_in_constraint_probability": float(
                    comb(n, target_species) / 2**n
                ),
                "mean_sampled_energy": (
                    float(np.mean(all_energies)) if all_energies else float("nan")
                ),
            },
        )


def _build_qaoa_circuit(
    h: np.ndarray,
    j: np.ndarray,
    n: int,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> QuantumCircuit:
    """Build a QAOA circuit for the given Ising Hamiltonian.

    The circuit alternates p layers of:
      1. Cost unitary: exp(-iγ H_C) decomposed into RZZ and RZ gates
      2. Mixer unitary: exp(-iβ H_M) using RX gates

    Args:
        h: Local fields (N-vector).
        j: Upper-triangular coupling matrix (N x N).
        n: Number of qubits.
        gamma: Cost layer parameters (length p).
        beta: Mixer layer parameters (length p).

    Returns:
        Parameterized QuantumCircuit.
    """
    p = len(gamma)
    qc = QuantumCircuit(n)

    # Initial state: uniform superposition
    qc.h(range(n))

    for layer in range(p):
        g = gamma[layer]
        b = beta[layer]

        # Cost unitary: ZZ interactions
        for i in range(n):
            for jj in range(i + 1, n):
                if abs(j[i, jj]) > 1e-12:
                    qc.rzz(2 * g * j[i, jj], i, jj)

        # Cost unitary: Z rotations (local fields)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * g * h[i], i)

        # Mixer unitary: X rotations
        for i in range(n):
            qc.rx(2 * b, i)

    # Measurement
    qc.measure_all()

    return qc


def _evaluate_qaoa_energy(
    h: np.ndarray,
    j: np.ndarray,
    offset: float,
    n: int,
    params: np.ndarray,
    depth: int,
    shots: int,
    rng: np.random.Generator,
) -> float:
    """Evaluate the expected Ising energy for given QAOA parameters.

    Runs the circuit, measures, and computes the expectation value of H_C.
    """
    gamma = params[:depth]
    beta = params[depth:]

    qc = _build_qaoa_circuit(h, j, n, gamma, beta)

    simulator = AerSimulator(method="automatic")
    seed_sim = int(rng.integers(0, 2**31))
    result = simulator.run(qc, shots=shots, seed_simulator=seed_sim).result()
    counts = result.get_counts()

    # Compute expected Ising energy
    total_energy = 0.0
    for bitstring, count in counts.items():
        # Convert bitstring to Ising spins: '0' -> +1, '1' -> -1
        # Qiskit bitstrings are reversed (qubit 0 is rightmost)
        spins = np.array([1 - 2 * int(b) for b in reversed(bitstring)], dtype=float)

        # Ising energy: H = offset + Σ h_i s_i + Σ_{i<j} J_ij s_i s_j
        e = offset
        e += np.dot(h, spins)
        for i in range(n):
            for jj in range(i + 1, n):
                e += j[i, jj] * spins[i] * spins[jj]

        total_energy += e * count

    return total_energy / shots


def _optimize_params(
    h: np.ndarray,
    j: np.ndarray,
    offset: float,
    n: int,
    config: QAOAConfig,
    init_params: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    """Optimize QAOA variational parameters using a classical optimizer.

    Returns:
        (optimal_params, optimal_cost, num_evaluations)
    """
    from scipy.optimize import minimize

    n_evals = 0

    def objective(params):
        nonlocal n_evals
        n_evals += 1
        return _evaluate_qaoa_energy(
            h, j, offset, n, params, config.depth, config.shots, rng
        )

    result = minimize(
        objective,
        init_params,
        method=config.optimizer,
        options={"maxiter": config.max_iter},
    )

    return result.x, result.fun, n_evals


def _sample_qaoa(
    h: np.ndarray,
    j: np.ndarray,
    n: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Sample from the QAOA circuit with optimized parameters.

    Returns:
        Dictionary mapping bitstrings to measurement counts.
    """
    qc = _build_qaoa_circuit(h, j, n, gamma, beta)
    simulator = AerSimulator(method="automatic")
    seed_sim = int(rng.integers(0, 2**31))
    result = simulator.run(qc, shots=shots, seed_simulator=seed_sim).result()
    return result.get_counts()
