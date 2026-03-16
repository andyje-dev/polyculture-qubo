"""Benchmark runner: solve the real polyculture QUBO with all solvers.

Loads the preprocessed interaction data, builds the QUBO matrix for each
target species count k ∈ {3, 4, 5}, and runs all solvers. Prints a
comparison table and detailed solution breakdowns.

Usage:
    python -m polyculture_qubo.solvers.benchmark
    python -m polyculture_qubo.solvers.benchmark --k 4
    python -m polyculture_qubo.solvers.benchmark --skip-qaoa
"""

import argparse
import sys

import numpy as np
import pandas as pd

from polyculture_qubo.data import PROCESSED_DIR
from polyculture_qubo.matrix.qubo import (
    QUBOConfig,
    build_qubo_matrix,
    evaluate_solution,
    print_solution,
)
from polyculture_qubo.solvers.annealing import SimulatedAnnealingSolver
from polyculture_qubo.solvers.exact import ExactSolver
from polyculture_qubo.solvers.qaoa import QAOAConfig, QAOASolver
from polyculture_qubo.solvers.result import SolverResult


def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed matrices from disk."""
    j = pd.read_csv(PROCESSED_DIR / "interaction_matrix.csv", index_col=0)
    d = pd.read_csv(PROCESSED_DIR / "diversity_matrix.csv", index_col=0)
    b = pd.read_csv(PROCESSED_DIR / "linear_biases.csv", index_col=0)
    c = pd.read_csv(PROCESSED_DIR / "confidence_matrix.csv", index_col=0)
    return j, d, b, c


def run_benchmark(
    k_values: list[int] | None = None,
    skip_qaoa: bool = False,
    qaoa_depths: list[int] | None = None,
) -> dict[int, dict[str, SolverResult]]:
    """Run all solvers on the real QUBO for each target species count.

    Args:
        k_values: Target species counts to sweep. Default: [3, 4, 5].
        skip_qaoa: Skip QAOA solver (much slower than others).
        qaoa_depths: QAOA circuit depths to test. Default: [1, 2, 3].

    Returns:
        Nested dict: results[k][solver_name] = SolverResult
    """
    if k_values is None:
        k_values = [3, 4, 5]
    if qaoa_depths is None:
        qaoa_depths = [1, 2, 3]

    j_matrix, d_matrix, b_matrix, c_matrix = load_matrices()
    results: dict[int, dict[str, SolverResult]] = {}

    for k in k_values:
        print(f"\n{'='*70}")
        print(f"  TARGET SPECIES COUNT: k = {k}")
        print(f"{'='*70}")

        config = QUBOConfig(target_species=k)
        q, species_keys = build_qubo_matrix(j_matrix, d_matrix, b_matrix, config, c_matrix)
        n = len(species_keys)
        results[k] = {}

        # --- Exact solver ---
        print(f"\n--- Exact solver (enumerating all C({n},{k}) solutions) ---")
        exact = ExactSolver().solve(q, target_species=k)
        results[k]["exact"] = exact
        print(f"  Time: {exact.wall_time_seconds:.3f}s")
        print(f"  Solutions evaluated: {exact.num_solutions_evaluated}")
        print(f"  Best energy: {exact.best_energy:.4f}")
        exact_eval = evaluate_solution(
            exact.best_solution, species_keys, q, j_matrix, d_matrix, config, c_matrix
        )
        print_solution(exact_eval)

        # Energy landscape stats
        if len(exact.all_energies) > 0:
            energies = exact.all_energies
            gap = np.sort(energies)[1] - np.sort(energies)[0] if len(energies) > 1 else 0
            near_optimal = np.sum(energies <= exact.best_energy * 0.95) if exact.best_energy < 0 else 0
            print(f"\n  Landscape: min={energies.min():.4f}, max={energies.max():.4f}, "
                  f"mean={energies.mean():.4f}, std={energies.std():.4f}")
            print(f"  Spectral gap (E₁ - E₀): {gap:.6f}")
            print(f"  Near-optimal solutions (within 5%): {near_optimal}/{len(energies)}")

        # --- Simulated annealing ---
        print(f"\n--- Simulated annealing (100 reads × 1000 sweeps) ---")
        sa = SimulatedAnnealingSolver().solve(
            q, target_species=k, num_reads=100, num_sweeps=1000, seed=42
        )
        results[k]["simulated_annealing"] = sa
        print(f"  Time: {sa.wall_time_seconds:.3f}s")
        print(f"  Solutions evaluated: {sa.num_solutions_evaluated}")
        print(f"  Best energy: {sa.best_energy:.4f}")
        sa_eval = evaluate_solution(
            sa.best_solution, species_keys, q, j_matrix, d_matrix, config, c_matrix
        )
        print_solution(sa_eval)

        # Approximation ratio
        if exact.best_energy != 0:
            approx_ratio = sa.best_energy / exact.best_energy
            print(f"  Approximation ratio (SA/exact): {approx_ratio:.6f}")

        # --- QAOA ---
        if not skip_qaoa:
            for p in qaoa_depths:
                print(f"\n--- QAOA (depth p={p}, 5 restarts × 200 iter, 4096 shots) ---")
                qaoa_config = QAOAConfig(
                    depth=p, num_restarts=5, max_iter=200, shots=4096, seed=42
                )
                qaoa = QAOASolver().solve(q, target_species=k, config=qaoa_config)
                results[k][f"qaoa_p{p}"] = qaoa
                print(f"  Time: {qaoa.wall_time_seconds:.3f}s")
                print(f"  Solutions evaluated: {qaoa.num_solutions_evaluated}")
                print(f"  Best energy: {qaoa.best_energy:.4f}")
                print(f"  Feasible solution found: {qaoa.metadata['feasible_solution_found']}")
                print(f"  Unique samples: {qaoa.metadata['num_unique_samples']}")
                qaoa_eval = evaluate_solution(
                    qaoa.best_solution, species_keys, q, j_matrix, d_matrix, config, c_matrix
                )
                print_solution(qaoa_eval)

                if exact.best_energy != 0:
                    approx_ratio = qaoa.best_energy / exact.best_energy
                    print(f"  Approximation ratio (QAOA/exact): {approx_ratio:.6f}")

        # --- Summary table ---
        print(f"\n--- Summary for k={k} ---")
        print(f"{'Solver':<25} {'Energy':>12} {'Time (s)':>10} {'Approx Ratio':>14}")
        print("-" * 65)
        for name, r in results[k].items():
            ratio = r.best_energy / exact.best_energy if exact.best_energy != 0 else float("nan")
            print(f"{name:<25} {r.best_energy:>12.4f} {r.wall_time_seconds:>10.3f} {ratio:>14.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark QUBO solvers")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 4, 5],
                        help="Target species counts to sweep (default: 3 4 5)")
    parser.add_argument("--skip-qaoa", action="store_true",
                        help="Skip QAOA solver (much slower)")
    parser.add_argument("--qaoa-depths", type=int, nargs="+", default=[1, 2, 3],
                        help="QAOA circuit depths to test (default: 1 2 3)")
    args = parser.parse_args()

    run_benchmark(k_values=args.k, skip_qaoa=args.skip_qaoa, qaoa_depths=args.qaoa_depths)


if __name__ == "__main__":
    main()
