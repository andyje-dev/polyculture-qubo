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

import numpy as np
import pandas as pd

from polyculture_qubo.data import PROCESSED_DIR
from polyculture_qubo.matrix.qubo import (
    QUBOConfig,
    build_qubo_matrix,
    compute_penalty_strength,
    evaluate_solution,
    print_solution,
)
from polyculture_qubo.solvers.annealing import SimulatedAnnealingSolver
from polyculture_qubo.solvers.exact import ExactSolver
from polyculture_qubo.solvers.metrics import (
    evaluate_quality,
    format_quality,
    random_baseline,
)
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
        print(f"\n{'=' * 70}")
        print(f"  TARGET SPECIES COUNT: k = {k}")
        print(f"{'=' * 70}")

        config = QUBOConfig(target_species=k)
        q, species_keys = build_qubo_matrix(
            j_matrix, d_matrix, b_matrix, config, c_matrix
        )
        n = len(species_keys)
        results[k] = {}

        # Constant penalty energy shared by every feasible solution. Needed to
        # strip the offset out of energy-ratio metrics.
        lam = compute_penalty_strength(
            j_matrix, d_matrix, b_matrix, config, species_keys
        )
        penalty_offset = -(k**2) * lam

        # One random-feasible baseline per instance, shared by all solvers so
        # their beta values are directly comparable.
        baseline = random_baseline(q, target_species=k, seed=42)

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
            gap = (
                np.sort(energies)[1] - np.sort(energies)[0] if len(energies) > 1 else 0
            )
            # "Within 5%" must be measured on the objective, not on total energy.
            # Total energy is dominated by the constant penalty offset, so the
            # total-energy version counts 100% of feasible solutions at every k
            # regardless of how differentiated the landscape actually is.
            obj = energies - penalty_offset
            obj_best = exact.best_energy - penalty_offset
            near_optimal = int(np.sum(obj <= obj_best * 0.95))
            near_optimal_total = int(np.sum(energies <= exact.best_energy * 0.95))
            print(
                f"\n  Landscape: min={energies.min():.4f}, max={energies.max():.4f}, "
                f"mean={energies.mean():.4f}, std={energies.std():.4f}"
            )
            print(
                f"  Penalty offset (-k²λ): {penalty_offset:.4f} "
                f"({100 * abs(penalty_offset / exact.best_energy):.1f}% of optimum)"
            )
            print(f"  Spectral gap (E₁ - E₀): {gap:.6f}")
            print(
                f"  Near-optimal (within 5% of objective): "
                f"{near_optimal}/{len(energies)} "
                f"[total-energy version would say {near_optimal_total}]"
            )
            print(
                f"  Random feasible baseline: {baseline.mean_energy:.4f} "
                f"(std {baseline.std_energy:.4f}, "
                f"{'exhaustive' if baseline.exhaustive else f'{baseline.n_samples} samples'})"
            )

        # --- Simulated annealing ---
        print("\n--- Simulated annealing (100 reads × 1000 sweeps) ---")
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

        sa_quality = evaluate_quality(
            q,
            target_species=k,
            energy=sa.best_energy,
            reference_energy=exact.best_energy,
            baseline=baseline,
            all_feasible_energies=exact.all_energies,
            penalty_offset=penalty_offset,
        )
        print("  " + format_quality(sa_quality, "Quality"))

        # --- QAOA ---
        if not skip_qaoa:
            for p in qaoa_depths:
                print(
                    f"\n--- QAOA (depth p={p}, 5 restarts × 200 iter, 4096 shots) ---"
                )
                qaoa_config = QAOAConfig(
                    depth=p, num_restarts=5, max_iter=200, shots=4096, seed=42
                )
                qaoa = QAOASolver().solve(q, target_species=k, config=qaoa_config)
                results[k][f"qaoa_p{p}"] = qaoa
                print(f"  Time: {qaoa.wall_time_seconds:.3f}s")
                print(f"  Solutions evaluated: {qaoa.num_solutions_evaluated}")
                print(f"  Best energy: {qaoa.best_energy:.4f}")
                print(
                    f"  Feasible solution found: {qaoa.metadata['feasible_solution_found']}"
                )
                print(f"  Unique samples: {qaoa.metadata['num_unique_samples']}")
                qaoa_eval = evaluate_solution(
                    qaoa.best_solution,
                    species_keys,
                    q,
                    j_matrix,
                    d_matrix,
                    config,
                    c_matrix,
                )
                print_solution(qaoa_eval)

                qaoa_quality = evaluate_quality(
                    q,
                    target_species=k,
                    energy=qaoa.best_energy,
                    reference_energy=exact.best_energy,
                    baseline=baseline,
                    all_feasible_energies=exact.all_energies,
                    penalty_offset=penalty_offset,
                )
                results[k][f"qaoa_p{p}"].metadata["quality"] = qaoa_quality
                print("  " + format_quality(qaoa_quality, "Quality"))
                # In-constraint probability: the share of shots landing in the
                # feasible subspace. Offset-invariant, and the one figure that
                # distinguishes QAOA from a uniform sampler.
                p_in = qaoa.metadata["in_constraint_probability"]
                p_uni = qaoa.metadata["uniform_in_constraint_probability"]
                enrich = p_in / p_uni if p_uni > 0 else float("nan")
                print(
                    f"  In-constraint probability: {p_in:.4%} "
                    f"(uniform {p_uni:.4%}, enrichment {enrich:.1f}×)"
                )

        # --- Summary table ---
        # beta is the primary quality metric (1 = optimal, 0 = no better than a
        # random feasible draw, < 0 = worse). Rank is reported alongside it
        # because it needs no baseline and cannot be misread. The raw
        # total-energy ratio is shown last, with its floor, only so the reader
        # can see how little of its range is usable.
        print(f"\n--- Summary for k={k} ---")
        print(
            f"{'Solver':<22} {'Energy':>12} {'Time (s)':>10} "
            f"{'beta':>9} {'Rank':>12} {'ObjRatio':>10} {'TotRatio':>10}"
        )
        print("-" * 90)
        for name, r in results[k].items():
            qual = evaluate_quality(
                q,
                target_species=k,
                energy=r.best_energy,
                reference_energy=exact.best_energy,
                baseline=baseline,
                all_feasible_energies=exact.all_energies,
                penalty_offset=penalty_offset,
            )
            rank_str = f"{qual.rank}/{qual.n_feasible}"
            print(
                f"{name:<22} {r.best_energy:>12.4f} {r.wall_time_seconds:>10.3f} "
                f"{qual.beta:>+9.4f} {rank_str:>12} "
                f"{qual.objective_ratio:>10.4f} {qual.total_energy_ratio:>10.4f}"
            )
        # Reference rows: what each metric assigns to a random draw and to the
        # worst possible feasible answer. The distance between these and the
        # solver rows is the metric's real discriminating power.
        n_feas = len(exact.all_energies)
        for label, energy in (
            ("(random feasible)", baseline.mean_energy),
            ("(worst feasible)", float(exact.all_energies.max())),
        ):
            ref_qual = evaluate_quality(
                q,
                target_species=k,
                energy=energy,
                reference_energy=exact.best_energy,
                baseline=baseline,
                all_feasible_energies=exact.all_energies,
                penalty_offset=penalty_offset,
            )
            print(
                f"{label:<22} {energy:>12.4f} {'—':>10} "
                f"{ref_qual.beta:>+9.4f} {f'{ref_qual.rank}/{n_feas}':>12} "
                f"{ref_qual.objective_ratio:>10.4f} "
                f"{ref_qual.total_energy_ratio:>10.4f}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark QUBO solvers")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Target species counts to sweep (default: 3 4 5)",
    )
    parser.add_argument(
        "--skip-qaoa", action="store_true", help="Skip QAOA solver (much slower)"
    )
    parser.add_argument(
        "--qaoa-depths",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="QAOA circuit depths to test (default: 1 2 3)",
    )
    args = parser.parse_args()

    run_benchmark(
        k_values=args.k, skip_qaoa=args.skip_qaoa, qaoa_depths=args.qaoa_depths
    )


if __name__ == "__main__":
    main()
