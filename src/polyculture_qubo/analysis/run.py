"""Analysis runner: execute all Phase 4 analyses and produce figures.

Can run standalone or accept benchmark results from Phase 3 to include
QAOA performance data in the scalability analysis.

Usage:
    python -m polyculture_qubo.analysis.run
    python -m polyculture_qubo.analysis.run --skip-sensitivity
    python -m polyculture_qubo.analysis.run --skip-loo
    python -m polyculture_qubo.analysis.run --with-qaoa --qaoa-depths 1 2
"""

import argparse

import pandas as pd

from polyculture_qubo.data import PROCESSED_DIR
from polyculture_qubo.matrix.qubo import (
    QUBOConfig,
    build_qubo_matrix,
    compute_penalty_strength,
    solution_to_species,
)
from polyculture_qubo.solvers.annealing import SimulatedAnnealingSolver
from polyculture_qubo.solvers.exact import ExactSolver
from polyculture_qubo.solvers.metrics import (
    evaluate_quality,
    format_quality,
    random_baseline,
)

from polyculture_qubo.analysis.landscape import (
    analyze_landscape,
    frustration_analysis,
    print_landscape_metrics,
)
from polyculture_qubo.analysis.plots import (
    plot_data_coverage,
    plot_energy_histogram,
    plot_gap_structure,
    plot_interaction_network,
    plot_objective_decomposition,
    plot_scalability,
    plot_sensitivity_heatmap,
    plot_solver_comparison,
)
from polyculture_qubo.analysis.scalability import (
    compute_scalability_metrics,
    print_scalability_summary,
)
from polyculture_qubo.analysis.sensitivity import (
    analyze_weight_sweep,
    data_masking_analysis,
    print_sensitivity_summary,
    weight_sweep,
)
from polyculture_qubo.analysis.validation import (
    leave_one_out_cv,
    pairwise_contribution_table,
    pairwise_ranking_check,
    print_validation_summary,
)


def load_matrices():
    """Load preprocessed matrices from disk."""
    j = pd.read_csv(PROCESSED_DIR / "interaction_matrix.csv", index_col=0)
    d = pd.read_csv(PROCESSED_DIR / "diversity_matrix.csv", index_col=0)
    b = pd.read_csv(PROCESSED_DIR / "linear_biases.csv", index_col=0)
    c = pd.read_csv(PROCESSED_DIR / "confidence_matrix.csv", index_col=0)
    return j, d, b, c


def run_analysis(
    k_values: list[int] | None = None,
    skip_sensitivity: bool = False,
    skip_loo: bool = False,
    with_qaoa: bool = False,
    qaoa_depths: list[int] | None = None,
) -> None:
    """Run all Phase 4 analyses.

    Args:
        k_values: Target species counts. Default: [3, 4, 5].
        skip_sensitivity: Skip weight sweep and data masking.
        skip_loo: Skip leave-one-out cross-validation.
        with_qaoa: Run QAOA benchmark and include in solver comparison.
        qaoa_depths: QAOA circuit depths. Default: [1, 2].
    """
    if k_values is None:
        k_values = [3, 4, 5]
    if qaoa_depths is None:
        qaoa_depths = [1, 2]

    j_matrix, d_matrix, b_matrix, c_matrix = load_matrices()
    ler_stats = pd.read_csv(PROCESSED_DIR / "pairwise_ler_stats.csv")

    landscape_metrics = {}
    q_matrices = {}
    solver_data = {}  # For solver comparison plot
    # Primary QAOA quality metric: beta, relative to a random feasible draw.
    # 1 = optimal, 0 = no better than random, < 0 = worse than random.
    qaoa_beta: dict[int, dict[int, float]] = {}
    qaoa_rank: dict[int, dict[int, tuple[int | None, int | None]]] = {}

    # =========================================================
    # 1. ENERGY LANDSCAPE ANALYSIS
    # =========================================================
    print("\n" + "=" * 70)
    print("  PHASE 4: ANALYSIS & VISUALIZATION")
    print("=" * 70)

    for k in k_values:
        config = QUBOConfig(target_species=k)
        q, species_keys = build_qubo_matrix(
            j_matrix, d_matrix, b_matrix, config, c_matrix
        )
        q_matrices[k] = q

        # Compute λ for penalty decomposition
        lam = compute_penalty_strength(
            j_matrix, d_matrix, b_matrix, config, species_keys
        )

        # Exact solve to get full landscape
        exact = ExactSolver().solve(q, target_species=k)
        metrics = analyze_landscape(exact, q, penalty_strength=lam)
        landscape_metrics[k] = metrics
        print_landscape_metrics(metrics)

        # Plots
        path = plot_energy_histogram(exact.all_energies, exact.best_energy, k)
        print(f"  -> Saved: {path}")
        path = plot_gap_structure(metrics.sorted_energies_bottom, k)
        print(f"  -> Saved: {path}")
        path = plot_objective_decomposition(
            exact.all_energies, metrics.penalty_offset, k
        )
        print(f"  -> Saved: {path}")

        # Collect solver data
        solver_data[f"exact (k={k})"] = {
            "energy": exact.best_energy,
            "time": exact.wall_time_seconds,
        }

        # SA benchmark
        sa = SimulatedAnnealingSolver().solve(
            q, target_species=k, num_reads=100, num_sweeps=1000, seed=42
        )
        solver_data[f"SA (k={k})"] = {
            "energy": sa.best_energy,
            "time": sa.wall_time_seconds,
        }
        baseline = random_baseline(q, target_species=k, seed=42)
        sa_quality = evaluate_quality(
            q,
            target_species=k,
            energy=sa.best_energy,
            reference_energy=exact.best_energy,
            baseline=baseline,
            all_feasible_energies=exact.all_energies,
            penalty_offset=metrics.penalty_offset,
        )
        print(
            f"\n  SA: energy={sa.best_energy:.4f}, time={sa.wall_time_seconds:.3f}s, "
            f"{format_quality(sa_quality)}"
        )

        # QAOA benchmark (optional)
        if with_qaoa:
            from polyculture_qubo.solvers.qaoa import QAOAConfig, QAOASolver

            qaoa_beta[k] = {}
            qaoa_rank[k] = {}
            for p in qaoa_depths:
                print(f"  QAOA p={p}: running...", end="", flush=True)
                qaoa_config = QAOAConfig(
                    depth=p, num_restarts=5, max_iter=200, shots=4096, seed=42
                )
                qaoa = QAOASolver().solve(q, target_species=k, config=qaoa_config)
                quality = evaluate_quality(
                    q,
                    target_species=k,
                    energy=qaoa.best_energy,
                    reference_energy=exact.best_energy,
                    baseline=baseline,
                    all_feasible_energies=exact.all_energies,
                    penalty_offset=metrics.penalty_offset,
                )
                qaoa_beta[k][p] = quality.beta
                qaoa_rank[k][p] = (quality.rank, quality.n_feasible)

                solver_data[f"QAOA p={p} (k={k})"] = {
                    "energy": qaoa.best_energy,
                    "time": qaoa.wall_time_seconds,
                }
                p_in = qaoa.metadata["in_constraint_probability"]
                p_uni = qaoa.metadata["uniform_in_constraint_probability"]
                print(
                    f" energy={qaoa.best_energy:.4f}, time={qaoa.wall_time_seconds:.1f}s, "
                    f"{format_quality(quality)}, "
                    f"P_in={p_in:.4f} (uniform {p_uni:.5f}), "
                    f"feasible={qaoa.metadata['feasible_solution_found']}"
                )

        # Frustration analysis (for the primary k)
        if k == 4:
            frust = frustration_analysis(exact.best_solution, species_keys, j_matrix)
            print(f"\n  Frustration Analysis (k={k}):")
            if frust["selected_competitive"]:
                print("    Competitive pairs forced together:")
                for sp_a, sp_b, j_val in frust["selected_competitive"]:
                    print(f"      {sp_a} + {sp_b}: J = {j_val:.4f}")
            else:
                print("    No competitive pairs in solution (good!)")

            if frust["split_complementary"][:5]:
                print("    Top complementary pairs split apart:")
                for sp_a, sp_b, j_val in frust["split_complementary"][:5]:
                    print(f"      {sp_a} + {sp_b}: J = {j_val:.4f}")

    # =========================================================
    # 2. RETROSPECTIVE VALIDATION (k=4 as primary)
    # =========================================================
    print("\n")
    config_k4 = QUBOConfig(target_species=4)
    q_k4, species_keys = build_qubo_matrix(
        j_matrix, d_matrix, b_matrix, config_k4, c_matrix
    )
    exact_k4 = ExactSolver().solve(q_k4, target_species=4, collect_all=False)
    selected_k4 = solution_to_species(exact_k4.best_solution, species_keys)

    # Contribution table
    contrib = pairwise_contribution_table(
        selected_k4, j_matrix, c_matrix, d_matrix, ler_stats
    )

    # Ranking check
    ranking = pairwise_ranking_check(selected_k4, ler_stats)

    # LOO-CV
    loo_results = None
    companion = None
    bischoff = None
    if not skip_loo or not skip_sensitivity:
        from polyculture_qubo.data import load_companion_plants, load_bischoff_pairs
        from polyculture_qubo.data.loader import (
            aggregate_bischoff_pairs,
            filter_to_candidate_pairs,
        )

        companion = filter_to_candidate_pairs(load_companion_plants())
        bischoff = aggregate_bischoff_pairs(
            filter_to_candidate_pairs(load_bischoff_pairs())
        )

    if not skip_loo:
        loo_results = leave_one_out_cv(ler_stats, companion, bischoff, config_k4)

    print_validation_summary(contrib, ranking, loo_results)

    # Interaction network plot
    path = plot_interaction_network(j_matrix, selected_k4)
    print(f"  -> Saved: {path}")

    # =========================================================
    # 3. SENSITIVITY ANALYSIS
    # =========================================================
    if not skip_sensitivity:
        sweep_results = weight_sweep(
            j_matrix, d_matrix, b_matrix, c_matrix, target_species=4
        )
        sweep_analysis = analyze_weight_sweep(sweep_results)

        masking_results = data_masking_analysis(
            ler_stats, companion, bischoff, config_k4
        )

        print_sensitivity_summary(sweep_analysis, masking_results)

        # Plots
        path = plot_sensitivity_heatmap(sweep_results)
        print(f"  -> Saved: {path}")
        path = plot_data_coverage(masking_results)
        print(f"  -> Saved: {path}")

    # =========================================================
    # 4. SCALABILITY PROJECTION
    # =========================================================
    scalability = compute_scalability_metrics(
        landscape_metrics,
        q_matrices,
        qaoa_beta=qaoa_beta if qaoa_beta else None,
        qaoa_rank=qaoa_rank if qaoa_rank else None,
    )
    print_scalability_summary(scalability)

    path = plot_scalability(
        scalability.k_values,
        scalability.spectral_gaps,
        scalability.degeneracies_5pct,
        scalability.solution_space_sizes,
    )
    print(f"  -> Saved: {path}")

    # Solver comparison plot
    path = plot_solver_comparison(solver_data)
    print(f"  -> Saved: {path}")

    print(f"\n{'=' * 70}")
    print("  ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 analysis")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Target species counts (default: 3 4 5)",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip weight sweep and data masking",
    )
    parser.add_argument(
        "--skip-loo", action="store_true", help="Skip leave-one-out cross-validation"
    )
    parser.add_argument(
        "--with-qaoa", action="store_true", help="Include QAOA benchmark (slower)"
    )
    parser.add_argument(
        "--qaoa-depths",
        type=int,
        nargs="+",
        default=[1, 2],
        help="QAOA circuit depths (default: 1 2)",
    )
    args = parser.parse_args()

    run_analysis(
        k_values=args.k,
        skip_sensitivity=args.skip_sensitivity,
        skip_loo=args.skip_loo,
        with_qaoa=args.with_qaoa,
        qaoa_depths=args.qaoa_depths,
    )


if __name__ == "__main__":
    main()
