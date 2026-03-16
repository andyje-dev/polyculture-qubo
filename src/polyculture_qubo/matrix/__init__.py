"""Matrix construction for interaction coefficients and QUBO formulation."""

from polyculture_qubo.matrix.interaction import (
    build_diversity_matrix,
    build_interaction_matrix,
    compute_confidence_weight,
    compute_diversity_bonus,
    compute_linear_biases,
)
from polyculture_qubo.matrix.qubo import (
    QUBOConfig,
    build_nitrogen_matrix,
    build_qubo_matrix,
    evaluate_solution,
    normalize_qubo,
    print_solution,
    qubo_energy,
    qubo_to_ising,
    solution_to_species,
)

__all__ = [
    "QUBOConfig",
    "build_diversity_matrix",
    "build_interaction_matrix",
    "build_nitrogen_matrix",
    "build_qubo_matrix",
    "compute_confidence_weight",
    "compute_diversity_bonus",
    "compute_linear_biases",
    "evaluate_solution",
    "normalize_qubo",
    "print_solution",
    "qubo_energy",
    "qubo_to_ising",
    "solution_to_species",
]
