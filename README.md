# Quantum Polyculture Optimizer

A quantum-computational framework for optimizing polyculture species selection in regenerative agriculture. Formulates the problem of selecting complementary crop species as a QUBO (Quadratic Unconstrained Binary Optimization) problem, solves it using exact, classical heuristic, and quantum approximate methods, and characterizes the energy landscape for quantum advantage projection.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the preprocessing pipeline (builds interaction matrices from raw data)
uv run python -m polyculture_qubo.pipeline

# Benchmark all solvers on the real problem
uv run python -m polyculture_qubo.solvers.benchmark --k 4

# Run the full analysis (landscape, validation, sensitivity, scalability)
uv run python -m polyculture_qubo.analysis.run

# Include QAOA in the analysis (slower)
uv run python -m polyculture_qubo.analysis.run --with-qaoa --qaoa-depths 1 2

# Run tests
uv run pytest tests/ -v
```

## Development

```bash
# Install git hooks (ruff format + lint + ty on commit, gitleaks + pytest on push)
uv run lefthook install

# The hooks require gitleaks on PATH
brew install gitleaks
```

Run `uv run ruff format .`, `uv run ruff check .`, and `uv run ty check src tests scripts`
to reproduce the pre-commit checks across the whole tree.

## Project Structure

```
src/polyculture_qubo/
  species.py              # 20 candidate species with trait data
  pipeline.py             # End-to-end preprocessing pipeline
  data/                   # Data loading, filtering, normalization
  matrix/                 # QUBO matrix construction
    interaction.py        # J_ij coefficients, diversity, confidence
    qubo.py               # Q matrix assembly, energy evaluation, Ising conversion
  solvers/                # Three solver implementations
    exact.py              # Brute-force enumeration (ground truth)
    annealing.py          # Simulated annealing with constraint-preserving swaps
    qaoa.py               # QAOA via Qiskit Aer simulator
    benchmark.py          # Solver comparison runner
  analysis/               # Phase 4 analysis pipeline
    landscape.py          # Energy landscape characterization
    validation.py         # Retrospective validation against LER data
    sensitivity.py        # Weight sweep and data masking
    scalability.py        # Quantum advantage projection
    plots.py              # Publication-quality figures
    run.py                # Analysis orchestrator
```

## What It Does

Given 20 candidate crop species and their pairwise interaction data (Land Equivalent Ratios from intercropping experiments), the optimizer selects the best subset of k species (k = 3, 4, or 5) that maximizes:

- **Land-use efficiency** (LER-derived interaction coefficients, weight 0.7)
- **Nitrogen balance** (fixer + non-fixer complementarity, weight 0.2)
- **Functional diversity** (height, root depth, functional group differences, weight 0.1)

The k=4 optimal solution is **common bean + maize + pea + pepper** -- a cereal-legume polyculture with 2 nitrogen fixers and 2 non-fixers, validated against empirical intercropping data.

## Key Results

- Exact enumeration and simulated annealing find the same optimum at every k. QAOA reaches it in 3 of 9 depth-by-target configurations, and at (k=3, p=1) returns a solution worse than a random feasible draw.
- The energy landscape appears flat in total energy (100% of solutions within 5% of optimum) but has real structure when the constant penalty offset is subtracted (only 0.04% within 5% of the objective-only optimum).
- The same penalty offset invalidates the conventional approximation ratio: its floor over the whole feasible set is 0.974 at k=4, so the worst possible answer still scores 97%. Solver quality is reported as β (1 = optimal, 0 = random feasible draw) and rank instead.
- QAOA concentrates 5.7x-366x more probability in the feasible subspace than uniform sampling in 8 of 9 configurations, using a standard transverse-field mixer.
- 12 distinct optimal solutions emerge across 120 weight configurations. Maize appears in 88% of all configs.
- No quantum advantage at N=20 (brute force solves in 7ms). The spectral gap shrinks with k, suggesting harder structure at larger scale.

## Data Sources

1. **Paut et al. (2024)** -- 2,231 intercropping experiments, 118 species, LER data
2. **Bischoff et al. (2024)** -- 274 species pairs from PNAS systems agroecology study
3. **Sula companion planting network** -- 995 signed edges from traditional knowledge

See [paper/report.md](paper/report.md) for the full technical report with methodology, results, and analysis.
