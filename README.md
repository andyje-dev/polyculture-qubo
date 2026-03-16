# Quantum Polyculture Optimizer

A quantum-computational framework for optimizing polyculture species selection in regenerative agriculture. Formulates the problem of selecting complementary crop species as a QUBO (Quadratic Unconstrained Binary Optimization) problem, solves it using exact, classical heuristic, and quantum approximate methods, and characterizes the energy landscape for quantum advantage projection.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the preprocessing pipeline (builds interaction matrices from raw data)
python -m polyculture_qubo.pipeline

# Benchmark all solvers on the real problem
python -m polyculture_qubo.solvers.benchmark --k 4

# Run the full analysis (landscape, validation, sensitivity, scalability)
python -m polyculture_qubo.analysis.run

# Include QAOA in the analysis (slower)
python -m polyculture_qubo.analysis.run --with-qaoa --qaoa-depths 1 2

# Run tests
python -m pytest tests/ -v
```

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

- All three solvers find the same optimum. SA matches exact; QAOA achieves 99.8% approximation ratio.
- The energy landscape appears flat in total energy (100% of solutions within 5% of optimum) but has real structure when the constant penalty offset is subtracted (only 0.04% within 5% of the objective-only optimum).
- 12 distinct optimal solutions emerge across 120 weight configurations. Maize appears in 88% of all configs.
- No quantum advantage at N=20 (brute force solves in 7ms). The spectral gap shrinks with k, suggesting harder structure at larger scale.

## Data Sources

1. **Paut et al. (2024)** -- 2,231 intercropping experiments, 118 species, LER data
2. **Bischoff et al. (2024)** -- 274 species pairs from PNAS systems agroecology study
3. **Sula companion planting network** -- 995 signed edges from traditional knowledge

See `CLAUDE.md` for full data source documentation and methodology.
