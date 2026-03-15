# Quantum Polyculture Optimizer

## Project Overview

This project develops the first quantum-computational framework for optimizing polyculture species selection in regenerative agriculture. We formulate the problem of selecting complementary crop species for a single field as a Quadratic Unconstrained Binary Optimization (QUBO) problem, solve it using both quantum and classical methods, and benchmark the results against empirical intercropping data.

The primary contribution is the **formulation itself** — mapping real agronomic interaction data to a quantum-native optimization structure — along with energy landscape characterization to assess quantum advantage potential at scale.

---

## Problem Definition

### Core Problem
Given a single agricultural field with known soil and climate parameters, select a subset of species (3–5) from a candidate pool of 15–20 well-studied crop species that maximizes a composite objective combining:

1. **Land Equivalent Ratio (LER)** — the dominant term, measuring land-use efficiency of the mixture vs. monocultures
2. **Nitrogen balance** — a bonus for including nitrogen-fixing species (legumes) proportional to the non-legume fraction
3. **Functional diversity** — a mild bonus for selecting species with complementary trait profiles (root depth, height, growth timing)

### Constraints
- Maximum number of species the field can support (e.g., 3–5)
- Species viability for the given soil type and USDA hardiness zone
- Minimum inclusion of at least one cash crop (economic viability)

### What We Are NOT Solving
- Spatial arrangement within the field (row spacing, strip configuration)
- Planting density optimization
- Multi-season temporal rotation planning
- Cultivar-level selection (we work at the species level)

These are explicitly flagged as future extensions.

---

## Objective Function Design

### Composite Objective (Plain Language)
Maximize: `α × LER_score + β × N_balance_score + γ × diversity_score`

Where:
- `α` ≈ 0.7 (LER dominates — this is where we have the most data)
- `β` ≈ 0.2 (nitrogen fixation bonus)
- `γ` ≈ 0.1 (functional diversity bonus)

Weights are tunable hyperparameters for sensitivity analysis.

### LER Score
For a selected subset S of species, the LER score is derived from pairwise interaction coefficients:
- Each pair (i, j) has a coefficient `J_ij` derived from observed LER data
- `J_ij > 0` means species i and j are complementary (LER > 1.0 when intercropped)
- `J_ij < 0` means they are competitive (LER < 1.0)
- `J_ij ≈ 0` means neutral or unknown interaction

### Interaction Coefficient Classification
Based on empirical LER data from the intercropping literature:
- **Positive interaction**: LER > 1.05 with adequate replication → negative QUBO coupling (rewards co-selection)
- **Negative interaction**: LER < 0.95 with adequate replication → positive QUBO coupling (penalizes co-selection)
- **Inconclusive**: LER between 0.95–1.05, or insufficient replication → coefficient shrunk toward zero

### Confidence Weighting
Each `J_ij` is scaled by a confidence factor based on:
- Number of independent observations for the pair
- Variance across experiments
- Environmental context match between source data and target field

This is critical because most species pairs have sparse or no direct observation data.

### Handling Unknown Pairs
For species pairs with no empirical interaction data:
- Apply a **mildly negative prior** (small penalty for co-selection)
- Rationale: random species combinations are more likely to compete than complement; positive interactions require specific functional trait complementarity
- The prior can be refined using trait-based predictions (height difference, root depth complementarity, N-fixation capacity)

---

## QUBO Formulation Sketch

### Decision Variables
Binary variables `x_i ∈ {0, 1}` for each candidate species i, where:
- `x_i = 1` means species i is selected for the polyculture
- `x_i = 0` means it is excluded

For N candidate species, we have N binary variables.

### Objective (to minimize)
```
H = -Σ_i h_i * x_i - Σ_{i<j} J_ij * x_i * x_j + P_count + P_viability
```

Where:
- `h_i` = linear bias for species i (encodes individual species value: yield potential, N-fixation bonus, market value)
- `J_ij` = pairwise coupling between species i and j (derived from LER data, positive = complementary, negative = competitive)
- `P_count` = penalty term enforcing species count constraint (e.g., select exactly k species)
- `P_viability` = penalty for selecting species not viable in the target environment

Note: QUBO minimizes, so we negate the beneficial terms.

### Species Count Constraint
Enforced via penalty: `P_count = λ * (Σ_i x_i - k)²`

Where k is the target number of species and λ is a penalty strength that must be large enough to enforce the constraint but not so large that it dominates the objective.

### Expansion to QUBO Standard Form
The full objective expands to:
```
H = Σ_i Q_ii * x_i + Σ_{i<j} Q_ij * x_i * x_j
```

Where:
- `Q_ii` = -h_i + λ * (1 - 2k)  (diagonal terms: linear biases adjusted by constraint)
- `Q_ij` = -J_ij + 2λ              (off-diagonal terms: interaction coefficients adjusted by constraint)

This Q matrix is the input to both quantum and classical solvers.

---

## Data Sources

### Primary: Species Interaction Data (QUBO Coefficients)

#### 1. PNAS Systems Agroecology Dataset (Bischoff et al., Dec 2024)
- **Content**: 2,258 intercropping experiments, 274 species pairs, 69 plant species
- **Variables**: 4 soil characteristics, 5 environmental/farming conditions, 8 traits per plant
- **Key value**: Authors performed dimensionality reduction showing a few variables predict IC yield vs. sole cultivation
- **Format**: Supplementary data from PNAS paper
- **Paper DOI**: 10.1073/pnas.2415315121
- **Access**: Open access (CC BY-NC-ND)

#### 2. Paut et al. Horticulture Intercropping Dataset (Scientific Data, Jan 2024)
- **Content**: 1,544 experiments from 191 articles, 118 crop species, 5 continents
- **Variables**: 45 columns including soil/climate conditions, intercropping design, management, yields, LER
- **Key value**: Freely reusable spreadsheet, lat/long coordinates link to ISRIC/WorldClim
- **Paper DOI**: 10.1038/s41597-023-02831-7
- **Access**: Open access, freely downloadable

#### 3. npj Sustainable Agriculture Meta-Analysis (Jan 2026)
- **Content**: 4,195 partial LER observations, 334 studies, 60 countries
- **Key value**: Identifies relative planting density, temporal niche differentiation, and relative height difference as key optimization levers
- **Paper DOI**: 10.1038/s44264-025-00110-z

#### 4. Chris Sula Companion Planting Network (Simple Graph)
- **Content**: Species pair graph with signed weights (+2 beneficial, -1 antagonistic, +1 general benefit, +3 pest control)
- **Key value**: Already structured as a weighted network; useful for first-pass QUBO before incorporating LER data
- **Format**: CSV (Source, Target, Type, Weight, Label)
- **Access**: Open, available on Kaggle as "Companion Plants" dataset

### Secondary: Plant Functional Traits (Interaction Model Parameterization)

#### 5. TRY Plant Trait Database
- **Content**: Global database of plant functional traits
- **Key traits for this project**:
  - N-fixation capacity (~11,000 species)
  - Plant height (~18,000 species)
  - Root rooting depth (~733 species)
  - Specific leaf area / SLA (~9,000 species)
  - Seed mass (~27,000 species)
  - Leaf nitrogen content (~7,000 species)
- **Access**: Request-based at try-db.org (free for research)
- **Use**: Predict interaction coefficients for untested species pairs based on trait complementarity

#### 6. NodDB — Global Database of Plants with Root-Symbiotic Nitrogen Fixation
- **Content**: Catalog of species with confirmed N-fixation capacity
- **Use**: Parameterize the nitrogen balance term in the objective function

### Tertiary: Environmental Context (Constraint Data)

#### 7. USDA SSURGO (Soil Survey Geographic Database)
- **Content**: Soil properties for most US counties — available water capacity, soil pH, electrical conductivity, flooding frequency, crop yields
- **Access**: Free via Web Soil Survey or Soil Data Access API
- **Use**: Constrain candidate species to those viable for a target field's soil type

#### 8. WorldClim
- **Content**: Global climate data at ~1km resolution (temperature, precipitation, bioclimatic variables)
- **Access**: Free download at worldclim.org
- **Use**: Constrain candidate species to appropriate climate zones

#### 9. USDA NASS (National Agricultural Statistics Service)
- **Content**: Crop productivity benchmarks and commodity pricing
- **Use**: Economic viability constraints and market value terms in objective function

---

## Recommended Target Scenario

### Focus: Cereal-Legume-Cover Crop System in Temperate Climate
**Rationale**: This is where intercropping literature is richest, LER data most abundant, and regenerative agriculture relevance strongest (nitrogen fixation, soil cover, carbon inputs).

### Suggested Candidate Species Pool (15–20 species)
Select from the most well-studied species in the PNAS and Paut et al. datasets:

**Cereals**: Maize (Zea mays), Wheat (Triticum aestivum), Barley (Hordeum vulgare), Oat (Avena sativa), Sorghum (Sorghum bicolor)

**Legumes**: Faba bean (Vicia faba), Pea (Pisum sativum), Soybean (Glycine max), Cowpea (Vigna unguiculata), Common bean (Phaseolus vulgaris), Lentil (Lens culinaris)

**Cover crops / Support species**: Crimson clover (Trifolium incarnatum), Radish (Raphanus sativus), Sunflower (Helianthus annuus), Camelina (Camelina sativa), Lupine (Lupinus spp.)

### Target Climate Zone
USDA Hardiness Zones 5–7 (temperate continental), or Köppen Cfb/Dfb — where the majority of intercropping experiments were conducted.

### Final species pool selection
Before implementation, filter the candidate pool to retain only species with ≥3 pairwise interaction observations in the combined dataset. This ensures minimum data coverage for QUBO coefficients.

---

## Validation Strategy

### Approach: Retrospective Validation
We cannot validate full polyculture recommendations (no ground-truth multi-species combinatorial trials exist). Instead:

1. **Pairwise ranking validation**: Take a climate zone/soil type from the Paut et al. dataset where multiple species pairs were tested. Reconstruct the interaction matrix. Run the QUBO. Check whether the solver's top-ranked pairs match the experimentally observed high-LER pairs.

2. **Leave-one-out cross-validation**: For species pairs with multiple observations, hold out one observation, build the QUBO from remaining data, and check whether the solver still identifies that pair as beneficial/detrimental consistent with the held-out observation.

3. **Solver benchmarking**: Compare solution quality and runtime across:
   - QAOA (Qiskit Aer simulator, varying circuit depth p=1 to p=5)
   - Quantum annealing (D-Wave Advantage, if access available)
   - Simulated annealing (classical baseline)
   - Exact solver (brute force for small instances, Gurobi for larger)

### Metrics
- Solution quality: objective function value of best solution found
- Time to solution: wall-clock time to reach within 1% of best known solution
- Solution diversity: number of distinct near-optimal solutions (within 5% of optimum)
- Approximation ratio: QAOA solution quality vs. exact optimum

---

## Known Limitations (Explicitly Flagged)

1. **Pairwise approximation**: QUBO encodes only pairwise species interactions. Real polycultures have higher-order effects (3-way, 4-way interactions) that are not captured. Future work: PUBO (Polynomial Unconstrained Binary Optimization) for higher-order terms.

2. **Aggregated interaction data**: LER coefficients are averaged across sites, years, and management practices. Site-specific predictions require local calibration data we don't have.

3. **No spatial arrangement optimization**: We select which species to include but not where to place them. Spatial configuration significantly affects outcomes.

4. **No density optimization**: Planting density ratios are collapsed. The same species pair can be complementary at one density and competitive at another.

5. **No temporal dynamics**: We optimize for a single season snapshot. Within-season interaction dynamics are not modeled.

6. **No microbiome data**: Soil microbiome mediates many species interactions (mycorrhizal networks, rhizosphere effects) but no open dataset links intercropping outcomes to microbiome composition.

7. **No quantum advantage expected at this scale**: With 15–20 species, classical solvers will match or beat quantum. The value is in formulation, problem structure characterization, and scalability projection.

---

## Implementation Plan

### Phase 1: Data Acquisition & Preprocessing
- Download Paut et al. dataset (Scientific Data supplementary)
- Download/extract PNAS systems agroecology dataset
- Download Sula companion planting graph from Kaggle
- Request relevant traits from TRY database (height, root depth, N-fixation, SLA)
- Build unified species interaction matrix with confidence scores
- Filter candidate species pool based on data coverage

### Phase 2: QUBO Construction
- Implement interaction coefficient calculation from LER data
- Implement confidence weighting
- Implement trait-based prior for unknown pairs
- Build Q matrix with constraint penalty terms
- Validate Q matrix properties (symmetry, sparsity pattern, coefficient distribution)

### Phase 3: Solver Implementation
- QAOA implementation using Qiskit (Aer simulator)
  - Parameterized circuits with depth p = 1, 2, 3, 5
  - Classical optimizer for variational parameters (COBYLA or SPSA)
- Simulated annealing baseline (e.g., D-Wave Neal or custom)
- Exact solver for ground truth (brute force at N ≤ 20 is feasible)
- Optional: D-Wave Advantage via Leap cloud access

### Phase 4: Analysis & Visualization
- Compare solver results across methods
- Energy landscape characterization (degeneracy, gap structure, local minima density)
- Retrospective validation against empirical data
- Scalability projection (how does problem hardness grow with N?)
- Sensitivity analysis on objective function weights (α, β, γ)

### Phase 5: Documentation & Output
- Technical report / paper draft
- Visualization of species interaction graph with QUBO solution highlighted
- Scalability analysis figures
- Code repository with reproducible pipeline

---

## Tech Stack

- **Language**: Python 3.10+
- **Quantum**: Qiskit (QAOA, Aer simulator), optionally D-Wave Ocean SDK
- **Data processing**: pandas, numpy, scipy
- **Graph analysis**: networkx (species interaction graph)
- **Visualization**: matplotlib, seaborn, plotly
- **Optimization**: Gurobi (exact QUBO solver, free academic license) or PuLP
- **Data storage**: CSV/parquet for interaction matrices, JSON for configuration

---

## Key References

1. Bischoff et al. "Toward systems agroecology: Design and control of intercropping." PNAS, Dec 2024. DOI: 10.1073/pnas.2415315121
2. Paut et al. "A global dataset of experimental intercropping and agroforestry studies in horticulture." Scientific Data, Jan 2024. DOI: 10.1038/s41597-023-02831-7
3. npj Sustainable Agriculture meta-analysis on intercropping drivers. Jan 2026. DOI: 10.1038/s44264-025-00110-z
4. "The productive performance of intercropping." PNAS, Jan 2023. DOI: 10.1073/pnas.2201886120
5. Kattge et al. "TRY plant trait database – enhanced coverage and open access." Global Change Biology, 2020.
6. Mead & Willey. "The concept of a 'land equivalent ratio' and advantages in yields from intercropping." Experimental Agriculture, 1980.

---

## Success Criteria

The project is successful if we:
1. ✅ Produce a valid QUBO formulation grounded in real agronomic data
2. ✅ Demonstrate the formulation works (solver finds physically meaningful solutions — not random species subsets)
3. ✅ Benchmark quantum vs. classical solvers on the same instance
4. ✅ Characterize the energy landscape to make credible projections about quantum advantage at scale
5. ✅ Identify and articulate the specific data gaps that would need to be filled for practical deployment
6. ✅ Produce a reusable, extensible codebase that the community can build on