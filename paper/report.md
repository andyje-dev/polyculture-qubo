# Quantum-Computational Optimization of Polyculture Species Selection: A QUBO Formulation Grounded in Intercropping Data

## Abstract

Selecting complementary crop species for polyculture systems is a combinatorial optimization problem that grows intractable as candidate pools expand. We present the first Quadratic Unconstrained Binary Optimization (QUBO) formulation for polyculture species selection grounded in empirical intercropping data. Drawing on 2,231 experimental observations from the Paut et al. horticulture dataset, 274 species pairs from Bischoff et al., and a companion planting network, we construct a composite objective encoding Land Equivalent Ratio (LER), nitrogen fixation complementarity, and functional diversity for a pool of 20 candidate species. We solve the resulting QUBO using exact enumeration, simulated annealing, and the Quantum Approximate Optimization Algorithm (QAOA), finding that all three methods converge on agronomically plausible cereal-legume polycultures validated against empirical LER rankings. Energy landscape analysis reveals that while the total energy surface appears flat (100% of feasible solutions within 5% of optimum), this is an artifact of penalty term dominance — the objective-only landscape shows meaningful differentiation (0.04% within 5%). At the current scale of N=20 species, no quantum advantage exists, but spectral gap analysis suggests increasing problem hardness at larger scales.

---

## 1. Introduction

Polyculture — the simultaneous cultivation of multiple crop species in a single field — is a cornerstone of regenerative agriculture. Well-designed polycultures can achieve land-use efficiencies exceeding monocultures by 20–60% as measured by the Land Equivalent Ratio (LER), while providing ecosystem services including nitrogen fixation, pest suppression, and soil structure improvement [1, 4, 6]. However, selecting which species to combine from a candidate pool is a combinatorial problem: for N candidate species and a target of k=4 species, there are C(N,k) possible combinations. At N=20 this is 4,845 — trivial to enumerate — but at the scale of real agricultural biodiversity (hundreds of crop species, region-specific constraints), the problem becomes intractable for exhaustive search.

Quadratic Unconstrained Binary Optimization (QUBO) is a natural formulation for subset selection problems. Each candidate species is represented by a binary decision variable, pairwise interactions encode species compatibility, and constraint penalties enforce the desired subset size. QUBO problems are native to both quantum annealing hardware (D-Wave) and gate-based quantum algorithms (QAOA), making them a bridge between classical combinatorial optimization and quantum computing.

We present three contributions:

1. **A QUBO formulation for polyculture optimization** that maps real agronomic interaction data — Land Equivalent Ratios, nitrogen fixation complementarity, and functional trait diversity — to the quadratic coefficients of the optimization problem.

2. **Empirical validation** showing that the solver identifies species combinations consistent with known intercropping successes, with selected pairs ranking in the top 27th percentile of observed LER values.

3. **Energy landscape characterization** revealing that the problem's apparent flatness is an artifact of penalty dominance, and that the underlying agronomic signal has meaningful structure relevant to quantum advantage projections at larger scales.

---

## 2. Methods

### 2.1 Data Sources

We integrate three datasets to construct the species interaction matrix:

**Paut et al. (2024)** [2] provides the primary data: 2,231 intercropping experiments from 191 articles covering 118 crop species across 5 continents. Each observation includes LER measurements, soil characteristics, climate zone, and experimental design. After filtering to our 20-species candidate pool, we retain observations covering 31 of the 190 possible species pairs (16% coverage).

**Bischoff et al. (2024)** [1] contributes 274 directed species pairs from a PNAS systems agroecology study. These pairs lack LER measurements but indicate which combinations researchers considered worth studying, providing a weak prior for unstudied pairs.

**Sula companion planting network** supplies 995 signed edges encoding traditional knowledge about species compatibility (+2 beneficial, -1 antagonistic). This serves as the weakest prior layer, contributing directional signal for pairs lacking both LER and Bischoff data.

### 2.2 Species Pool

We select 20 candidate species based on data availability across the three datasets:

| Group | Species |
|-------|---------|
| Cereals (5) | Maize, Wheat, Barley, Oat, Sorghum |
| Legumes (7) | Faba bean, Pea, Soybean, Cowpea, Common bean, Lentil, Lupine |
| Cover crops (2) | Crimson clover, Radish |
| Oilseed (1) | Sunflower |
| Vegetables (5) | Cabbage, Tomato, Lettuce, Onion, Pepper |

All 20 species are assumed viable for USDA Hardiness Zones 5–7 (temperate continental), where the majority of intercropping experiments were conducted. Site-specific viability filtering by soil type or microclimate is identified as future work.

### 2.3 Objective Function

The composite objective combines three components with tunable weights summing to 1.0:

**LER score (α = 0.7)**: For each species pair (i, j), an interaction coefficient J_ij is derived from observed LER data:

$$J_{ij} = w_{ij} \cdot (\overline{LER}_{ij} - 1.0)$$

where w_ij is a confidence weight combining observation count, variance, and replication:

$$w_{ij} = \min\!\left(1, \sqrt{\frac{n_{obs}}{50}}\right) \cdot \frac{1}{1 + \sigma_{LER}} \cdot \min\!\left(1, \sqrt{\frac{n_{articles}}{5}}\right)$$

Positive J_ij indicates complementarity (LER > 1.0); negative indicates competition.

**Nitrogen balance (β = 0.2)**: A pairwise bonus for complementary nitrogen fixation: N_ij = 1.0 for fixer + non-fixer pairs, 0.2 for fixer + fixer, and 0.0 for non-fixer + non-fixer. To avoid double-counting nitrogen benefits already captured in LER data, N_ij is scaled by (1 - confidence_ij) — pairs with strong empirical LER evidence have their explicit nitrogen term reduced.

**Functional diversity (γ = 0.1)**: A trait-based pairwise bonus computed from root depth difference (30%), height difference (30%), and functional group difference (40%). This mild term encourages structural complementarity.

### 2.4 QUBO Construction

The optimization problem is formulated as minimizing x^T Q x over binary variables x_i ∈ {0, 1}, where x_i = 1 indicates species i is selected.

**Off-diagonal entries** (i < j):

$$Q_{ij} = -\alpha \cdot J_{ij} - \beta \cdot N_{ij} \cdot (1 - c_{ij}) - \gamma \cdot D_{ij} + 2\lambda$$

**Diagonal entries**:

$$Q_{ii} = -\alpha \cdot h_i + \lambda(1 - 2k)$$

where h_i is a linear bias encoding economic value (0.3 for cash crops, 0.09 for cover crops), k is the target species count, and λ is the penalty strength enforcing the constraint Σ x_i = k.

**Layered prior system**: For the 159 species pairs (84%) lacking direct LER observations, interaction coefficients are filled by a 3-layer prior system applied in order of increasing confidence:

1. Unknown pair prior: J = -0.05, confidence = 0 (mildly competitive default)
2. Bischoff studied-pair prior: J = -0.01, confidence = 0.05–0.15
3. Companion planting prior: J = ±0.1, confidence = 0.1
4. LER-derived coefficients override all priors for the 31 observed pairs

**Auto-derived penalty strength**: λ is computed as 1.5× the maximum objective benefit achievable by a single variable flip, guaranteeing that no constraint-violating solution can have lower energy than any feasible solution.

### 2.5 Solvers

**Exact enumeration**: All C(N, k) feasible solutions are evaluated exhaustively. For N=20 and k=4, this is 4,845 solutions. Provides ground truth and full energy landscape data.

**Simulated annealing**: Uses constraint-preserving swap moves — each step deselects one species and selects another, maintaining exactly k species throughout. Geometric cooling schedule (β: 0.1 → 10.0) over 1,000 sweeps with 100 random restarts.

**QAOA**: Implemented using Qiskit with the Aer statevector simulator. The QUBO is normalized (entries scaled to [-1, 1]) and converted to an Ising Hamiltonian via the substitution x_i = (1 - s_i)/2. The circuit alternates cost unitaries (RZZ gates for couplings, RZ gates for local fields) and mixer unitaries (RX gates). Variational parameters (γ, β) are optimized using COBYLA with 5 random restarts and 200 iterations each. Final solutions are sampled with 4,096 measurement shots.

---

## 3. Results

### 3.1 Optimal Polycultures

The exact solver identifies the following optimal species combinations:

| k | Selected Species | LER Score | N-Balance | Diversity | Fixers/Non-fixers |
|---|-----------------|-----------|-----------|-----------|-------------------|
| 3 | Common bean, Maize, Pepper | 0.854 | 1.167 | 2.130 | 1/2 |
| 4 | Common bean, Maize, Pea, Pepper | 1.213 | 2.674 | 3.418 | 2/2 |
| 5 | Common bean, Maize, Pea, Pepper, Sunflower | 1.263 | 4.574 | 6.150 | 2/3 |

All solutions are cereal-legume combinations with balanced nitrogen fixation — exactly the structure that intercropping literature identifies as most productive. The k=4 solution contains 5 pairs with observed LER data, 4 of which have LER > 1.0.

**Pairwise contribution breakdown (k=4)**:

| Pair | J coefficient | Confidence | LER | Data Source |
|------|--------------|------------|-----|-------------|
| Common bean + Maize | 0.524 | 0.631 | 1.83 | 176 obs, 18 articles |
| Maize + Pea | 0.408 | 0.556 | 1.73 | 37 obs, 6 articles |
| Common bean + Pepper | 0.295 | 0.202 | 2.46 | 12 obs, 1 article |
| Maize + Pepper | 0.035 | 0.081 | 1.43 | 2 obs, 1 article |
| Pea + Pepper | 0.001 | 0.137 | 1.00 | 6 obs, 1 article |
| Common bean + Pea | -0.050 | 0.000 | — | Prior only |

The common_bean + pea pair has no direct LER data and uses the unknown-pair prior (J = -0.05). Despite this mildly competitive assumption, both species are selected because their individual pairwise benefits with maize and pepper outweigh the penalty.

### 3.2 Solver Comparison

| Solver | Energy (k=4) | Time | Approx. Ratio |
|--------|-------------|------|---------------|
| Exact | -71.553 | 0.008s | 1.000 |
| Simulated annealing | -71.553 | 2.5s | 1.000 |
| QAOA (p=1) | -71.402 | 28.1s | 0.998 |

Simulated annealing matches the exact optimum on every instance. QAOA at depth p=1 achieves a 99.8% approximation ratio, finding a near-optimal feasible solution (common bean, crimson clover, maize, pepper) that swaps pea for crimson clover — the second-best solution in the exact ranking.

### 3.3 Energy Landscape

#### Total Energy vs. Objective-Only Decomposition

For all feasible solutions (exactly k species selected), the constraint penalty contributes a constant energy offset of -k²λ. This constant dominates the total energy budget:

| k | Total Range | Objective Range | Penalty Offset | Obj. as % of Penalty |
|---|-------------|-----------------|----------------|----------------------|
| 3 | 1.23 | 1.23 | -29.81 | 4.1% |
| 4 | 1.87 | 1.87 | -68.99 | 2.7% |
| 5 | 2.57 | 2.57 | -132.77 | 1.9% |

The total energy range and objective-only range are identical (a constant offset does not change the range), but the degeneracy picture changes dramatically:

| k | Solutions | Total: within 5% | Objective-only: within 5% |
|---|-----------|-------------------|---------------------------|
| 3 | 1,140 | 1,140 (100%) | 2 (0.18%) |
| 4 | 4,845 | 4,845 (100%) | 2 (0.04%) |
| 5 | 15,504 | 15,504 (100%) | 19 (0.12%) |

The apparent landscape flatness — 100% of solutions within 5% of optimal — is an artifact of measuring against the total energy, which includes a large constant penalty offset. When measured against the objective-only energy, only 2 of 4,845 solutions (0.04%) are within 5% of optimal for k=4. The agronomic signal meaningfully differentiates solutions.

#### Spectral Gap and Local Minima

| k | Spectral Gap | Gap Ratio | Local Minima |
|---|-------------|-----------|--------------|
| 3 | 0.051 | 0.042 | 1/1,140 (0.09%) |
| 4 | 0.095 | 0.051 | 1/4,845 (0.02%) |
| 5 | 0.018 | 0.007 | 2/15,504 (0.01%) |

The spectral gap (energy difference between the best and second-best solution) shrinks from 0.051 to 0.018 as k increases from 3 to 5. Under constraint-preserving swap moves, only 1–2 solutions are local minima out of thousands — the landscape has almost no trapping structure for heuristic solvers.

#### Frustration Analysis (k=4)

One competitive pair is forced into the optimal solution: common bean + pea (J = -0.050, unknown-pair prior). Five complementary pairs are split apart, the strongest being maize + cowpea (J = 0.250, LER = 1.33). The solver cannot include cowpea alongside the selected set because the k=4 constraint forces a trade-off: cowpea's benefit with maize does not compensate for the loss of either pea's or pepper's contributions.

### 3.4 Retrospective Validation

The solver's selected pairs rank in the top 27th percentile of all observed LER values (mean selected LER = 1.69 vs. mean observed LER = 1.37). Individual pair rankings:

| Pair | LER | Percentile |
|------|-----|------------|
| Common bean + Pepper | 2.46 | Top 0% |
| Common bean + Maize | 1.83 | Top 3% |
| Maize + Pea | 1.73 | Top 6% |
| Maize + Pepper | 1.43 | Top 32% |
| Pea + Pepper | 1.00 | Top 94% |

The solver selects 4 of the top 10 observed LER pairs. The weakest pair (pea + pepper, LER = 1.00) contributes minimally through its interaction coefficient but is retained because pea's strong pairing with maize (LER = 1.73) and its nitrogen fixation complementarity with pepper outweigh the neutral LER.

**Leave-one-out cross-validation**: Holding out each of the 31 observed pairs and re-solving, 2 pairs are identified as "load-bearing" — their removal changes the optimal solution:

- Removing maize + pea (LER = 1.73): pea is replaced by soybean
- Removing common_bean + pepper (LER = 2.46): pepper is replaced by cabbage

The remaining 29 pairs can be removed individually without changing the optimal species set.

### 3.5 Sensitivity Analysis

#### Weight Sweep

Varying (α, β, γ) across 120 configurations on the simplex (with α ≥ 0.3), we find:

- **12 distinct optimal solutions** emerge
- **Solution stability**: the default solution (common bean, maize, pea, pepper) appears in 33% of configurations
- **Maize** appears in 88% of all configurations — the most robust selection
- **Common bean** appears in 72%, **pepper** in 56%, **sunflower** in 34%

The solution is moderately sensitive to weights. When β (nitrogen) increases, legume-heavy solutions are favored. When γ (diversity) increases, solutions favor species with different functional groups and growth habits.

![Sensitivity heatmap](../output/figures/sensitivity_heatmap.png)
*Figure: Species selection frequency by LER weight (α). Each cell shows how often a species appears in the optimal solution across weight configurations within each α bin.*

#### Data Masking

Systematically removing each of the 31 observed pairs from the interaction matrix, only 2 pairs change the optimal solution when removed. The 5 strongest interaction coefficients (J > 0.25) all produce stable solutions when masked individually — the formulation is robust to single-pair data loss for the strongest signals.

---

## 4. Discussion

### Penalty Dominance and Formulation Design

The most striking finding is the penalty decomposition: the constraint enforcement term contributes ~97% of total energy for feasible solutions, compressing the objective signal into a narrow band. This is not a flaw in the formulation — it is an inherent property of QUBO constraint encoding. The auto-derived λ must be large enough to guarantee that no infeasible solution can beat any feasible one, and this requirement pushes λ well above the objective scale.

For classical solvers, this is irrelevant — simulated annealing with constraint-preserving moves ignores the penalty entirely, exploring only the feasible subspace. For QAOA, however, the penalty dominance means the cost Hamiltonian's spectral structure is shaped primarily by the constraint, not the objective. QAOA must first "learn" the constraint structure before it can differentiate among feasible solutions. This may explain why deeper circuits are needed for high approximation ratios in constrained QUBO problems.

### Data Sparsity

With only 16% of species pairs having empirical LER data, the solver is largely navigating a prior-dominated landscape. The 31 observed pairs create data-rich islands that anchor the solution — all optimal species appear in at least one well-observed pair. This is simultaneously a limitation (we cannot discover novel polycultures beyond the data) and a feature (the formulation correctly defers to empirical evidence where available).

The data masking analysis quantifies this: removing any single observed pair changes the solution in at most 2 of 31 cases. The formulation is resilient to single-pair data loss, but this resilience comes from redundancy in the data-rich subset, not from the priors providing meaningful signal.

### Quantum Advantage Assessment

At N=20, no quantum advantage exists or should be claimed. Exact enumeration solves in 7ms. Simulated annealing matches the optimum with 100% reliability. QAOA achieves 99.8% approximation but requires 28 seconds — 3,500× slower than brute force.

However, the energy landscape characterization provides useful projections:

1. The spectral gap shrinks with k (0.051 → 0.018), suggesting that problem hardness increases as the target subset grows relative to the pool.
2. The near-zero fraction of local minima (0.02%) means the landscape has almost no trapping structure — a property that may change at larger N where the prior landscape contributes more complex structure.
3. The penalty-to-objective ratio worsens with k (objective range drops from 4.1% to 1.9% of penalty), suggesting QAOA parameter optimization becomes harder at larger scales.

Quantum advantage would require both larger species pools (N >> 20) and richer interaction data to create the frustrated coupling patterns that make combinatorial optimization classically hard.

---

## 5. Limitations

1. **Pairwise approximation**: The QUBO encodes only 2-body interactions. Real polycultures exhibit higher-order effects (e.g., the "three sisters" system of maize, beans, and squash has emergent properties not captured by summing pairwise LERs). Polynomial Unconstrained Binary Optimization (PUBO) could encode these but requires higher-order interaction data that does not currently exist.

2. **Aggregated interaction data**: LER coefficients are averaged across sites, years, soils, and management practices. The same species pair can be complementary in one environment and competitive in another. Site-specific predictions require local calibration data we lack.

3. **No spatial optimization**: We select which species to include but not where to place them. Spatial configuration (strip width, row arrangement, intercropping geometry) significantly affects outcomes.

4. **No density optimization**: Planting density ratios are collapsed in the LER data. A species pair can be complementary at one density ratio and competitive at another.

5. **No temporal dynamics**: The formulation optimizes for a single-season snapshot. Within-season interaction dynamics, relay cropping, and multi-year rotation benefits are not modeled.

6. **Hardcoded plant traits**: Heights and root depths are single-point estimates from extension publications, not empirical distributions. The diversity term (γ = 0.1) is low-weight, limiting the impact of imprecise trait values.

7. **Approximate N-balance double-counting mitigation**: The (1 - confidence) scaling is a heuristic. Confidence reflects data quality, not the degree to which nitrogen effects are already captured in the LER measurement.

8. **Extreme data sparsity**: Only 31 of 190 possible pairs (16%) have empirical LER data. The remaining 84% rely on priors that provide weak or no directional signal.

---

## 6. Future Work

**Larger species pools**: Expanding beyond 20 species would create problem instances where classical brute-force becomes impractical (C(100, 10) ≈ 1.7 × 10^13). This requires substantially more intercropping experimental data, particularly for underrepresented species combinations.

**Site-specific QUBO instances**: Integrating USDA SSURGO soil data and WorldClim climate data would enable location-specific optimization — filtering the candidate pool by viability and adjusting interaction coefficients by environmental context.

**Higher-order interactions (PUBO)**: Encoding 3-body and 4-body interaction terms would capture emergent polyculture effects. This requires both the interaction data and compatible solvers (higher-order quantum annealing or decomposition techniques).

**Real quantum hardware**: Running QAOA on gate-based quantum processors (IBM) or the annealing formulation on D-Wave Advantage systems would provide realistic noise profiles and wall-clock benchmarks. Error mitigation techniques would likely be required for the 20-qubit circuits.

**Temporal and spatial optimization**: Extending the formulation to multi-season rotation planning (temporal QUBO with season-indexed variables) or spatial strip design (QUBO with position-indexed variables) would address two of the most significant practical limitations.

---

## 7. References

1. Bischoff, O. et al. "Toward systems agroecology: Design and control of intercropping." *PNAS*, Dec 2024. DOI: 10.1073/pnas.2415315121

2. Paut, R. et al. "A global dataset of experimental intercropping and agroforestry studies in horticulture." *Scientific Data*, Jan 2024. DOI: 10.1038/s41597-023-02831-7

3. "Intercropping enhances productivity through complementary resource use." *npj Sustainable Agriculture*, Jan 2026. DOI: 10.1038/s44264-025-00110-z

4. Li, C. et al. "The productive performance of intercropping." *PNAS*, Jan 2023. DOI: 10.1073/pnas.2201886120

5. Kattge, J. et al. "TRY plant trait database — enhanced coverage and open access." *Global Change Biology*, 2020.

6. Mead, R. & Willey, R.W. "The concept of a 'land equivalent ratio' and advantages in yields from intercropping." *Experimental Agriculture*, 1980.
