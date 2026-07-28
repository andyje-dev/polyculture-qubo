# Quantum-Computational Optimization of Polyculture Species Selection: A QUBO Formulation Grounded in Intercropping Data

## Abstract

Selecting complementary crop species for polyculture systems is a combinatorial optimization problem that grows intractable as candidate pools expand. We present the first Quadratic Unconstrained Binary Optimization (QUBO) formulation for polyculture species selection grounded in empirical intercropping data. Drawing on 2,231 experimental observations from the Paut et al. horticulture dataset, 274 species pairs from Bischoff et al., and a companion planting network, we construct a composite objective encoding Land Equivalent Ratio (LER), nitrogen fixation complementarity, and functional diversity for a pool of 20 candidate species. We solve the resulting QUBO using exact enumeration, simulated annealing, and the Quantum Approximate Optimization Algorithm (QAOA). Exact enumeration and simulated annealing agree on agronomically plausible cereal-legume polycultures at every target size; QAOA reaches the same optimum in 3 of 9 depth-by-target configurations but is unreliable, in one case returning a solution worse than a random feasible draw. Energy landscape analysis reveals that while the total energy surface appears flat (100% of feasible solutions within 5% of optimum), this is an artifact of penalty term dominance — the objective-only landscape shows meaningful differentiation (0.04% within 5%). The same artifact invalidates the conventional approximation ratio, whose floor over the entire feasible set is 0.974 at k=4; we therefore report offset-invariant quality metrics throughout. At the current scale of N=20 species, no quantum advantage exists.

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

Positive J_ij indicates complementarity (LER > 1.0); negative indicates competition. The multiplicative combination of the three factors aggressively discounts sparse data: a pair with 10 observations, σ_LER = 0.5, from 1 article receives only 13% of its raw coefficient value (0.447 × 0.667 × 0.447 = 0.133). This is by design — sparse pairs defer to priors rather than contributing noisy signal — but it means only the most well-replicated pairs drive the optimization.

**Nitrogen balance (β = 0.2)**: A pairwise bonus for complementary nitrogen fixation: N_ij = 1.0 for fixer + non-fixer pairs, 0.2 for fixer + fixer (a heuristic reflecting diminishing returns on nitrogen when both species fix — the specific value lacks quantitative basis but is exercised by the β weight sweep), and 0.0 for non-fixer + non-fixer. To avoid double-counting nitrogen benefits already captured in LER data, N_ij is scaled by (1 - confidence_ij) — pairs with strong empirical LER evidence have their explicit nitrogen term reduced.

**Functional diversity (γ = 0.1)**: A trait-based pairwise bonus computed from root depth difference (30%), height difference (30%), and functional group difference (40%). This mild term encourages structural complementarity.

### 2.4 QUBO Construction

The optimization problem is formulated as minimizing x^T Q x over binary variables x_i ∈ {0, 1}, where x_i = 1 indicates species i is selected.

**Off-diagonal entries** (i < j):

$$Q_{ij} = -\alpha \cdot J_{ij} - \beta \cdot N_{ij} \cdot (1 - c_{ij}) - \gamma \cdot D_{ij} + 2\lambda$$

**Diagonal entries**:

$$Q_{ii} = -\alpha \cdot h_i + \lambda(1 - 2k)$$

where h_i is a linear bias encoding economic value (0.3 for cash crops, 0.09 for cover crops), k is the target species count, and λ is the penalty strength enforcing the constraint Σ x_i = k.

**Layered prior system**: For the 159 species pairs (84%) lacking direct LER observations, interaction coefficients are filled by a 3-layer prior system applied in order of increasing confidence:

1. Unknown pair prior: J = -0.05, confidence = 0 (a conservative default reflecting that arbitrary species combinations are more likely to compete than complement; the specific value was not empirically calibrated but is bounded in impact by the fact that LER-derived coefficients override it for all 31 observed pairs)
2. Bischoff studied-pair prior: J = -0.01, confidence = 0.05–0.15
3. Companion planting prior: J = ±0.1, confidence = 0.1
4. LER-derived coefficients override all priors for the 31 observed pairs

**Auto-derived penalty strength**: λ is computed as 1.5× the maximum objective benefit achievable by a single variable flip, guaranteeing that no constraint-violating solution can have lower energy than any feasible solution.

### 2.5 Solvers

**Exact enumeration**: All C(N, k) feasible solutions are evaluated exhaustively. For N=20 and k=4, this is 4,845 solutions. Provides ground truth and full energy landscape data.

**Simulated annealing**: Uses constraint-preserving swap moves — each step deselects one species and selects another, maintaining exactly k species throughout. Geometric cooling schedule (β: 0.1 → 10.0) over 1,000 sweeps with 100 random restarts.

**QAOA**: Implemented using Qiskit with the Aer statevector simulator. The QUBO is normalized (entries scaled to [-1, 1]) and converted to an Ising Hamiltonian via the substitution x_i = (1 - s_i)/2. The circuit alternates cost unitaries (RZZ gates for couplings, RZ gates for local fields) and mixer unitaries (RX gates). Variational parameters (γ, β) are optimized using COBYLA with 5 random restarts and 200 iterations each. Final solutions are sampled with 4,096 measurement shots. Notably, the mixer is a standard transverse-field (Σ X_i), not a constraint-preserving mixer. Unlike the simulated annealing solver, which enforces the cardinality constraint via swap moves, QAOA explores the full 2^N Hilbert space and relies on the penalty term λ to suppress infeasible states. At larger N, a constraint-preserving mixer (e.g., XY ring mixer or Grover-style mixer) would avoid wasting circuit depth learning the constraint structure — this is identified as future work.

### 2.6 Solution Quality Metrics

The conventional approximation ratio E_alg / E_opt is not a valid quality measure for this problem, and reporting it would substantially overstate solver performance. Expanding the cardinality penalty λ(Σx − k)² produces a constant λk² that the QUBO construction drops, since a constant cannot change which solution is optimal. Every feasible solution therefore carries a shared offset of −k²λ, which at k=4 is −68.99 against an optimum of −71.55: **96.4% of the energy magnitude is common to every feasible solution**. The ratio compresses toward 1 accordingly. Its floor over the entire feasible set is 0.974 at k=4, so even the worst possible answer scores 97%.

The optimization problem is invariant under adding a constant to the energy; a ratio of energies is not. We therefore report two invariant metrics.

**β (primary)** measures quality relative to a uniformly random feasible solution:

$$\beta = \frac{E_{\text{random}} - E_{\text{alg}}}{E_{\text{random}} - E_{\text{opt}}}$$

β = 1 is optimal, β = 0 is no better than drawing a feasible solution at random, and β < 0 is worse than random. Every term is an energy difference, so β is unchanged by a constant shift. E_random is estimated by uniform sampling from the feasible set rather than exhaustive enumeration, so the metric remains computable at problem sizes where enumeration is impossible; at N=20 the sampled estimate agrees with the exhaustive mean to within one standard error at every k.

**Rank** is the position of the solution among all feasible solutions (1 = optimal), using competition ranking so that ties share a rank. It requires no baseline and cannot be misread, but requires enumeration and so is only available at small N.

Where the total-energy ratio appears below it is accompanied by its floor, so the reader can see how little of its range is usable.

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

At k=4:

| Solver | Energy | Time | β | Rank | Total ratio |
|--------|--------|------|-----|------|-------------|
| Exact | -71.553 | 0.008s | +1.000 | 1 / 4,845 | 1.000 |
| Simulated annealing | -71.553 | 2.5s | +1.000 | 1 / 4,845 | 1.000 |
| QAOA (p=1) | -71.402 | 28.9s | +0.842 | 5 / 4,845 | 0.998 |
| QAOA (p=2) | -71.410 | 44.0s | +0.849 | 4 / 4,845 | 0.998 |
| QAOA (p=3) | -71.553 | 66.7s | +1.000 | 1 / 4,845 | 1.000 |
| *(random feasible draw)* | -70.601 | — | 0.000 | — | 0.987 |
| *(worst feasible solution)* | -69.684 | — | -0.963 | 4,845 / 4,845 | 0.974 |

The last two rows are the reason the total-energy ratio must not be used: a random draw scores 0.987 on it and the worst possible answer scores 0.974. The entire feasible set spans 2.6 percentage points of that metric.

Simulated annealing matches the exact optimum on every instance and at every k. QAOA's behavior is far more variable than a single depth or a single k suggests:

| k | p=1 | p=2 | p=3 |
|---|-----|-----|-----|
| 3 | **-0.525** (1067 / 1,140) | **+1.000** (1 / 1,140) | +0.643 (12 / 1,140) |
| 4 | +0.842 (5 / 4,845) | +0.849 (4 / 4,845) | **+1.000** (1 / 4,845) |
| 5 | **+1.000** (1 / 15,504) | +0.775 (47 / 15,504) | +0.867 (12 / 15,504) |

*β (rank among the C(20,k) feasible solutions). Bold marks the best depth for each k.*

Three observations follow, none of which is visible through the total-energy ratio (which reports every one of these nine cells as ≥ 0.961):

1. **QAOA reaches the exact optimum in 3 of 9 configurations** — at (k=3, p=2), (k=4, p=3), and (k=5, p=1).
2. **At k=3, p=1 is worse than a random feasible draw** (β = −0.525, rank 1067 of 1,140). It returns barley, maize, and sorghum — three cereals, no nitrogen fixer, and not a single pair with empirical LER data. The total-energy ratio rates this same run at 0.967.
3. **Quality is not monotone in circuit depth.** It improves with depth at k=4, collapses then partially recovers at k=5, and swings from worse-than-random to optimal and back at k=3. This is consistent with the variational outer loop failing to converge reliably rather than with depth being insufficient (see §4).

**In-constraint probability.** The share of measurement shots landing in the feasible subspace (Hamming weight exactly k) is offset-invariant and measures how much constraint structure the circuit learned. It is the one figure that distinguishes QAOA from a uniform sampler:

| k | uniform C(20,k)/2²⁰ | p=1 | p=2 | p=3 |
|---|---------------------|-----|-----|-----|
| 3 | 0.109% | 0.05% (0.4×) | 39.75% (366×) | 17.75% (163×) |
| 4 | 0.462% | 31.27% (68×) | 36.72% (79×) | 26.93% (58×) |
| 5 | 1.479% | 34.67% (23×) | 8.50% (5.7×) | 41.53% (28×) |

Eight of the nine configurations concentrate probability in the feasible subspace by 5.7× to 366× over uniform sampling, with a standard transverse-field mixer and no constraint-preserving machinery. The exception is (k=3, p=1), which is *below* uniform at 0.4× — it actively avoids the feasible subspace, which explains its worse-than-random solution.

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

The solver's selected pairs rank in the top 27th percentile of all observed LER values (mean selected LER = 1.69 vs. mean observed LER = 1.37). A permutation test (10,000 random draws of 5 pairs from the 31 observed) confirms this ranking is statistically significant, establishing that the solver's selections are not an artifact of chance. Individual pair rankings:

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

Systematically removing each of the 31 observed pairs from the interaction matrix, only 2 pairs change the optimal solution when removed under the default weight configuration. The 5 strongest interaction coefficients (J > 0.25) all produce stable solutions when masked individually. Cross-weight masking analysis (testing each pair's removal across 5 representative weight configurations) reveals that the set of load-bearing pairs is weight-dependent — pairs critical under one weight configuration may be stable under others, and vice versa.

---

## 4. Discussion

### Penalty Dominance and Formulation Design

The most striking finding is the penalty decomposition: the constraint enforcement term contributes ~97% of total energy for feasible solutions, compressing the objective signal into a narrow band. This is not a flaw in the formulation — it is an inherent property of QUBO constraint encoding. The auto-derived λ must be large enough to guarantee that no infeasible solution can beat any feasible one, and this requirement pushes λ well above the objective scale.

For classical solvers, this is irrelevant — simulated annealing with constraint-preserving moves ignores the penalty entirely, exploring only the feasible subspace. For QAOA, however, the penalty dominance means the cost Hamiltonian's spectral structure is shaped primarily by the constraint, not the objective. QAOA must first "learn" the constraint structure before it can differentiate among feasible solutions.

The in-constraint probabilities in §3.2 show this happening: eight of nine configurations concentrate 5.7× to 366× more probability in the feasible subspace than uniform sampling, so the circuit does learn the constraint. What it does not reliably do is discriminate *within* that subspace, which is where the 2.7% of energy carrying agronomic signal lives. The depth sweep does not support the intuition that deeper circuits monotonically recover this: quality improves with depth at k=4 but degrades at k=5 and is non-monotone at k=3.

Penalty dominance also has a measurement consequence that is easy to miss and that we initially made ourselves. Any quality metric shaped like E_alg / E_opt inherits the constant offset and compresses toward 1, so it will report a solver as near-optimal almost regardless of what it found. This is a general property of penalty-encoded constrained QUBOs, not a quirk of this instance, and it argues for offset-invariant reporting (§2.6) in any such benchmark.

### Data Sparsity

With only 16% of species pairs having empirical LER data, the solver is largely navigating a prior-dominated landscape. The 31 observed pairs create data-rich islands that anchor the solution — all optimal species appear in at least one well-observed pair. This is simultaneously a limitation (we cannot discover novel polycultures beyond the data) and a feature (the formulation correctly defers to empirical evidence where available).

The data masking analysis quantifies this: removing any single observed pair changes the solution in at most 2 of 31 cases. The formulation is resilient to single-pair data loss, but this resilience comes from redundancy in the data-rich subset, not from the priors providing meaningful signal.

### Quantum Advantage Assessment

At N=20, no quantum advantage exists or should be claimed. Exact enumeration solves in 9ms. Simulated annealing matches the optimum with 100% reliability at every k. QAOA reaches the exact optimum in 3 of 9 (k, p) configurations but requires 29–67 seconds — three to four orders of magnitude slower than brute force — and its quality is not reliable: at (k=3, p=1) it returns a solution worse than a random feasible draw.

The instability is more informative than the average quality. QAOA's variational objective is estimated from 4,096 measurement shots and handed to COBYLA, a deterministic trust-region method that assumes a smooth, noise-free objective. The resulting shot noise on the objective is of the same order as the entire objective range being optimized over — the constant penalty offset means the agronomic signal occupies only 2.7% of the total energy at k=4, while the sampling noise does not shrink correspondingly. The non-monotonicity in depth is therefore better explained by outer-loop convergence failure than by insufficient circuit expressiveness, and deeper circuits alone should not be expected to fix it. Using an exact statevector expectation (cheap at N=20) or a stochastic-objective optimizer such as SPSA would isolate this.

However, the energy landscape characterization provides suggestive (though not conclusive) projections:

1. The spectral gap shrinks with k (0.051 → 0.018), suggesting that problem hardness increases as the target subset grows relative to the pool. However, this trend is observed across only three data points (k = 3, 4, 5) and is non-monotonic (the gap at k=4 is 0.095, larger than at k=3). Reliable scaling projections require problem instances at N = 50–100.
2. The near-zero fraction of local minima (0.02%) means the landscape has almost no trapping structure — a property that may change at larger N where the prior landscape contributes more complex structure.
3. The penalty-to-objective ratio worsens with k (objective range drops from 4.1% to 1.9% of penalty), suggesting QAOA parameter optimization becomes harder at larger scales. The QAOA's use of a standard transverse-field mixer (rather than a constraint-preserving mixer) compounds this issue, as circuit depth is partially spent learning the constraint structure rather than differentiating among feasible solutions.

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

9. **QAOA mixer is unconstrained**: The standard transverse-field mixer explores the full 2^N Hilbert space rather than the feasible subspace of exactly-k-selected states. Constraint enforcement depends entirely on the penalty strength λ, and at larger N the penalty-to-objective ratio worsens, potentially reducing QAOA effectiveness. A constraint-preserving mixer (XY ring or Grover-style) would address this but is not implemented.

10. **Diversity normalization constants are pool-specific**: The trait difference normalization (max_height_diff = 2.0m, max_root_diff = 1.0m) is calibrated to the extremes of the current 20-species pool. Adding species taller than maize or deeper-rooted than sorghum would saturate these bonuses. The low weight (γ = 0.1) limits practical impact but constrains extensibility.

11. **Scalability projections rest on 3 data points**: The spectral gap trend across k = {3, 4, 5} is non-monotonic (0.051 → 0.095 → 0.018), making extrapolation unreliable. Credible scaling projections would require problem instances at N = 50–100 with proportionally richer interaction data.

12. **Validation lacks statistical rigor in places**: The leave-one-out cross-validation runs 31 tests without multiple-comparison correction (1–2 false positives expected by chance at 95% threshold). The pairwise ranking check now includes a permutation significance test, but earlier results should be interpreted with this caveat.

---

## 6. Future Work

**Larger species pools**: Expanding beyond 20 species would create problem instances where classical brute-force becomes impractical (C(100, 10) ≈ 1.7 × 10^13). This requires substantially more intercropping experimental data, particularly for underrepresented species combinations.

**Site-specific QUBO instances**: Integrating USDA SSURGO soil data and WorldClim climate data would enable location-specific optimization — filtering the candidate pool by viability and adjusting interaction coefficients by environmental context.

**Higher-order interactions (PUBO)**: Encoding 3-body and 4-body interaction terms would capture emergent polyculture effects. This requires both the interaction data and compatible solvers (higher-order quantum annealing or decomposition techniques).

**Real quantum hardware**: Running QAOA on gate-based quantum processors (IBM) or the annealing formulation on D-Wave Advantage systems would provide realistic noise profiles and wall-clock benchmarks. Error mitigation techniques would likely be required for the 20-qubit circuits.

**Constraint-preserving QAOA mixer**: Replacing the standard transverse-field mixer with an XY ring mixer or Grover-style mixer that preserves the Hamming weight constraint (exactly k species selected) would confine QAOA's search to the feasible subspace, eliminating penalty dominance issues. The in-constraint probabilities in §3.2 bound the expected gain: the current mixer already reaches 27–42% feasible sampling at k=4 and k=5, so a constraint-preserving mixer would recover at most a factor of ~3 in effective shot count, and the harder problem — discriminating within the feasible subspace — would remain.

**Noise-aware variational optimization**: The QAOA outer loop currently passes a 4,096-shot sampled expectation to COBYLA, which assumes a deterministic objective. Substituting the exact statevector expectation (tractable at N=20) or a stochastic optimizer such as SPSA would establish whether the non-monotone depth behavior in §3.2 is a convergence artifact or a genuine property of the ansatz.

**Temporal and spatial optimization**: Extending the formulation to multi-season rotation planning (temporal QUBO with season-indexed variables) or spatial strip design (QUBO with position-indexed variables) would address two of the most significant practical limitations.

---

## 7. References

1. Bischoff, O. et al. "Toward systems agroecology: Design and control of intercropping." *PNAS*, Dec 2024. DOI: 10.1073/pnas.2415315121

2. Paut, R. et al. "A global dataset of experimental intercropping and agroforestry studies in horticulture." *Scientific Data*, Jan 2024. DOI: 10.1038/s41597-023-02831-7

3. "Intercropping enhances productivity through complementary resource use." *npj Sustainable Agriculture*, Jan 2026. DOI: 10.1038/s44264-025-00110-z

4. Li, C. et al. "The productive performance of intercropping." *PNAS*, Jan 2023. DOI: 10.1073/pnas.2201886120

5. Kattge, J. et al. "TRY plant trait database — enhanced coverage and open access." *Global Change Biology*, 2020.

6. Mead, R. & Willey, R.W. "The concept of a 'land equivalent ratio' and advantages in yields from intercropping." *Experimental Agriculture*, 1980.
