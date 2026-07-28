"""Offset-invariant solution quality metrics for constrained QUBO problems.

The naive approximation ratio E_alg / E_opt is not a valid quality measure
here. Expanding the cardinality penalty λ(Σx − k)² produces a constant λk²
that `build_qubo_matrix` drops, since a constant cannot change which solution
is optimal. Every feasible solution therefore sits near −k²λ, and the ratio
mostly measures that shared constant rather than solution quality.

Concretely, at k=4 the penalty offset is −68.99 out of an optimum of −71.55:
96.4% of the magnitude is common to every feasible solution. The ratio's floor
over the whole feasible set is 0.974, so even the worst possible answer scores
97%. At k=3 the metric rates a solution in the bottom 6% of the feasible set
at 0.967.

The problem is invariant under adding a constant to the energy; a ratio is not.
This module provides metrics that are.

Primary metric — β (quality relative to a random feasible solution):

    β = (E_random − E_alg) / (E_random − E_opt)

    β = 1   solution is optimal
    β = 0   no better than drawing a feasible solution at random
    β < 0   worse than a random draw

β is invariant under E → E + c because every term is an energy difference.

Secondary metric — rank among feasible solutions (1 = optimal). Fully
invariant and needs no baseline, but requires enumeration, so it is only
available at small N. `percentile_better` is its scalable estimate.

The random baseline is estimated by uniform sampling from the feasible set
rather than exhaustive enumeration, so the metric remains computable at
problem sizes where enumeration is impossible.
"""

from dataclasses import dataclass
from math import comb

import numpy as np

DEFAULT_BASELINE_SAMPLES = 100_000


@dataclass
class RandomBaseline:
    """Sampled estimate of the mean energy over uniformly random feasible solutions.

    Attributes:
        mean_energy: Estimated E[f(x)] for x drawn uniformly from the feasible set.
        std_energy: Sample standard deviation of feasible energies.
        stderr: Standard error of mean_energy (std_energy / sqrt(n_samples)).
        min_energy: Best energy seen while sampling (a weak optimum estimate).
        max_energy: Worst energy seen while sampling.
        n_samples: Number of feasible solutions sampled.
        exhaustive: True if the "sample" was the complete feasible set.
    """

    mean_energy: float
    std_energy: float
    stderr: float
    min_energy: float
    max_energy: float
    n_samples: int
    exhaustive: bool = False


@dataclass
class SolutionQuality:
    """Quality of one solution, measured against a random-feasible baseline.

    Attributes:
        energy: Energy of the solution being scored.
        reference_energy: Best known energy (exact optimum where available).
        beta: Random-normalized quality. 1 = optimal, 0 = random, < 0 = worse.
        beta_stderr: Standard error of beta, propagated from the baseline.
        rank: Exact rank among feasible solutions (1 = best). None without
            full enumeration.
        n_feasible: Total feasible solutions, when known.
        percentile_better: Estimated fraction of feasible solutions strictly
            better than this one. Scalable stand-in for rank.
        total_energy_ratio: The legacy E_alg / E_opt metric, retained only so
            it can be reported alongside its floor.
        total_energy_ratio_floor: Value the legacy metric assigns to the worst
            feasible solution. The gap to 1.0 is the metric's usable range.
        objective_ratio: Ratio after subtracting the penalty offset. Removes
            the dominant constant but still assumes a meaningful zero.
        baseline: The random baseline used.
    """

    energy: float
    reference_energy: float
    beta: float
    beta_stderr: float
    percentile_better: float
    total_energy_ratio: float
    baseline: RandomBaseline
    rank: int | None = None
    n_feasible: int | None = None
    total_energy_ratio_floor: float | None = None
    objective_ratio: float | None = None


def sample_feasible_energies(
    q: np.ndarray,
    target_species: int,
    n_samples: int = DEFAULT_BASELINE_SAMPLES,
    seed: int | None = 42,
) -> np.ndarray:
    """Sample uniformly from the feasible set and return the energies.

    Draws `n_samples` subsets of exactly `target_species` variables, each
    uniformly at random without replacement, and evaluates the QUBO energy
    of each. Fully vectorized — a Python loop over 100k samples would be
    far too slow to run by default.

    Args:
        q: Upper-triangular QUBO matrix (N x N).
        target_species: Cardinality k of the feasible set.
        n_samples: Number of feasible solutions to draw.
        seed: RNG seed. None for nondeterministic sampling.

    Returns:
        Array of `n_samples` energies.
    """
    n = q.shape[0]
    k = target_species
    if not 0 < k <= n:
        raise ValueError(f"target_species must be in (0, {n}], got {k}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.default_rng(seed)

    # Uniform k-subsets: rank random keys per row and keep the k smallest.
    # Sorting the chosen indices keeps every pair (a, b) with a < b, which is
    # required because q is upper triangular.
    idx = np.argsort(rng.random((n_samples, n)), axis=1)[:, :k]
    idx.sort(axis=1)

    # Diagonal contribution.
    energies = q[idx, idx].sum(axis=1)

    # Pairwise contribution: one vectorized gather per (a, b) position pair.
    for a in range(k):
        for b in range(a + 1, k):
            energies += q[idx[:, a], idx[:, b]]

    return energies


def random_baseline(
    q: np.ndarray,
    target_species: int,
    n_samples: int = DEFAULT_BASELINE_SAMPLES,
    seed: int | None = 42,
    method: str = "auto",
) -> RandomBaseline:
    """Estimate the mean energy of a uniformly random feasible solution.

    Args:
        method: "sample" always draws `n_samples` feasible solutions — this is
            the path that remains available at problem sizes where the feasible
            set cannot be enumerated. "exhaustive" enumerates the feasible set.
            "auto" (default) enumerates when C(n, k) <= n_samples, since that
            is both cheaper and exact, and samples otherwise.

    Reporting note: at N=20 "auto" resolves to exhaustive, so the baseline
    carries no sampling error. Run with method="sample" to confirm the
    estimator agrees before relying on it at larger N.
    """
    n = q.shape[0]
    n_feasible = comb(n, target_species)

    if method not in ("auto", "sample", "exhaustive"):
        raise ValueError(f"method must be auto/sample/exhaustive, got {method!r}")

    use_exhaustive = method == "exhaustive" or (
        method == "auto" and n_feasible <= n_samples
    )

    if use_exhaustive:
        energies = _enumerate_feasible_energies(q, target_species)
        return RandomBaseline(
            mean_energy=float(energies.mean()),
            std_energy=float(energies.std(ddof=1)) if len(energies) > 1 else 0.0,
            stderr=0.0,  # No sampling error: this is the whole population.
            min_energy=float(energies.min()),
            max_energy=float(energies.max()),
            n_samples=int(len(energies)),
            exhaustive=True,
        )

    energies = sample_feasible_energies(q, target_species, n_samples, seed)
    std = float(energies.std(ddof=1))
    return RandomBaseline(
        mean_energy=float(energies.mean()),
        std_energy=std,
        stderr=std / np.sqrt(len(energies)),
        min_energy=float(energies.min()),
        max_energy=float(energies.max()),
        n_samples=int(len(energies)),
        exhaustive=False,
    )


def _enumerate_feasible_energies(q: np.ndarray, target_species: int) -> np.ndarray:
    """Energies of every feasible solution. Only use when C(n, k) is small."""
    import itertools

    n = q.shape[0]
    out = []
    for combo in itertools.combinations(range(n), target_species):
        idx = np.array(combo)
        e = q[idx, idx].sum()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                e += q[idx[a], idx[b]]
        out.append(e)
    return np.array(out)


def beta_quality(energy: float, reference_energy: float, baseline_mean: float) -> float:
    """Random-normalized quality β = (E_random − E_alg) / (E_random − E_opt).

    Returns NaN if the baseline coincides with the reference (a degenerate
    landscape where every feasible solution is optimal).
    """
    denom = baseline_mean - reference_energy
    if abs(denom) < 1e-12:
        return float("nan")
    return (baseline_mean - energy) / denom


def evaluate_quality(
    q: np.ndarray,
    target_species: int,
    energy: float,
    reference_energy: float,
    baseline: RandomBaseline | None = None,
    all_feasible_energies: np.ndarray | None = None,
    penalty_offset: float | None = None,
    n_samples: int = DEFAULT_BASELINE_SAMPLES,
    seed: int | None = 42,
) -> SolutionQuality:
    """Score a solution with offset-invariant metrics.

    Args:
        q: Upper-triangular QUBO matrix.
        target_species: Cardinality constraint k.
        energy: Energy of the solution being scored.
        reference_energy: Best known energy (exact optimum where available).
        baseline: Precomputed random baseline. Sampled if omitted — pass one in
            when scoring several solvers on the same instance so they share it.
        all_feasible_energies: Full enumeration, when available. Enables exact
            rank and an exact `percentile_better`.
        penalty_offset: Constant penalty energy −k²λ. Enables `objective_ratio`.
        n_samples: Samples for the baseline when one is not supplied.
        seed: RNG seed for baseline sampling.
    """
    if baseline is None:
        baseline = random_baseline(q, target_species, n_samples, seed)

    beta = beta_quality(energy, reference_energy, baseline.mean_energy)

    # Propagate baseline uncertainty: d(beta)/d(mean) = (E - ref) / (mean - ref)^2
    denom = baseline.mean_energy - reference_energy
    if abs(denom) < 1e-12 or baseline.stderr == 0.0:
        beta_stderr = 0.0
    else:
        beta_stderr = abs(energy - reference_energy) / denom**2 * baseline.stderr

    rank: int | None = None
    n_feasible: int | None = None
    ratio_floor: float | None = None

    if all_feasible_energies is not None and len(all_feasible_energies) > 0:
        srt = np.sort(all_feasible_energies)
        rank = int(np.searchsorted(srt, energy - 1e-9) + 1)
        n_feasible = int(len(srt))
        percentile_better = float(np.sum(srt < energy - 1e-9) / n_feasible)
        if reference_energy != 0:
            ratio_floor = float(srt[-1] / reference_energy)
    else:
        # Scalable fallback: estimate from the baseline sample.
        percentile_better = _estimate_percentile_better(
            q, target_species, energy, baseline, n_samples, seed
        )
        if reference_energy != 0:
            ratio_floor = float(baseline.max_energy / reference_energy)

    total_ratio = energy / reference_energy if reference_energy != 0 else float("nan")

    objective_ratio: float | None = None
    if penalty_offset is not None:
        obj_ref = reference_energy - penalty_offset
        if abs(obj_ref) > 1e-12:
            objective_ratio = (energy - penalty_offset) / obj_ref

    return SolutionQuality(
        energy=energy,
        reference_energy=reference_energy,
        beta=beta,
        beta_stderr=beta_stderr,
        rank=rank,
        n_feasible=n_feasible,
        percentile_better=percentile_better,
        total_energy_ratio=total_ratio,
        total_energy_ratio_floor=ratio_floor,
        objective_ratio=objective_ratio,
        baseline=baseline,
    )


def _estimate_percentile_better(
    q: np.ndarray,
    target_species: int,
    energy: float,
    baseline: RandomBaseline,
    n_samples: int,
    seed: int | None,
) -> float:
    """Fraction of feasible solutions strictly better than `energy`, by sampling."""
    energies = sample_feasible_energies(q, target_species, n_samples, seed)
    return float(np.sum(energies < energy - 1e-9) / len(energies))


def format_quality(quality: SolutionQuality, label: str = "") -> str:
    """One-line human-readable summary for benchmark output."""
    q = quality
    parts = [f"β={q.beta:+.4f}"]
    if q.beta_stderr > 0:
        parts[0] += f"±{q.beta_stderr:.4f}"
    if q.rank is not None:
        parts.append(f"rank {q.rank}/{q.n_feasible}")
    else:
        parts.append(f"~top {100 * q.percentile_better:.2f}%")
    if q.objective_ratio is not None:
        parts.append(f"obj-ratio {q.objective_ratio:.4f}")
    parts.append(f"total-ratio {q.total_energy_ratio:.4f}")
    if q.total_energy_ratio_floor is not None:
        parts[-1] += f" (floor {q.total_energy_ratio_floor:.4f})"
    prefix = f"{label}: " if label else ""
    return prefix + "  ".join(parts)
