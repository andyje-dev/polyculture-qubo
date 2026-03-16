"""Visualization for polyculture QUBO analysis.

Each function takes computed metrics and saves a publication-quality figure.
All plots use matplotlib with seaborn styling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from polyculture_qubo.analysis.landscape import LandscapeMetrics
from polyculture_qubo.analysis.sensitivity import MaskingResult, WeightSweepResult

# Default output directory
FIGURES_DIR = Path(__file__).parents[3] / "output" / "figures"


def _save(fig: plt.Figure, name: str, output_dir: Path | None = None) -> Path:
    """Save a figure and close it."""
    d = output_dir or FIGURES_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_energy_histogram(
    all_energies: np.ndarray,
    best_energy: float,
    k: int,
    output_dir: Path | None = None,
) -> Path:
    """Plot distribution of all feasible solution energies.

    Shows where the optimum sits relative to the full distribution.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(all_energies, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(best_energy, color="red", linewidth=2, linestyle="--",
               label=f"Optimum: {best_energy:.2f}")
    ax.set_xlabel("QUBO Energy")
    ax.set_ylabel("Count")
    ax.set_title(f"Energy Distribution — All C(20,{k}) = {len(all_energies)} Feasible Solutions")
    ax.legend()

    return _save(fig, f"energy_histogram_k{k}", output_dir)


def plot_objective_decomposition(
    all_energies: np.ndarray,
    penalty_offset: float,
    k: int,
    output_dir: Path | None = None,
) -> Path:
    """Plot total energy vs. objective-only energy side by side.

    Shows how the penalty term compresses the apparent energy landscape.
    The left panel shows total QUBO energy (dominated by penalty offset),
    the right panel shows objective-only energy (penalty subtracted),
    revealing the actual agronomic signal structure.
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    obj_energies = all_energies - penalty_offset

    # Left: total energy
    ax1.hist(all_energies, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax1.axvline(np.min(all_energies), color="red", linewidth=2, linestyle="--",
                label=f"Optimum: {np.min(all_energies):.2f}")
    ax1.set_xlabel("Total QUBO Energy")
    ax1.set_ylabel("Count")
    ax1.set_title("Total Energy (penalty + objective)")
    ax1.legend()

    # Right: objective only
    ax2.hist(obj_energies, bins=50, color="#e67e22", alpha=0.7, edgecolor="white")
    ax2.axvline(np.min(obj_energies), color="red", linewidth=2, linestyle="--",
                label=f"Optimum: {np.min(obj_energies):.2f}")
    ax2.set_xlabel("Objective Energy (penalty subtracted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Agronomic Signal Only")
    ax2.legend()

    total_range = np.max(all_energies) - np.min(all_energies)
    obj_range = np.max(obj_energies) - np.min(obj_energies)
    fig.suptitle(
        f"Energy Decomposition — k={k}\n"
        f"Total range: {total_range:.2f} | Objective range: {obj_range:.2f} | "
        f"Penalty offset: {penalty_offset:.2f}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    return _save(fig, f"objective_decomposition_k{k}", output_dir)


def plot_gap_structure(
    sorted_energies: np.ndarray,
    k: int,
    output_dir: Path | None = None,
) -> Path:
    """Plot the lowest energy levels to visualize spectral gap structure.

    Shows the energy separation between the ground state and excited states.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    n = len(sorted_energies)
    indices = np.arange(n)

    ax.plot(indices, sorted_energies, "o-", color="steelblue", markersize=4)
    ax.axhline(sorted_energies[0], color="red", linewidth=1, linestyle=":",
               alpha=0.5, label=f"Ground state: {sorted_energies[0]:.4f}")

    if n > 1:
        gap = sorted_energies[1] - sorted_energies[0]
        ax.annotate(
            f"Gap = {gap:.4f}",
            xy=(1, sorted_energies[1]),
            xytext=(3, sorted_energies[1] + 0.05),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red",
        )

    ax.set_xlabel("Energy Level Index")
    ax.set_ylabel("QUBO Energy")
    ax.set_title(f"Low-Energy Spectrum — k = {k} (bottom {n} levels)")
    ax.legend()

    return _save(fig, f"gap_structure_k{k}", output_dir)


def plot_solver_comparison(
    results: dict[str, dict],
    output_dir: Path | None = None,
) -> Path:
    """Plot solver comparison: energy and wall-clock time.

    Args:
        results: Dict mapping solver_name -> {"energy": float, "time": float}.
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(results.keys())
    energies = [results[n]["energy"] for n in names]
    times = [results[n]["time"] for n in names]

    # Energy comparison
    colors = ["#2ecc71" if e == min(energies) else "#3498db" for e in energies]
    ax1.barh(names, energies, color=colors, edgecolor="white")
    ax1.set_xlabel("QUBO Energy (lower = better)")
    ax1.set_title("Solution Quality")

    # Time comparison (log scale)
    ax2.barh(names, times, color="#e74c3c", edgecolor="white")
    ax2.set_xlabel("Wall-Clock Time (seconds)")
    ax2.set_xscale("log")
    ax2.set_title("Solver Runtime")

    fig.suptitle("Solver Benchmark Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return _save(fig, "solver_comparison", output_dir)


def plot_sensitivity_heatmap(
    sweep_results: list[WeightSweepResult],
    output_dir: Path | None = None,
) -> Path:
    """Heatmap showing species selection frequency across weight configs.

    X-axis: species, Y-axis: grouped by dominant weight.
    Cell color: selection frequency.
    """
    sns.set_style("whitegrid")

    # Compute species frequency per alpha bin
    all_species = sorted(set(sp for r in sweep_results for sp in r.selected_species))
    alpha_bins = np.arange(0.3, 1.05, 0.1)
    bin_labels = [f"α={a:.1f}" for a in alpha_bins[:-1]]

    freq_matrix = np.zeros((len(bin_labels), len(all_species)))
    bin_counts = np.zeros(len(bin_labels))

    for r in sweep_results:
        bin_idx = min(
            int((r.alpha - 0.3) / 0.1),
            len(bin_labels) - 1,
        )
        if bin_idx < 0:
            continue
        bin_counts[bin_idx] += 1
        for sp in r.selected_species:
            sp_idx = all_species.index(sp)
            freq_matrix[bin_idx, sp_idx] += 1

    # Normalize by bin counts
    for i in range(len(bin_labels)):
        if bin_counts[i] > 0:
            freq_matrix[i] /= bin_counts[i]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        freq_matrix,
        xticklabels=[sp.replace("_", " ").title() for sp in all_species],
        yticklabels=bin_labels,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        annot=True, fmt=".0%",
        ax=ax,
    )
    ax.set_title("Species Selection Frequency by LER Weight (α)")
    ax.set_xlabel("Species")
    ax.set_ylabel("Weight Configuration")
    plt.xticks(rotation=45, ha="right")

    return _save(fig, "sensitivity_heatmap", output_dir)


def plot_interaction_network(
    j_matrix: pd.DataFrame,
    selected_species: list[str],
    output_dir: Path | None = None,
) -> Path:
    """Plot species interaction network with optimal solution highlighted.

    Nodes = species, edges = interactions (green=complementary, red=competitive).
    Selected species are highlighted with larger nodes and bold borders.
    """
    import networkx as nx

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12, 10))

    G = nx.Graph()
    species = j_matrix.columns.tolist()

    for sp in species:
        G.add_node(sp)

    # Add edges for non-trivial interactions
    for i, sp_a in enumerate(species):
        for sp_b in species[i + 1:]:
            j_val = j_matrix.loc[sp_a, sp_b]
            if abs(j_val) > 0.02:  # Skip near-zero (prior-only) edges
                G.add_edge(sp_a, sp_b, weight=j_val)

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Node styling
    selected_set = set(selected_species)
    node_colors = ["#2ecc71" if n in selected_set else "#bdc3c7" for n in G.nodes()]
    node_sizes = [800 if n in selected_set else 300 for n in G.nodes()]
    node_edge_colors = ["#27ae60" if n in selected_set else "#95a5a6" for n in G.nodes()]
    node_linewidths = [3 if n in selected_set else 1 for n in G.nodes()]

    # Edge styling
    edge_colors = []
    edge_widths = []
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        if w > 0:
            edge_colors.append("#27ae60")
        else:
            edge_colors.append("#e74c3c")
        edge_widths.append(min(abs(w) * 10, 4))

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        edgecolors=node_edge_colors, linewidths=node_linewidths, ax=ax,
    )

    # Labels
    labels = {sp: sp.replace("_", " ").title() for sp in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

    ax.set_title("Species Interaction Network\n(green = complementary, red = competitive, "
                 "highlighted = selected)", fontsize=12)
    ax.axis("off")

    return _save(fig, "interaction_network", output_dir)


def plot_data_coverage(
    masking_results: list[MaskingResult],
    output_dir: Path | None = None,
) -> Path:
    """Plot the impact of removing each observed pair on the solution.

    Shows which pairs are most influential — removing them changes the
    optimal polyculture.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by absolute J value
    sorted_results = sorted(masking_results, key=lambda r: -abs(r.masked_j))

    pair_names = [f"{r.masked_pair[0]}+{r.masked_pair[1]}" for r in sorted_results]
    j_values = [r.masked_j for r in sorted_results]
    changed = [r.solution_changed for r in sorted_results]

    colors = ["#e74c3c" if c else "#3498db" for c in changed]

    bars = ax.barh(range(len(pair_names)), j_values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels([p.replace("_", " ") for p in pair_names], fontsize=7)
    ax.set_xlabel("Interaction Coefficient (J)")
    ax.set_title("Data Coverage Analysis\n(red = removal changes solution, blue = stable)")
    ax.invert_yaxis()

    # Add LER annotations
    for i, r in enumerate(sorted_results):
        ax.annotate(
            f"LER={r.masked_ler:.2f}",
            xy=(r.masked_j, i),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7, va="center",
        )

    return _save(fig, "data_coverage", output_dir)


def plot_scalability(
    k_values: list[int],
    spectral_gaps: list[float],
    degeneracies: list[int],
    solution_sizes: list[int],
    output_dir: Path | None = None,
) -> Path:
    """Plot scalability metrics across k values."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Spectral gap vs k
    axes[0].plot(k_values, spectral_gaps, "o-", color="steelblue", linewidth=2)
    axes[0].set_xlabel("Target Species Count (k)")
    axes[0].set_ylabel("Spectral Gap (E1 - E0)")
    axes[0].set_title("Gap Scaling")

    # Degeneracy vs k
    axes[1].plot(k_values, degeneracies, "s-", color="#e74c3c", linewidth=2)
    axes[1].set_xlabel("Target Species Count (k)")
    axes[1].set_ylabel("Near-Optimal Solutions (5%)")
    axes[1].set_title("Degeneracy Scaling")

    # Fraction near-optimal
    fractions = [d / s for d, s in zip(degeneracies, solution_sizes)]
    axes[2].plot(k_values, fractions, "^-", color="#2ecc71", linewidth=2)
    axes[2].set_xlabel("Target Species Count (k)")
    axes[2].set_ylabel("Fraction Within 5% of Optimum")
    axes[2].set_title("Landscape Flatness")

    fig.suptitle("Scalability Projection", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return _save(fig, "scalability", output_dir)
