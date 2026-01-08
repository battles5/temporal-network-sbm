"""
Visualization Module
====================

Creates static visualizations of network analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from graph_tool.all import Graph, graph_draw, sfdp_layout
from typing import Dict, List, Tuple, Set
import os


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_degree_distribution(
    degrees: np.ndarray,
    output_path: str,
    dpi: int = 300
):
    """
    Plot degree distribution (linear and log-log).
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear histogram
    ax1 = axes[0]
    ax1.hist(degrees, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Degree Distribution')
    ax1.axvline(np.mean(degrees), color='red', linestyle='--', 
                label=f'Mean: {np.mean(degrees):.1f}')
    ax1.legend()
    
    # Log-log plot
    ax2 = axes[1]
    unique, counts = np.unique(degrees, return_counts=True)
    # Filter out zeros for log plot
    mask = (unique > 0) & (counts > 0)
    ax2.scatter(unique[mask], counts[mask], color='steelblue', alpha=0.7, s=50)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree (log)')
    ax2.set_ylabel('Frequency (log)')
    ax2.set_title('Degree Distribution (Log-Log)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_centrality_comparison(
    node_centrality: dict,
    output_path: str,
    dpi: int = 300
):
    """
    Plot scatter comparisons between centrality measures.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    degree = node_centrality['degree']
    betweenness = node_centrality['betweenness']
    closeness = node_centrality['closeness']
    
    # Handle NaN in closeness
    closeness_clean = np.nan_to_num(closeness, nan=0)
    
    # Degree vs Betweenness
    axes[0].scatter(degree, betweenness, alpha=0.6, c='steelblue', s=40)
    axes[0].set_xlabel('Degree Centrality')
    axes[0].set_ylabel('Betweenness Centrality')
    axes[0].set_title('Degree vs Betweenness')
    
    # Degree vs Closeness
    axes[1].scatter(degree, closeness_clean, alpha=0.6, c='forestgreen', s=40)
    axes[1].set_xlabel('Degree Centrality')
    axes[1].set_ylabel('Closeness Centrality')
    axes[1].set_title('Degree vs Closeness')
    
    # Betweenness vs Closeness
    axes[2].scatter(betweenness, closeness_clean, alpha=0.6, c='coral', s=40)
    axes[2].set_xlabel('Betweenness Centrality')
    axes[2].set_ylabel('Closeness Centrality')
    axes[2].set_title('Betweenness vs Closeness')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_centrality_network(
    g: Graph,
    node_centrality: dict,
    output_path: str,
    output_size: Tuple[int, int] = (2000, 2000)
):
    """
    Plot network with node size proportional to centrality.
    """
    # Use betweenness for size, degree for color
    degree = node_centrality['degree']
    betweenness = node_centrality['betweenness']
    
    # Node size based on betweenness
    size = g.new_vertex_property("double")
    min_size, max_size = 5, 30
    bet_min, bet_max = betweenness.min(), betweenness.max()
    for v in g.vertices():
        if bet_max > bet_min:
            normalized = (betweenness[int(v)] - bet_min) / (bet_max - bet_min)
        else:
            normalized = 0.5
        size[v] = min_size + (max_size - min_size) * normalized
    
    # Node color based on degree
    vcolor = g.new_vertex_property("vector<double>")
    cmap = plt.cm.plasma
    deg_min, deg_max = degree.min(), degree.max()
    for v in g.vertices():
        if deg_max > deg_min:
            normalized = (degree[int(v)] - deg_min) / (deg_max - deg_min)
        else:
            normalized = 0.5
        rgba = cmap(normalized)
        vcolor[v] = list(rgba)
    
    # Layout
    pos = sfdp_layout(g, K=1.5)
    
    graph_draw(g, pos=pos,
               vertex_fill_color=vcolor,
               vertex_size=size,
               vertex_pen_width=0.3,
               edge_color=[0.5, 0.5, 0.5, 0.3],
               edge_pen_width=0.5,
               output_size=output_size,
               output=output_path,
               bg_color=[1, 1, 1, 1])
    
    print(f"  Saved: {output_path}")


def plot_community_network(
    g: Graph,
    block_assignment: np.ndarray,
    output_path: str,
    output_size: Tuple[int, int] = (2000, 2000)
):
    """
    Plot network colored by SBM blocks.
    """
    n_blocks = len(np.unique(block_assignment))
    cmap = plt.cm.Set2 if n_blocks <= 8 else plt.cm.tab20
    
    vcolor = g.new_vertex_property("vector<double>")
    for v in g.vertices():
        block = block_assignment[int(v)]
        rgba = cmap(block / max(n_blocks - 1, 1))
        vcolor[v] = list(rgba)
    
    pos = sfdp_layout(g, K=1.5)
    
    graph_draw(g, pos=pos,
               vertex_fill_color=vcolor,
               vertex_size=12,
               vertex_pen_width=0.5,
               edge_color=[0.4, 0.4, 0.4, 0.4],
               edge_pen_width=0.5,
               output_size=output_size,
               output=output_path,
               bg_color=[1, 1, 1, 1])
    
    print(f"  Saved: {output_path}")


def plot_sbm_matrix(
    block_assignment: np.ndarray,
    connection_probability_matrix: np.ndarray,
    output_path: str,
    dpi: int = 300
):
    """
    Plot SBM connection probability matrix.
    """
    setup_style()
    
    n_blocks = connection_probability_matrix.shape[0]
    
    # Adjust figure size based on number of blocks
    fig_size = max(6, min(12, n_blocks * 0.5))
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
    
    im = ax.imshow(connection_probability_matrix, cmap='Blues', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r'$\hat{\pi}_{ql}$', fontsize=11)
    
    # Add labels - just numbers, no "Block" prefix
    ax.set_xticks(range(n_blocks))
    ax.set_yticks(range(n_blocks))
    ax.set_xticklabels(range(n_blocks), fontsize=8)
    ax.set_yticklabels(range(n_blocks), fontsize=8)
    ax.set_xlabel('Block $\\ell$', fontsize=10)
    ax.set_ylabel('Block $q$', fontsize=10)
    ax.set_title('Connection Probability Matrix', fontsize=12)
    
    # Remove grid lines
    ax.grid(False)
    
    # Add values as text - smaller font, 1 decimal for readability
    fontsize = max(5, min(8, 120 // n_blocks))  # Adaptive font size
    for i in range(n_blocks):
        for j in range(n_blocks):
            val = connection_probability_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            # Use 1 decimal for cleaner look, or 2 if value is small
            fmt = f'{val:.1f}' if val >= 0.1 or val == 0 else f'{val:.2f}'
            ax.text(j, i, fmt, ha='center', va='center', color=color, fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_temporal_activity(
    timeline: List[dict],
    output_path: str,
    dpi: int = 300
):
    """
    Plot temporal activity (edges and nodes over time).
    """
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    times = [t['time_minutes'] for t in timeline]
    edges = [t['n_edges'] for t in timeline]
    nodes = [t['n_active_nodes'] for t in timeline]
    
    # Edges over time
    axes[0].fill_between(times, edges, alpha=0.4, color='steelblue')
    axes[0].plot(times, edges, color='steelblue', linewidth=1.5)
    axes[0].set_ylabel('Number of Edges')
    axes[0].set_title('Network Activity Over Time')
    
    # Peak annotation
    peak_idx = np.argmax(edges)
    axes[0].annotate(f'Peak: {edges[peak_idx]}',
                     xy=(times[peak_idx], edges[peak_idx]),
                     xytext=(times[peak_idx] + 5, edges[peak_idx] * 0.9),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     color='red')
    
    # Nodes over time
    axes[1].fill_between(times, nodes, alpha=0.4, color='forestgreen')
    axes[1].plot(times, nodes, color='forestgreen', linewidth=1.5)
    axes[1].set_xlabel('Time (minutes)')
    axes[1].set_ylabel('Active Nodes')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_dynamic_sbm_transitions(
    transition_matrix: np.ndarray,
    block_labels: List[int],
    output_path: str,
    dpi: int = 300
):
    """
    Plot Dynamic SBM transition probability heatmap.
    """
    setup_style()
    
    n_blocks = len(block_labels)
    
    # Adjust figure size based on number of blocks
    fig_size = max(6, min(12, n_blocks * 0.6))
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
    
    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$P(b_{t+1} | b_t)$', fontsize=11)
    
    # Just numbers on axes
    ax.set_xticks(range(n_blocks))
    ax.set_yticks(range(n_blocks))
    ax.set_xticklabels(block_labels, fontsize=8)
    ax.set_yticklabels(block_labels, fontsize=8)
    ax.set_xlabel('Block at $t+1$', fontsize=10)
    ax.set_ylabel('Block at $t$', fontsize=10)
    ax.set_title('Block Transition Probabilities', fontsize=12)
    
    # Remove grid
    ax.grid(False)
    
    # Add values - smaller font, 1 decimal
    fontsize = max(5, min(8, 100 // n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            val = transition_matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            fmt = f'{val:.1f}' if val >= 0.1 or val == 0 else f'{val:.2f}'
            ax.text(j, i, fmt, ha='center', va='center', color=color, fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_dynamic_sbm_evolution(
    window_block_info: List[dict],
    output_path: str,
    dpi: int = 300
):
    """
    Plot evolution of block sizes over time.
    """
    setup_style()
    
    # Collect all blocks
    all_blocks = set()
    for winfo in window_block_info:
        all_blocks.update(winfo['block_sizes'].keys())
    all_blocks = sorted(all_blocks)
    
    # Build time series for each block
    n_windows = len(window_block_info)
    block_series = {b: np.zeros(n_windows) for b in all_blocks}
    
    for i, winfo in enumerate(window_block_info):
        for block, size in winfo['block_sizes'].items():
            block_series[block][i] = size
    
    # Plot stacked area
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cmap = plt.cm.Set2 if len(all_blocks) <= 8 else plt.cm.tab20
    colors = [cmap(i / max(len(all_blocks) - 1, 1)) for i in range(len(all_blocks))]
    
    x = range(n_windows)
    bottom = np.zeros(n_windows)
    
    for block, color in zip(all_blocks, colors):
        ax.fill_between(x, bottom, bottom + block_series[block], 
                        label=f'Block {block}', color=color, alpha=0.8)
        bottom += block_series[block]
    
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Number of Nodes')
    ax.set_title('Block Sizes Over Time (Dynamic SBM)')
    ax.legend(loc='upper right', ncol=min(4, len(all_blocks)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_all_visualizations(
    g: Graph,
    degrees: np.ndarray,
    node_centrality: dict,
    sbm_results: dict,
    dynamic_sbm_results: dict,
    timeline: List[dict],
    output_dir: str,
    dpi: int = 300
):
    """
    Generate all visualizations and save to output directory.
    """
    print("\nGenerating visualizations...")
    
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Degree distribution
    plot_degree_distribution(
        degrees,
        os.path.join(figures_dir, 'degree_distribution.png'),
        dpi=dpi
    )
    
    # Centrality comparison
    plot_centrality_comparison(
        node_centrality,
        os.path.join(figures_dir, 'centrality_comparison.png'),
        dpi=dpi
    )
    
    # Centrality network
    plot_centrality_network(
        g,
        node_centrality,
        os.path.join(figures_dir, 'centrality_network.png')
    )
    
    # SBM community network
    if 'block_assignment' in sbm_results:
        plot_community_network(
            g,
            sbm_results['block_assignment'],
            os.path.join(figures_dir, 'community_sbm.png')
        )
        
        # SBM matrix
        plot_sbm_matrix(
            sbm_results['block_assignment'],
            sbm_results['connection_probability_matrix'],
            os.path.join(figures_dir, 'sbm_block_matrix.png'),
            dpi=dpi
        )
    
    # Temporal activity
    if timeline:
        plot_temporal_activity(
            timeline,
            os.path.join(figures_dir, 'temporal_activity.png'),
            dpi=dpi
        )
    
    # Dynamic SBM
    if dynamic_sbm_results and 'transition_matrix' in dynamic_sbm_results:
        if dynamic_sbm_results['transition_matrix'].size > 0:
            plot_dynamic_sbm_transitions(
                dynamic_sbm_results['transition_matrix'],
                dynamic_sbm_results['block_labels'],
                os.path.join(figures_dir, 'dynamic_sbm_transitions.png'),
                dpi=dpi
            )
            
            if 'window_block_info' in dynamic_sbm_results:
                plot_dynamic_sbm_evolution(
                    dynamic_sbm_results['window_block_info'],
                    os.path.join(figures_dir, 'dynamic_sbm_evolution.png'),
                    dpi=dpi
                )
    
    print(f"Visualizations saved to {figures_dir}")
