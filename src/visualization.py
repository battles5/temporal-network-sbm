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
    cbar.set_label('Connection Probability', fontsize=11)
    
    # Add labels - just numbers, no "Block" prefix
    ax.set_xticks(range(n_blocks))
    ax.set_yticks(range(n_blocks))
    ax.set_xticklabels(range(n_blocks), fontsize=8)
    ax.set_yticklabels(range(n_blocks), fontsize=8)
    ax.set_xlabel('Block', fontsize=10)
    ax.set_ylabel('Block', fontsize=10)
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
    
    # Adjust figure size based on number of blocks (same as connection matrix)
    fig_size = max(6, min(12, n_blocks * 0.5))
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
    
    im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Transition Probability', fontsize=11)
    
    # Add labels - just numbers (same style as connection matrix)
    ax.set_xticks(range(n_blocks))
    ax.set_yticks(range(n_blocks))
    ax.set_xticklabels(block_labels, fontsize=8)
    ax.set_yticklabels(block_labels, fontsize=8)
    ax.set_xlabel('Block (t+1)', fontsize=10)
    ax.set_ylabel('Block (t)', fontsize=10)
    ax.set_title('Transition Probability Matrix', fontsize=12)
    
    # Remove grid lines
    ax.grid(False)
    
    # Add values as text - same adaptive font as connection matrix
    fontsize = max(5, min(8, 120 // n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            val = transition_matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            # Use 1 decimal for cleaner look, or 2 if value is small
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
    Plot evolution of block sizes over time as a heatmap.
    
    Shows blocks (rows) vs time windows (columns), with color intensity
    representing block size (number of nodes in that block at that time).
    
    This visualization is more interpretable than stacked area charts
    when there are many blocks, and clearly shows:
    - Block existence (non-zero cells)
    - Block size variations over time
    - Block births/deaths (rows that appear/disappear)
    """
    setup_style()
    
    # Collect all blocks that appear at least once
    all_blocks = set()
    for winfo in window_block_info:
        all_blocks.update(winfo['block_sizes'].keys())
    all_blocks = sorted(all_blocks)
    n_blocks = len(all_blocks)
    n_windows = len(window_block_info)
    
    if n_blocks == 0:
        print("  Warning: No blocks found, skipping evolution plot")
        return
    
    # Build matrix: rows = blocks, columns = time windows
    # Value = number of nodes in that block at that time
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    size_matrix = np.zeros((n_blocks, n_windows))
    
    for t, winfo in enumerate(window_block_info):
        for block, size in winfo['block_sizes'].items():
            size_matrix[block_to_idx[block], t] = size
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(4, n_blocks * 0.35)))
    
    # Use monochrome colormap (Blues) with power-law scaling for better visibility
    # of smaller blocks. gamma < 1 expands the lower range.
    from matplotlib.colors import PowerNorm
    max_size = size_matrix.max()
    im = ax.imshow(size_matrix, aspect='auto', cmap='Blues', 
                   interpolation='nearest',
                   norm=PowerNorm(gamma=0.5, vmin=0, vmax=max_size))
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Block Size (nodes)', fontsize=10)
    
    # Axis labels - clarify that each window covers ~39 minutes for 50 windows over 32h
    ax.set_xlabel('Time Window (each ≈ 39 min, total ≈ 32 hours)', fontsize=11)
    ax.set_ylabel('Block', fontsize=11)
    ax.set_title('Block Sizes Over Time (Dynamic SBM)', fontsize=12)
    
    # Y-axis: block labels
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels(all_blocks, fontsize=8)
    
    # X-axis: show every 5th or 10th window depending on count
    step = 5 if n_windows <= 50 else 10
    ax.set_xticks(range(0, n_windows, step))
    ax.set_xticklabels(range(0, n_windows, step), fontsize=8)
    
    # Add grid for readability
    ax.set_xticks(np.arange(-0.5, n_windows, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_blocks, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    
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


# ============================================================================
# Hypergraph Visualizations
# ============================================================================

def plot_group_size_distribution(
    distribution_df,
    output_path: str,
    dpi: int = 300
):
    """
    Plot the distribution of group sizes (hyperedge sizes).
    
    Creates a bar chart showing how many groups of each size were found,
    with both linear and log-scale views.
    
    Parameters
    ----------
    distribution_df : pandas.DataFrame
        DataFrame with columns: group_size, count, proportion
    output_path : str
        Path to save the figure
    dpi : int
        Output resolution
    """
    setup_style()
    
    if distribution_df.empty:
        print(f"  Skipped (no groups): {output_path}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sizes = distribution_df['group_size'].values
    counts = distribution_df['count'].values
    
    # Linear scale bar chart
    ax1 = axes[0]
    bars1 = ax1.bar(sizes, counts, color='steelblue', edgecolor='white', alpha=0.9)
    ax1.set_xlabel('Group Size (clique size)')
    ax1.set_ylabel('Count')
    ax1.set_title('Group Size Distribution')
    ax1.set_xticks(sizes)
    
    # Add proportion labels on bars
    total = counts.sum()
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        pct = 100 * count / total
        if pct >= 3:  # Only label bars >= 3%
            ax1.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Log scale bar chart
    ax2 = axes[1]
    ax2.bar(sizes, counts, color='coral', edgecolor='white', alpha=0.9)
    ax2.set_xlabel('Group Size (clique size)')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title('Group Size Distribution (Log Scale)')
    ax2.set_yscale('log')
    ax2.set_xticks(sizes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_group_size_over_time(
    groups_by_window: dict,
    output_path: str,
    dpi: int = 300
):
    """
    Plot median and quantiles of group size per time window.
    
    Parameters
    ----------
    groups_by_window : dict
        Keys are window_id, values are lists of group sizes
    output_path : str
        Path to save the figure
    dpi : int
        Output resolution
    """
    import pandas as pd
    
    setup_style()
    
    if not groups_by_window:
        print(f"  Skipped (no data): {output_path}")
        return
    
    # Compute statistics per window
    window_ids = sorted(groups_by_window.keys())
    medians = []
    q25 = []
    q75 = []
    means = []
    
    for w in window_ids:
        sizes = groups_by_window[w]
        if sizes:
            medians.append(np.median(sizes))
            q25.append(np.percentile(sizes, 25))
            q75.append(np.percentile(sizes, 75))
            means.append(np.mean(sizes))
        else:
            medians.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)
            means.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot with fill between for IQR
    ax.fill_between(window_ids, q25, q75, alpha=0.3, color='steelblue', 
                   label='IQR (25-75%)')
    ax.plot(window_ids, medians, 'o-', color='steelblue', linewidth=2, 
            markersize=4, label='Median')
    ax.plot(window_ids, means, '--', color='coral', linewidth=1.5, 
            alpha=0.8, label='Mean')
    
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Group Size')
    ax.set_title('Group Size Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_hypergraph_visualizations(
    hypergraph_results: dict,
    output_dir: str,
    dpi: int = 300
):
    """
    Create all hypergraph-related visualizations.
    
    Parameters
    ----------
    hypergraph_results : dict
        Results from run_hypergraph_analysis()
    output_dir : str
        Output directory
    dpi : int
        Output resolution
    """
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Creating hypergraph visualizations...")
    
    # Group size distribution
    if 'distribution' in hypergraph_results:
        plot_group_size_distribution(
            hypergraph_results['distribution'],
            os.path.join(figures_dir, 'group_size_distribution.png'),
            dpi=dpi
        )
    
    # Group size over time
    if 'groups' in hypergraph_results and hypergraph_results['groups']:
        # Build groups_by_window
        groups_by_window = {}
        for g in hypergraph_results['groups']:
            w = g['window_id']
            if w not in groups_by_window:
                groups_by_window[w] = []
            groups_by_window[w].append(g['group_size'])
        
        plot_group_size_over_time(
            groups_by_window,
            os.path.join(figures_dir, 'group_size_over_time.png'),
            dpi=dpi
        )
