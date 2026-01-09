"""
Temporal Hypergraph Module
==========================

Implements group interaction extraction from temporal networks using
maximal clique enumeration, inspired by:

Iacopini, I., Petri, G., Baronchelli, A., & Barrat, A. (2022).
Group interactions modulate critical mass dynamics in social convention.
Communications Physics, 5, 64. https://doi.org/10.1038/s42005-022-00845-y

The key idea is that face-to-face interactions recorded as pairwise contacts
can be "lifted" to group interactions by identifying cliques in the contact
graph within each time window. A clique of size k represents a group of k
individuals who all interacted with each other during that time window.

Theory
------
Given a temporal edge list (t, i, j), we:
1. Partition time into windows [t_start, t_end)
2. For each window, build snapshot graph G_w with all edges in that window
3. Extract maximal cliques from G_w
4. Each clique becomes a hyperedge in the temporal hypergraph

This approximates higher-order interactions that are not directly observable
from pairwise data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import time
import warnings


def build_snapshot_edges(
    edges: List[Tuple[int, int, int]],
    t_start: int,
    t_end: int
) -> List[Tuple[int, int]]:
    """
    Extract edges from a time window.
    
    Parameters
    ----------
    edges : list of (timestamp, node1, node2)
        All temporal edges
    t_start : int
        Window start (inclusive)
    t_end : int
        Window end (exclusive)
        
    Returns
    -------
    list of (node1, node2)
        Unique edges in the window (undirected, no duplicates)
    """
    edge_set = set()
    for t, n1, n2 in edges:
        if t_start <= t < t_end:
            # Normalize edge direction for undirected graph
            edge = (min(n1, n2), max(n1, n2))
            edge_set.add(edge)
    return list(edge_set)


def extract_maximal_cliques(
    edges: List[Tuple[int, int]],
    min_size: int = 3,
    max_size: int = 20,
    max_cliques: int = 10000
) -> Tuple[List[List[int]], bool]:
    """
    Extract maximal cliques from a graph defined by edges.
    
    Uses NetworkX for clique enumeration (graph-tool doesn't have
    a convenient maximal clique function).
    
    Parameters
    ----------
    edges : list of (node1, node2)
        Graph edges
    min_size : int
        Minimum clique size to return (default 3 = triangles+)
    max_size : int
        Maximum clique size to consider (safety limit)
    max_cliques : int
        Maximum number of cliques to return (safety limit)
        
    Returns
    -------
    cliques : list of list of int
        Each clique as a sorted list of node IDs
    truncated : bool
        True if enumeration was truncated by max_cliques limit
    """
    if len(edges) == 0:
        return [], False
    
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "NetworkX is required for clique enumeration. "
            "Install with: pip install networkx"
        )
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Safety check: skip very dense graphs
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes > 0:
        density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        if density > 0.8 and n_nodes > 50:
            warnings.warn(
                f"Skipping clique enumeration: graph too dense "
                f"(density={density:.2f}, nodes={n_nodes})"
            )
            return [], False
    
    # Enumerate maximal cliques with limits
    cliques = []
    truncated = False
    
    try:
        for clique in nx.find_cliques(G):
            size = len(clique)
            if size >= min_size and size <= max_size:
                cliques.append(sorted(clique))
            
            if len(cliques) >= max_cliques:
                truncated = True
                break
    except Exception as e:
        warnings.warn(f"Clique enumeration failed: {e}")
        return [], False
    
    return cliques, truncated


def compute_group_size_distribution(
    all_cliques: List[List[int]]
) -> pd.DataFrame:
    """
    Compute the distribution of group sizes (hyperedge sizes).
    
    Parameters
    ----------
    all_cliques : list of list of int
        All cliques across all windows
        
    Returns
    -------
    DataFrame with columns: group_size, count, proportion
    """
    if len(all_cliques) == 0:
        return pd.DataFrame(columns=['group_size', 'count', 'proportion'])
    
    sizes = [len(c) for c in all_cliques]
    size_counts = pd.Series(sizes).value_counts().sort_index()
    
    total = int(size_counts.sum())
    counts_list = size_counts.values.tolist()
    proportions = [c / total for c in counts_list]
    df = pd.DataFrame({
        'group_size': size_counts.index.tolist(),
        'count': counts_list,
        'proportion': proportions
    })
    
    return df


def run_hypergraph_analysis(
    temporal_edges: List[Tuple[int, int, int]],
    time_windows: List[Dict],
    min_group_size: int = 3,
    max_group_size: int = 20,
    max_cliques_per_window: int = 10000,
    verbose: bool = True
) -> Dict:
    """
    Run full hypergraph analysis on temporal network.
    
    Parameters
    ----------
    temporal_edges : list of (timestamp, node1, node2)
        All temporal edges
    time_windows : list of dict
        Each dict has 't_start', 't_end' keys
    min_group_size : int
        Minimum clique size (default 3)
    max_group_size : int
        Maximum clique size (default 20)
    max_cliques_per_window : int
        Safety limit per window (default 10000)
    verbose : bool
        Print progress
        
    Returns
    -------
    dict with keys:
        'groups': list of dicts with window_id, group_id, group_size, node_ids
        'distribution': DataFrame of group size distribution
        'stats': summary statistics
    """
    if verbose:
        print(f"\nRunning Hypergraph Analysis on {len(time_windows)} windows...")
        print(f"  Min group size: {min_group_size}")
        print(f"  Max group size: {max_group_size}")
        print(f"  Max cliques/window: {max_cliques_per_window}")
    
    all_groups = []
    all_cliques = []
    total_truncations = 0
    group_id_counter = 0
    
    start_time = time.time()
    
    for w_idx, window in enumerate(time_windows):
        t_start = window['t_start']
        t_end = window['t_end']
        
        # Build snapshot
        edges = build_snapshot_edges(temporal_edges, t_start, t_end)
        n_edges = len(edges)
        
        if n_edges == 0:
            continue
        
        # Get unique nodes
        nodes = set()
        for e in edges:
            nodes.add(e[0])
            nodes.add(e[1])
        n_nodes = len(nodes)
        
        # Extract cliques
        w_start = time.time()
        cliques, truncated = extract_maximal_cliques(
            edges, min_group_size, max_group_size, max_cliques_per_window
        )
        w_time = time.time() - w_start
        
        if truncated:
            total_truncations += 1
        
        # Store cliques
        for clique in cliques:
            all_cliques.append(clique)
            all_groups.append({
                'window_id': w_idx,
                'group_id': group_id_counter,
                'group_size': len(clique),
                'node_ids': ' '.join(map(str, clique))
            })
            group_id_counter += 1
        
        # Progress logging
        if verbose and (w_idx + 1) % 10 == 0:
            print(f"  Window {w_idx + 1}/{len(time_windows)}: "
                  f"{n_nodes} nodes, {n_edges} edges, "
                  f"{len(cliques)} cliques, {w_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Compute distribution
    distribution = compute_group_size_distribution(all_cliques)
    
    # Summary stats
    stats = {
        'total_groups': len(all_cliques),
        'windows_analyzed': len(time_windows),
        'windows_with_groups': len(set(g['window_id'] for g in all_groups)),
        'truncated_windows': total_truncations,
        'min_group_size_config': min_group_size,
        'max_group_size_config': max_group_size,
        'analysis_time_seconds': round(total_time, 2)
    }
    
    if len(all_cliques) > 0:
        sizes = [len(c) for c in all_cliques]
        stats['mean_group_size'] = round(np.mean(sizes), 2)
        stats['median_group_size'] = int(np.median(sizes))
        stats['max_group_size_observed'] = max(sizes)
    
    if verbose:
        print(f"\nHypergraph analysis complete:")
        print(f"  Total groups (cliques): {stats['total_groups']}")
        print(f"  Windows with groups: {stats['windows_with_groups']}/{len(time_windows)}")
        if total_truncations > 0:
            print(f"  ⚠️  Truncated windows: {total_truncations}")
        print(f"  Time: {total_time:.1f}s")
    
    return {
        'groups': all_groups,
        'distribution': distribution,
        'stats': stats,
        'cliques_by_window': all_cliques
    }


def write_hypergraph_outputs(
    results: Dict,
    output_dir: str
):
    """
    Write hypergraph analysis results to CSV files.
    """
    import os
    
    # Groups CSV
    if results['groups']:
        groups_df = pd.DataFrame(results['groups'])
        groups_path = os.path.join(output_dir, 'hypergraph_groups.csv')
        groups_df.to_csv(groups_path, index=False)
        print(f"  Saved: {groups_path}")
    
    # Distribution CSV
    dist_path = os.path.join(output_dir, 'group_size_distribution.csv')
    results['distribution'].to_csv(dist_path, index=False)
    print(f"  Saved: {dist_path}")
