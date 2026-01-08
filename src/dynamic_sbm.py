"""
Dynamic Stochastic Block Model Module
======================================

Implements dynamic SBM analysis using a sliding window approach,
following the methodology of the dynsbm R package.

Theory
------
Dynamic SBM extends the static SBM to temporal networks:

1. The time domain is divided into T windows
2. For each window t, we observe a network G_t
3. Nodes are assigned to blocks, and block membership can change over time
4. The model estimates:
   - Block assignments B_t for each time window
   - Transition matrix P(b_{t+1} | b_t) between consecutive windows
   - Connection probabilities θ_{rs} between blocks (can be time-varying or fixed)

This implementation uses a sliding window approach:
1. For each time window, fit a static SBM
2. Align block labels across windows (solve label switching problem)
3. Compute empirical transition probabilities

References
----------
- Matias, C., & Miele, V. (2017). Statistical clustering of temporal 
  dynamic networks. Statistical Computing, 27(4), 1065-1086.
- The dynsbm R package
"""

import numpy as np
from graph_tool.all import Graph
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from .sbm import fit_sbm, compute_block_densities


def remap_to_contiguous(assignment: np.ndarray) -> np.ndarray:
    """
    Remap block assignments to contiguous integers 0, 1, ..., K-1.
    """
    unique_blocks = np.unique(assignment)
    mapping = {b: i for i, b in enumerate(unique_blocks)}
    return np.array([mapping[b] for b in assignment])


def align_block_labels(
    prev_assignment: np.ndarray,
    curr_assignment: np.ndarray
) -> np.ndarray:
    """
    Align block labels between consecutive time windows.
    
    Uses the Hungarian algorithm to find the optimal matching
    that maximizes overlap between blocks.
    
    Parameters
    ----------
    prev_assignment : array
        Block assignments from previous window (should be 0..K-1)
    curr_assignment : array
        Block assignments from current window
        
    Returns
    -------
    aligned : array
        Realigned block assignments for current window (0..K'-1)
    """
    # First remap current to contiguous
    curr_assignment = remap_to_contiguous(curr_assignment)
    
    prev_blocks = np.unique(prev_assignment)
    curr_blocks = np.unique(curr_assignment)
    
    n_prev = len(prev_blocks)
    n_curr = len(curr_blocks)
    
    if n_curr == 0 or n_prev == 0:
        return curr_assignment
    
    # Build cost matrix (negative overlap for minimization)
    cost_matrix = np.zeros((n_curr, n_prev))
    for i, cb in enumerate(curr_blocks):
        for j, pb in enumerate(prev_blocks):
            overlap = np.sum((curr_assignment == cb) & (prev_assignment == pb))
            cost_matrix[i, j] = -overlap  # Negative for minimization
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping: current block -> aligned label
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[curr_blocks[r]] = prev_blocks[c]
    
    # Handle unmapped blocks (new blocks that didn't exist in prev)
    # Assign them new labels starting from max(prev) + 1
    next_label = int(max(prev_blocks)) + 1 if len(prev_blocks) > 0 else 0
    for cb in curr_blocks:
        if cb not in mapping:
            mapping[cb] = next_label
            next_label += 1
    
    # Apply mapping
    aligned = np.array([mapping[b] for b in curr_assignment])
    
    return aligned


def compute_transition_matrix(
    assignments: List[np.ndarray]
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute empirical transition matrix between blocks.
    
    Parameters
    ----------
    assignments : list of arrays
        Block assignments for each time window
        
    Returns
    -------
    transition_matrix : array
        P(b_{t+1} = s | b_t = r)
    block_labels : list
        Contiguous block labels 0, 1, 2, ..., K-1
    """
    if len(assignments) < 2:
        return np.array([]), []
    
    # Get all unique blocks
    all_blocks = set()
    for a in assignments:
        all_blocks.update(np.unique(a))
    original_labels = sorted(all_blocks)
    n_blocks = len(original_labels)
    
    # Map original labels to contiguous 0..K-1
    label_to_idx = {b: i for i, b in enumerate(original_labels)}
    
    # Return contiguous labels 0, 1, 2, ..., K-1
    block_labels = list(range(n_blocks))
    
    # Count transitions
    transition_counts = np.zeros((n_blocks, n_blocks))
    
    for t in range(len(assignments) - 1):
        prev = assignments[t]
        curr = assignments[t + 1]
        
        for node in range(len(prev)):
            if node < len(curr):
                from_block = label_to_idx[prev[node]]
                to_block = label_to_idx[curr[node]]
                transition_counts[from_block, to_block] += 1
    
    # Normalize to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_counts / row_sums
    
    return transition_matrix, block_labels


def compute_block_stability(assignments: List[np.ndarray]) -> Dict[int, float]:
    """
    Compute stability of each block over time.
    
    Stability is the average probability that a node stays in the same block.
    """
    if len(assignments) < 2:
        return {}
    
    all_blocks = set()
    for a in assignments:
        all_blocks.update(np.unique(a))
    
    stability = {}
    for block in all_blocks:
        stays = 0
        total = 0
        
        for t in range(len(assignments) - 1):
            nodes_in_block = np.where(assignments[t] == block)[0]
            for node in nodes_in_block:
                if node < len(assignments[t + 1]):
                    total += 1
                    if assignments[t + 1][node] == block:
                        stays += 1
        
        stability[int(block)] = round(stays / total, 4) if total > 0 else 0
    
    return stability


def identify_movers(
    assignments: List[np.ndarray],
    node_list: List[int]
) -> List[dict]:
    """
    Identify nodes that change blocks over time.
    
    Returns list of nodes with their block history.
    """
    n_nodes = len(assignments[0]) if assignments else 0
    n_windows = len(assignments)
    
    movers = []
    for node in range(n_nodes):
        history = [assignments[t][node] if node < len(assignments[t]) else -1 
                   for t in range(n_windows)]
        unique_blocks = len(set(h for h in history if h >= 0))
        
        if unique_blocks > 1:
            movers.append({
                'node_id': node_list[node],
                'node_index': node,
                'n_block_changes': unique_blocks - 1,
                'block_history': history
            })
    
    # Sort by number of changes
    movers.sort(key=lambda x: -x['n_block_changes'])
    
    return movers


def build_window_graph(
    n_nodes: int,
    edges: Set[Tuple[int, int]]
) -> Graph:
    """Build a graph from edges for a time window."""
    g = Graph(directed=False)
    g.add_vertex(n_nodes)
    
    for e1, e2 in edges:
        g.add_edge(e1, e2)
    
    return g


def run_dynamic_sbm(
    n_nodes: int,
    time_windows: List[Tuple[int, int, Set[Tuple[int, int]]]],
    node_list: List[int],
    max_blocks: int = 10,
    min_edges_per_window: int = 10
) -> dict:
    """
    Run Dynamic SBM analysis on temporal network.
    
    Parameters
    ----------
    n_nodes : int
        Total number of nodes
    time_windows : list
        List of (start_time, end_time, edges) tuples
    node_list : list
        Original node IDs
    max_blocks : int
        Maximum blocks for SBM
    min_edges_per_window : int
        Minimum edges required to fit SBM for a window
        
    Returns
    -------
    results : dict
        Dynamic SBM results including transition matrix
    """
    print(f"\nRunning Dynamic SBM on {len(time_windows)} time windows...")
    
    assignments = []
    window_results = []
    prev_assignment = None
    
    for i, (t_start, t_end, edges) in enumerate(time_windows):
        if len(edges) < min_edges_per_window:
            print(f"  Window {i+1}: {len(edges)} edges (skipped - too few)")
            continue
        
        # Build graph for this window
        g = build_window_graph(n_nodes, edges)
        
        # Fit SBM
        try:
            state, sbm_results = fit_sbm(g, max_blocks=max_blocks, equilibrate=False, n_init=3)
            block_assignment = sbm_results['block_assignment']
            
            # Remap to contiguous 0..K-1 for first window
            if prev_assignment is None:
                block_assignment = remap_to_contiguous(block_assignment)
            else:
                # Align labels with previous window
                block_assignment = align_block_labels(prev_assignment, block_assignment)
            
            assignments.append(block_assignment)
            prev_assignment = block_assignment
            
            window_results.append({
                'window_index': i,
                'time_start': t_start,
                'time_end': t_end,
                'n_edges': len(edges),
                'n_blocks': sbm_results['n_blocks'],
                'description_length': sbm_results['description_length']
            })
            
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Window {i+1}/{len(time_windows)}: {len(edges)} edges, {sbm_results['n_blocks']} blocks")
                
        except Exception as e:
            print(f"  Window {i+1}: Error fitting SBM - {e}")
            continue
    
    if len(assignments) < 2:
        print("Warning: Not enough windows for transition analysis")
        return {
            'n_windows_analyzed': len(assignments),
            'window_results': window_results,
            'transition_matrix': np.array([]),
            'block_stability': {},
            'movers': []
        }
    
    # Compute transition matrix
    print("Computing transition probabilities...")
    transition_matrix, block_labels = compute_transition_matrix(assignments)
    
    # Compute block stability
    block_stability = compute_block_stability(assignments)
    
    # Identify movers
    movers = identify_movers(assignments, node_list)
    
    # Per-window block sizes
    window_block_info = []
    for i, assignment in enumerate(assignments):
        unique, counts = np.unique(assignment, return_counts=True)
        window_block_info.append({
            'window_index': window_results[i]['window_index'],
            'block_sizes': dict(zip(unique.astype(int).tolist(), counts.tolist()))
        })
    
    results = {
        'n_windows_analyzed': len(assignments),
        'window_results': window_results,
        'window_block_info': window_block_info,
        'transition_matrix': transition_matrix,
        'block_labels': block_labels,
        'block_stability': block_stability,
        'movers': movers[:20],  # Top 20 movers
        'assignments': assignments
    }
    
    print(f"Dynamic SBM complete: {len(assignments)} windows analyzed")
    print(f"  Transition matrix size: {transition_matrix.shape}")
    print(f"  Nodes that changed blocks: {len(movers)}")
    
    return results
