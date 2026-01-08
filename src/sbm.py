"""
Stochastic Block Model Module
=============================

Implements SBM fitting using graph-tool's inference machinery.

Theory
------
The Stochastic Block Model (SBM) is a generative model for networks where:
- Nodes are partitioned into K blocks (communities)
- The probability of an edge between nodes depends only on their block memberships
- P(edge between i,j) = θ_{b_i, b_j} where b_i is the block of node i

The model is fitted by minimizing the description length (MDL principle),
which balances model complexity with data fit.
"""

import numpy as np
from graph_tool.all import Graph, minimize_blockmodel_dl, BlockState
from typing import Tuple, Dict, List, Set
import warnings


def fit_sbm(
    g: Graph,
    max_blocks: int = 20,
    equilibrate: bool = True,
    n_init: int = 10
) -> Tuple[BlockState, dict]:
    """
    Fit a Stochastic Block Model to the graph.
    
    Uses graph-tool's minimize_blockmodel_dl which finds the optimal
    number of blocks by minimizing the description length (MDL).
    
    Parameters
    ----------
    g : Graph
        The input graph
    max_blocks : int
        Maximum number of blocks to consider
    equilibrate : bool
        Whether to run MCMC equilibration for better estimates
    n_init : int
        Number of random initializations
        
    Returns
    -------
    state : BlockState
        The fitted block model state
    results : dict
        Summary of SBM results
    """
    print("Fitting Stochastic Block Model...")
    print(f"  Max blocks: {max_blocks}, Equilibrate: {equilibrate}")
    
    # Run multiple initializations and keep the best
    best_state = None
    best_dl = float('inf')
    
    for i in range(n_init):
        state = minimize_blockmodel_dl(
            g,
            state_args=dict(B=max_blocks),
            multilevel_mcmc_args=dict(B_min=1, B_max=max_blocks)
        )
        
        if equilibrate:
            # MCMC equilibration for better estimates
            for _ in range(100):
                state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
        
        dl = state.entropy()
        if dl < best_dl:
            best_dl = dl
            best_state = state
    
    # Extract results
    blocks = best_state.get_blocks()
    block_array = np.array([blocks[v] for v in g.vertices()])
    
    # Count block sizes
    unique_blocks, block_counts = np.unique(block_array, return_counts=True)
    n_blocks = len(unique_blocks)
    
    # Compute block statistics
    block_sizes = {int(b): int(c) for b, c in zip(unique_blocks, block_counts)}
    
    # Get connection matrix between blocks
    # This is the expected number of edges between blocks
    e_rs = best_state.get_matrix()
    connection_matrix = np.array(e_rs.todense())
    
    # Normalize to get probabilities
    n = g.num_vertices()
    block_prob_matrix = np.zeros((n_blocks, n_blocks))
    for r in range(n_blocks):
        for s in range(n_blocks):
            n_r = block_counts[r]
            n_s = block_counts[s]
            if r == s:
                max_edges = n_r * (n_r - 1) / 2
            else:
                max_edges = n_r * n_s
            if max_edges > 0:
                block_prob_matrix[r, s] = connection_matrix[r, s] / max_edges
    
    results = {
        'n_blocks': n_blocks,
        'description_length': round(best_dl, 2),
        'block_sizes': block_sizes,
        'block_assignment': block_array,
        'connection_matrix': connection_matrix,
        'connection_probability_matrix': block_prob_matrix
    }
    
    print(f"  Optimal blocks: {n_blocks}")
    print(f"  Description length: {best_dl:.2f}")
    print(f"  Block sizes: {list(block_counts)}")
    
    return best_state, results


def compute_block_densities(
    g: Graph,
    block_assignment: np.ndarray
) -> Dict[int, float]:
    """Compute internal density for each block."""
    unique_blocks = np.unique(block_assignment)
    densities = {}
    
    for b in unique_blocks:
        nodes_in_block = np.where(block_assignment == b)[0]
        n_b = len(nodes_in_block)
        
        if n_b < 2:
            densities[int(b)] = 0.0
            continue
        
        # Count edges within block
        internal_edges = 0
        node_set = set(nodes_in_block)
        for v in nodes_in_block:
            for e in g.vertex(v).out_edges():
                neighbor = int(e.target())
                if neighbor in node_set and neighbor > v:
                    internal_edges += 1
        
        max_internal = n_b * (n_b - 1) / 2
        densities[int(b)] = round(internal_edges / max_internal, 4) if max_internal > 0 else 0
    
    return densities


def get_sbm_summary(
    results: dict,
    node_list: List[int]
) -> List[dict]:
    """Generate summary table for each block."""
    block_assignment = results['block_assignment']
    unique_blocks = np.unique(block_assignment)
    
    summary = []
    for b in unique_blocks:
        nodes_in_block = np.where(block_assignment == b)[0]
        original_nodes = [node_list[i] for i in nodes_in_block]
        
        summary.append({
            'block': int(b),
            'size': len(nodes_in_block),
            'nodes': original_nodes[:10],  # First 10 nodes
            'nodes_truncated': len(original_nodes) > 10
        })
    
    return summary


def run_sbm_analysis(
    g: Graph,
    node_list: List[int],
    max_blocks: int = 20,
    equilibrate: bool = True
) -> Tuple[dict, List[dict], np.ndarray]:
    """
    Run complete SBM analysis.
    
    Returns
    -------
    results : dict
        SBM results including matrices
    block_summary : list
        Summary for each block
    block_assignment : array
        Block assignment for each node
    """
    state, results = fit_sbm(g, max_blocks, equilibrate)
    
    # Add block densities
    densities = compute_block_densities(g, results['block_assignment'])
    results['block_densities'] = densities
    
    # Generate summary
    block_summary = get_sbm_summary(results, node_list)
    
    # Add density to summary
    for item in block_summary:
        item['internal_density'] = densities[item['block']]
    
    return results, block_summary, results['block_assignment']
