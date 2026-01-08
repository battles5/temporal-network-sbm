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
    
    # Remap blocks to 0, 1, ..., n_blocks-1 for consistent indexing
    block_map = {b: i for i, b in enumerate(unique_blocks)}
    remapped_blocks = np.array([block_map[b] for b in block_array])
    
    # Compute block statistics
    block_sizes = {int(b): int(c) for b, c in zip(unique_blocks, block_counts)}
    
    # Count edges between blocks manually (more reliable than get_matrix)
    edge_counts = np.zeros((n_blocks, n_blocks))
    for edge in g.edges():
        i, j = int(edge.source()), int(edge.target())
        q_i, q_j = remapped_blocks[i], remapped_blocks[j]
        # For undirected: count in upper triangle only, then symmetrize
        if q_i <= q_j:
            edge_counts[q_i, q_j] += 1
        else:
            edge_counts[q_j, q_i] += 1
    
    # Store raw edge counts (symmetric)
    connection_matrix = edge_counts + edge_counts.T - np.diag(np.diag(edge_counts))
    
    # Compute probability matrix: pi_ql = m_ql / max_possible_edges
    n = g.num_vertices()
    block_prob_matrix = np.zeros((n_blocks, n_blocks))
    for r in range(n_blocks):
        for s in range(r, n_blocks):  # Upper triangle + diagonal
            n_r = block_counts[r]
            n_s = block_counts[s]
            if r == s:
                # Within-block: n_r choose 2
                max_edges = n_r * (n_r - 1) / 2
            else:
                # Between-block: n_r * n_s
                max_edges = n_r * n_s
            
            if max_edges > 0:
                # Use edge_counts (upper triangle)
                pi_rs = edge_counts[r, s] / max_edges
                block_prob_matrix[r, s] = pi_rs
                block_prob_matrix[s, r] = pi_rs  # Symmetric
    
    results = {
        'n_blocks': n_blocks,
        'description_length': round(best_dl, 2),
        'block_sizes': block_sizes,
        'block_assignment': block_array,
        'connection_matrix': connection_matrix,
        'connection_probability_matrix': block_prob_matrix,
        'edge_counts_between_blocks': edge_counts
    }
    
    print(f"  Optimal blocks: {n_blocks}")
    print(f"  Description length: {best_dl:.2f}")
    print(f"  Block sizes: {list(block_counts)}")
    
    # Diagnostic: probability matrix stats
    pi_vals = block_prob_matrix[block_prob_matrix > 0]
    if len(pi_vals) > 0:
        print(f"  Prob matrix: min={pi_vals.min():.4f}, max={pi_vals.max():.4f}, "
              f"mean={pi_vals.mean():.4f}, non-zero cells={len(pi_vals)}")
    
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


def compute_icl(
    g: Graph,
    block_assignment: np.ndarray
) -> float:
    """
    Compute the Integrated Classification Likelihood (ICL) for SBM.
    
    Following the course notation (M.F. Marino), ICL is an approximation
    of the integrated likelihood that penalizes model complexity.
    
    ICL = log p(y | z_hat, theta_hat) - penalty_alpha - penalty_pi
    
    where:
    - penalty_alpha = (Q-1)/2 * log(n)  for block proportions
    - penalty_pi = Q(Q+1)/4 * log(n(n-1)/2)  for connection probabilities
    
    Parameters
    ----------
    g : Graph
        The input graph
    block_assignment : np.ndarray
        Block assignment for each node
        
    Returns
    -------
    icl : float
        The ICL value (higher is better)
    """
    n = g.num_vertices()
    m = g.num_edges()
    
    unique_blocks = np.unique(block_assignment)
    Q = len(unique_blocks)
    
    if Q == 0 or n < 2:
        return float('-inf')
    
    # Remap blocks to 0, 1, ..., Q-1
    block_map = {b: i for i, b in enumerate(unique_blocks)}
    z = np.array([block_map[b] for b in block_assignment])
    
    # Count block sizes
    n_q = np.bincount(z, minlength=Q)
    
    # Count edges within and between blocks
    e_ql = np.zeros((Q, Q))
    for edge in g.edges():
        i, j = int(edge.source()), int(edge.target())
        q_i, q_j = z[i], z[j]
        if q_i <= q_j:
            e_ql[q_i, q_j] += 1
        else:
            e_ql[q_j, q_i] += 1
    
    # Compute log-likelihood
    log_lik = 0.0
    for q in range(Q):
        for l in range(q, Q):
            if q == l:
                # Within-block edges
                n_possible = n_q[q] * (n_q[q] - 1) / 2
            else:
                # Between-block edges
                n_possible = n_q[q] * n_q[l]
            
            if n_possible > 0:
                e_obs = e_ql[q, l]
                pi_ql = e_obs / n_possible if n_possible > 0 else 0
                
                # Avoid log(0)
                if pi_ql > 0 and pi_ql < 1:
                    log_lik += e_obs * np.log(pi_ql)
                    log_lik += (n_possible - e_obs) * np.log(1 - pi_ql)
                elif pi_ql == 1 and e_obs > 0:
                    # All possible edges exist, log(1) = 0
                    pass
                elif pi_ql == 0:
                    # No edges, log(1) = 0
                    pass
    
    # Penalty terms (BIC-like)
    # Penalty for alpha (Q-1 free parameters)
    penalty_alpha = (Q - 1) / 2 * np.log(n)
    
    # Penalty for pi (Q(Q+1)/2 parameters for undirected network)
    n_pairs = n * (n - 1) / 2
    penalty_pi = Q * (Q + 1) / 4 * np.log(n_pairs) if n_pairs > 1 else 0
    
    icl = log_lik - penalty_alpha - penalty_pi
    
    return round(icl, 2)


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
    
    # Compute ICL (Integrated Classification Likelihood)
    icl = compute_icl(g, results['block_assignment'])
    results['icl'] = icl
    print(f"  ICL: {icl}")
    
    # Generate summary
    block_summary = get_sbm_summary(results, node_list)
    
    # Add density to summary
    for item in block_summary:
        item['internal_density'] = densities[item['block']]
    
    return results, block_summary, results['block_assignment']
