"""
Static Network Analysis Module
==============================

Computes network metrics on the aggregated (static) graph.
Includes centrality measures, clustering, and network-level statistics.
"""

import numpy as np
from graph_tool.all import Graph, shortest_distance, pseudo_diameter
from graph_tool.centrality import betweenness, closeness, eigenvector
from graph_tool.clustering import local_clustering, global_clustering
from typing import Dict, Set, Tuple, List


def build_graph(n_nodes: int, edges: Set[Tuple[int, int]]) -> Graph:
    """Build a graph-tool Graph from nodes and edges."""
    g = Graph(directed=False)
    g.add_vertex(n_nodes)
    
    for e1, e2 in edges:
        g.add_edge(e1, e2)
    
    return g


def compute_basic_stats(g: Graph) -> dict:
    """Compute basic network statistics."""
    n = g.num_vertices()
    m = g.num_edges()
    max_edges = n * (n - 1) / 2
    density = m / max_edges if max_edges > 0 else 0
    
    # Check connectivity
    from graph_tool.topology import label_components
    comp, hist = label_components(g)
    n_components = len(hist)
    largest_component = max(hist) if hist.size > 0 else 0
    
    return {
        'n_nodes': n,
        'n_edges': m,
        'density': density,
        'n_components': n_components,
        'largest_component_size': largest_component,
        'is_connected': n_components == 1
    }


def compute_distance_stats(g: Graph) -> dict:
    """Compute distance-based statistics (diameter, avg path length)."""
    from graph_tool.topology import label_components
    
    comp, hist = label_components(g)
    
    # Work on largest component for meaningful distances
    if len(hist) > 1:
        # Find largest component
        largest_comp_idx = np.argmax(hist)
        nodes_in_largest = [int(v) for v in g.vertices() if comp[v] == largest_comp_idx]
        
        # Create subgraph
        vfilt = g.new_vertex_property("bool")
        for v in nodes_in_largest:
            vfilt[g.vertex(v)] = True
        g_sub = Graph(g, vp_filter=vfilt)
    else:
        g_sub = g
    
    # Pseudo-diameter (approximate)
    try:
        dist, ends = pseudo_diameter(g_sub)
        diameter = int(dist)
    except:
        diameter = -1
    
    # Average path length (sample for large graphs)
    n_sub = g_sub.num_vertices()
    if n_sub < 500:
        # Compute all pairs
        total_dist = 0
        count = 0
        for v in g_sub.vertices():
            dists = shortest_distance(g_sub, source=v)
            for d in dists.a:
                if d > 0 and d < 2147483647:  # Valid distance
                    total_dist += d
                    count += 1
        avg_path = total_dist / count if count > 0 else 0
    else:
        # Sample
        import random
        sample_size = 100
        vertices = list(g_sub.vertices())
        sampled = random.sample(vertices, min(sample_size, len(vertices)))
        
        total_dist = 0
        count = 0
        for v in sampled:
            dists = shortest_distance(g_sub, source=v)
            for d in dists.a:
                if d > 0 and d < 2147483647:
                    total_dist += d
                    count += 1
        avg_path = total_dist / count if count > 0 else 0
    
    return {
        'diameter': diameter,
        'avg_path_length': round(avg_path, 4),
        'analysis_on_largest_component': len(hist) > 1
    }


def compute_degree_stats(g: Graph) -> Tuple[dict, np.ndarray]:
    """Compute degree distribution statistics."""
    degrees = g.get_out_degrees(g.get_vertices())
    
    stats = {
        'degree_min': int(np.min(degrees)),
        'degree_max': int(np.max(degrees)),
        'degree_mean': round(float(np.mean(degrees)), 4),
        'degree_std': round(float(np.std(degrees)), 4),
        'degree_median': float(np.median(degrees))
    }
    
    return stats, degrees


def compute_clustering_stats(g: Graph) -> dict:
    """Compute clustering coefficients."""
    # Local clustering per node
    local_clust = local_clustering(g)
    local_values = local_clust.a
    
    # Filter out NaN (nodes with degree < 2)
    valid_values = local_values[~np.isnan(local_values)]
    avg_local = float(np.mean(valid_values)) if len(valid_values) > 0 else 0
    
    # Global clustering (transitivity)
    global_clust = global_clustering(g)
    
    return {
        'clustering_coefficient': round(avg_local, 4),
        'transitivity': round(global_clust[0], 4)
    }


def compute_centrality(g: Graph) -> Tuple[dict, dict]:
    """
    Compute centrality measures for all nodes.
    
    Returns
    -------
    centrality_stats : dict
        Network-level centralization measures
    node_centrality : dict
        Per-node centrality values
    """
    n = g.num_vertices()
    
    # Degree centrality (normalized)
    degrees = g.get_out_degrees(g.get_vertices())
    degree_centrality = degrees / (n - 1) if n > 1 else degrees
    
    # Betweenness centrality
    # Note: graph-tool's betweenness() with norm=True (default) already normalizes
    # by dividing by (n-1)(n-2)/2, so values are in [0, 1]
    vb, eb = betweenness(g)
    betweenness_vals = vb.a.copy()
    # No additional normalization needed - graph-tool already normalizes
    
    # Closeness centrality
    closeness_vals = closeness(g).a.copy()
    
    # Eigenvector centrality
    ev_max, eigenvector_vals = eigenvector(g)
    eigenvector_vals = eigenvector_vals.a.copy()
    
    # Compute network centralization (Freeman's centralization)
    # C = sum(C_max - C_i) / max_possible_sum
    
    def freeman_centralization(values):
        """Compute Freeman's centralization index."""
        values = np.array(values)
        max_val = float(np.max(values))
        sum_diff = float(np.sum(max_val - values))
        # Theoretical max for star graph
        max_possible = (n - 1) * (n - 2) / (n - 1) if n > 2 else 1
        return sum_diff / max_possible if max_possible > 0 else 0.0
    
    def freeman_centralization_betweenness(values):
        """Compute Freeman's centralization for betweenness (already normalized [0,1])."""
        values = np.array(values)
        max_val = float(np.max(values))
        sum_diff = float(np.sum(max_val - values))
        # For normalized betweenness in [0,1], max possible sum of differences
        # is (n-1) (star graph: center=1, all others=0)
        max_possible = n - 1
        return sum_diff / max_possible if max_possible > 0 else 0.0
    
    centr_degree = float(freeman_centralization(degree_centrality))
    centr_betweenness = float(freeman_centralization_betweenness(betweenness_vals))
    centr_closeness = float(freeman_centralization(closeness_vals[~np.isnan(closeness_vals)]))
    
    centrality_stats = {
        'centralization_degree': round(centr_degree, 4),
        'centralization_betweenness': round(centr_betweenness, 4),
        'centralization_closeness': round(centr_closeness, 4)
    }
    
    node_centrality = {
        'degree': degree_centrality,
        'betweenness': betweenness_vals,
        'closeness': closeness_vals,
        'eigenvector': eigenvector_vals
    }
    
    return centrality_stats, node_centrality


def get_top_nodes(
    node_centrality: dict,
    node_list: List[int],
    top_k: int = 20
) -> List[dict]:
    """Get top K nodes by different centrality measures."""
    n = len(node_list)
    
    # Create ranking for each measure
    rankings = {}
    for measure, values in node_centrality.items():
        order = np.argsort(-values)  # Descending
        rank = np.zeros(n, dtype=int)
        rank[order] = np.arange(1, n + 1)
        rankings[measure] = rank
    
    # Get top K by degree (primary)
    top_indices = np.argsort(-node_centrality['degree'])[:top_k]
    
    top_nodes = []
    for idx in top_indices:
        node_info = {
            'node_id': node_list[idx],
            'node_index': idx,
            'degree': round(float(node_centrality['degree'][idx]), 4),
            'degree_rank': int(rankings['degree'][idx]),
            'betweenness': round(float(node_centrality['betweenness'][idx]), 4),
            'betweenness_rank': int(rankings['betweenness'][idx]),
            'closeness': round(float(node_centrality['closeness'][idx]), 4),
            'closeness_rank': int(rankings['closeness'][idx]),
            'eigenvector': round(float(node_centrality['eigenvector'][idx]), 4),
            'eigenvector_rank': int(rankings['eigenvector'][idx])
        }
        top_nodes.append(node_info)
    
    return top_nodes


def run_static_analysis(
    n_nodes: int,
    edges: Set[Tuple[int, int]],
    node_list: List[int]
) -> Tuple[dict, List[dict], dict, Graph]:
    """
    Run complete static analysis.
    
    Returns
    -------
    metrics : dict
        All network-level metrics
    top_nodes : list
        Top nodes by centrality
    node_centrality : dict
        Per-node centrality values
    g : Graph
        The graph-tool Graph object
    """
    print("Building graph...")
    g = build_graph(n_nodes, edges)
    
    print("Computing basic statistics...")
    metrics = compute_basic_stats(g)
    
    print("Computing distance statistics...")
    metrics.update(compute_distance_stats(g))
    
    print("Computing degree statistics...")
    degree_stats, degrees = compute_degree_stats(g)
    metrics.update(degree_stats)
    
    print("Computing clustering...")
    metrics.update(compute_clustering_stats(g))
    
    print("Computing centrality measures...")
    centrality_stats, node_centrality = compute_centrality(g)
    metrics.update(centrality_stats)
    
    print("Ranking top nodes...")
    top_nodes = get_top_nodes(node_centrality, node_list)
    
    return metrics, top_nodes, node_centrality, g
