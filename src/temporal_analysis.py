"""
Temporal Analysis Module
========================

Analyzes temporal patterns in the network.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def compute_temporal_stats(
    temporal_edges: Dict[int, List[Tuple[int, int]]],
    time_windows: List[Tuple[int, int, Set[Tuple[int, int]]]]
) -> dict:
    """
    Compute temporal statistics for the network.
    """
    timestamps = sorted(temporal_edges.keys())
    
    # Edge counts per timestamp
    edges_per_ts = [len(temporal_edges[t]) for t in timestamps]
    
    # Edge counts per window
    edges_per_window = [len(edges) for _, _, edges in time_windows]
    
    # Find peak activity
    if edges_per_window:
        peak_idx = np.argmax(edges_per_window)
        peak_time = time_windows[peak_idx][0]
        peak_edges = edges_per_window[peak_idx]
    else:
        peak_time, peak_edges = 0, 0
    
    # Active nodes per window
    active_nodes_per_window = []
    for _, _, edges in time_windows:
        nodes = set()
        for e1, e2 in edges:
            nodes.add(e1)
            nodes.add(e2)
        active_nodes_per_window.append(len(nodes))
    
    stats = {
        'n_timestamps': len(timestamps),
        'duration_seconds': timestamps[-1] - timestamps[0] if timestamps else 0,
        'duration_minutes': round((timestamps[-1] - timestamps[0]) / 60, 2) if timestamps else 0,
        'n_time_windows': len(time_windows),
        'avg_edges_per_window': round(np.mean(edges_per_window), 2) if edges_per_window else 0,
        'std_edges_per_window': round(np.std(edges_per_window), 2) if edges_per_window else 0,
        'min_edges_per_window': int(np.min(edges_per_window)) if edges_per_window else 0,
        'max_edges_per_window': int(np.max(edges_per_window)) if edges_per_window else 0,
        'peak_activity_time': peak_time,
        'peak_activity_edges': peak_edges,
        'avg_active_nodes_per_window': round(np.mean(active_nodes_per_window), 2) if active_nodes_per_window else 0
    }
    
    return stats


def compute_activity_timeline(
    time_windows: List[Tuple[int, int, Set[Tuple[int, int]]]],
    reference_time: int = None
) -> List[dict]:
    """
    Create timeline of network activity.
    
    Returns list of dicts with time, n_edges, n_active_nodes.
    """
    if reference_time is None and time_windows:
        reference_time = time_windows[0][0]
    
    timeline = []
    for t_start, t_end, edges in time_windows:
        nodes = set()
        for e1, e2 in edges:
            nodes.add(e1)
            nodes.add(e2)
        
        timeline.append({
            'time_start': t_start,
            'time_minutes': round((t_start - reference_time) / 60, 2),
            'n_edges': len(edges),
            'n_active_nodes': len(nodes)
        })
    
    return timeline


def compute_node_activity(
    time_windows: List[Tuple[int, int, Set[Tuple[int, int]]]],
    node_list: List[int],
    n_nodes: int
) -> Dict[int, dict]:
    """
    Compute activity statistics per node.
    """
    node_stats = defaultdict(lambda: {'n_windows_active': 0, 'total_contacts': 0})
    
    for _, _, edges in time_windows:
        active_in_window = defaultdict(int)
        for e1, e2 in edges:
            active_in_window[e1] += 1
            active_in_window[e2] += 1
        
        for node, contacts in active_in_window.items():
            node_stats[node]['n_windows_active'] += 1
            node_stats[node]['total_contacts'] += contacts
    
    # Convert to node_id
    result = {}
    for idx in range(n_nodes):
        if idx in node_stats:
            result[node_list[idx]] = {
                'n_windows_active': node_stats[idx]['n_windows_active'],
                'total_contacts': node_stats[idx]['total_contacts'],
                'activity_ratio': round(node_stats[idx]['n_windows_active'] / len(time_windows), 4)
            }
        else:
            result[node_list[idx]] = {
                'n_windows_active': 0,
                'total_contacts': 0,
                'activity_ratio': 0
            }
    
    return result


def run_temporal_analysis(
    temporal_edges: Dict[int, List[Tuple[int, int]]],
    time_windows: List[Tuple[int, int, Set[Tuple[int, int]]]],
    node_list: List[int],
    n_nodes: int
) -> Tuple[dict, List[dict]]:
    """
    Run complete temporal analysis.
    
    Returns
    -------
    stats : dict
        Temporal statistics
    timeline : list
        Activity timeline for plotting
    """
    print("Running temporal analysis...")
    
    stats = compute_temporal_stats(temporal_edges, time_windows)
    timeline = compute_activity_timeline(time_windows)
    
    print(f"  Duration: {stats['duration_minutes']:.1f} minutes")
    print(f"  Time windows: {stats['n_time_windows']}")
    print(f"  Peak activity: {stats['peak_activity_edges']} edges")
    
    return stats, timeline
