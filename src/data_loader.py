"""
Data Loader Module
==================

Handles loading of temporal network data from various file formats.
Expected input format: edge list with timestamps (t, node1, node2)
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List, Set, Optional
import os


def detect_separator(filepath: str) -> str:
    """Auto-detect the separator used in the file."""
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        
    if '\t' in first_line:
        return '\t'
    elif ',' in first_line:
        return ','
    elif ';' in first_line:
        return ';'
    else:
        return None  # whitespace


def load_temporal_edges(
    filepath: str,
    separator: str = "auto",
    skip_header: bool = False
) -> Tuple[Dict[int, List[Tuple[int, int]]], List[int], Dict[int, int], int]:
    """
    Load temporal edge data from a file.
    
    Parameters
    ----------
    filepath : str
        Path to the input file
    separator : str
        Column separator ('auto', 'space', 'tab', 'comma', 'semicolon')
    skip_header : bool
        Whether to skip the first line
        
    Returns
    -------
    temporal_edges : dict
        Dictionary mapping timestamp -> list of (node1, node2) edges
    node_list : list
        Sorted list of all unique node IDs
    node_to_idx : dict
        Mapping from original node ID to index (0, 1, 2, ...)
    n_nodes : int
        Total number of unique nodes
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    # Detect separator
    if separator == "auto":
        sep = detect_separator(filepath)
    elif separator == "tab":
        sep = '\t'
    elif separator == "comma":
        sep = ','
    elif separator == "semicolon":
        sep = ';'
    else:
        sep = None  # whitespace
    
    temporal_edges = defaultdict(list)
    all_nodes = set()
    
    with open(filepath, 'r') as f:
        if skip_header:
            f.readline()
        
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                if sep:
                    parts = line.split(sep)
                else:
                    parts = line.split()
                
                if len(parts) < 3:
                    print(f"Warning: Line {line_num} has fewer than 3 columns, skipping")
                    continue
                
                t = int(parts[0])
                n1 = int(parts[1])
                n2 = int(parts[2])
                
                if n1 != n2:  # Skip self-loops
                    temporal_edges[t].append((n1, n2))
                    all_nodes.add(n1)
                    all_nodes.add(n2)
                    
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num}: {line}")
                continue
    
    if not temporal_edges:
        raise ValueError("No valid edges found in the input file")
    
    # Create node mapping
    node_list = sorted(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)
    
    print(f"Loaded {sum(len(e) for e in temporal_edges.values())} temporal edges")
    print(f"Unique nodes: {n_nodes}")
    print(f"Unique timestamps: {len(temporal_edges)}")
    print(f"Time range: {min(temporal_edges.keys())} - {max(temporal_edges.keys())}")
    
    return temporal_edges, node_list, node_to_idx, n_nodes


def aggregate_to_static(
    temporal_edges: Dict[int, List[Tuple[int, int]]],
    node_to_idx: Dict[int, int]
) -> Set[Tuple[int, int]]:
    """
    Aggregate all temporal edges into a static edge set.
    
    Returns edges as (idx1, idx2) where idx1 < idx2 (undirected).
    """
    static_edges = set()
    
    for t, edges in temporal_edges.items():
        for n1, n2 in edges:
            i1, i2 = node_to_idx[n1], node_to_idx[n2]
            edge = (min(i1, i2), max(i1, i2))
            static_edges.add(edge)
    
    return static_edges


def create_time_windows(
    temporal_edges: Dict[int, List[Tuple[int, int]]],
    node_to_idx: Dict[int, int],
    window_size: int,
    window_step: int
) -> List[Tuple[int, int, Set[Tuple[int, int]]]]:
    """
    Create time windows for dynamic analysis.
    
    Parameters
    ----------
    temporal_edges : dict
        Dictionary mapping timestamp -> list of edges
    node_to_idx : dict
        Node ID to index mapping
    window_size : int
        Size of each window in timestamp units
    window_step : int
        Step between consecutive windows
        
    Returns
    -------
    windows : list of tuples
        Each tuple is (start_time, end_time, set of edges in window)
    """
    timestamps = sorted(temporal_edges.keys())
    min_t, max_t = timestamps[0], timestamps[-1]
    
    windows = []
    t = min_t
    
    while t <= max_t:
        window_edges = set()
        window_end = t + window_size
        
        for ts in timestamps:
            if t <= ts < window_end:
                for n1, n2 in temporal_edges[ts]:
                    i1, i2 = node_to_idx[n1], node_to_idx[n2]
                    edge = (min(i1, i2), max(i1, i2))
                    window_edges.add(edge)
        
        if window_edges:
            windows.append((t, window_end, window_edges))
        
        t += window_step
    
    print(f"Created {len(windows)} time windows")
    return windows


def get_dataset_info(
    temporal_edges: Dict[int, List[Tuple[int, int]]],
    n_nodes: int
) -> dict:
    """Get summary statistics about the dataset."""
    timestamps = sorted(temporal_edges.keys())
    total_edges = sum(len(e) for e in temporal_edges.values())
    
    return {
        'n_nodes': n_nodes,
        'n_temporal_edges': total_edges,
        'n_timestamps': len(timestamps),
        'time_min': timestamps[0],
        'time_max': timestamps[-1],
        'duration': timestamps[-1] - timestamps[0],
        'avg_edges_per_timestamp': total_edges / len(timestamps)
    }
