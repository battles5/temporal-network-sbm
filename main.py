#!/usr/bin/env python3
"""
Temporal Network SBM Analysis
=============================

Main entry point for analyzing temporal networks using Stochastic Block Models.

Usage:
    python main.py --input <data_file> --output <output_dir> [--config <config.yaml>]
    
Example:
    python main.py --input data/tij_LyonSchool.dat --output output/

Author: Orso Peruzzi
"""

import argparse
import os
import sys
import yaml
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_temporal_edges, 
    aggregate_to_static, 
    create_time_windows,
    get_dataset_info
)
from src.static_analysis import run_static_analysis
from src.temporal_analysis import run_temporal_analysis
from src.sbm import run_sbm_analysis
from src.dynamic_sbm import run_dynamic_sbm
from src.visualization import create_all_visualizations, create_hypergraph_visualizations
from src.output_writer import write_all_outputs
from src.animation import create_animation_frames
from src.temporal_hypergraph import run_hypergraph_analysis, write_hypergraph_outputs


def load_config(config_path: Optional[str]) -> dict:
    """Load configuration from YAML file or use defaults."""
    
    defaults = {
        'input': {
            'separator': 'auto',
            'skip_header': False
        },
        'time_windows': {
            'window_size_seconds': 300,
            'window_step_seconds': 60
        },
        'sbm': {
            'max_blocks': 20,
            'equilibrate': True
        },
        'dynamic_sbm': {
            'enabled': True,
            'max_windows': 50,
            'max_blocks': 10
        },
        'visualization': {
            'dpi': 300,
            'format': 'png'
        },
        'animation': {
            'enabled': False,
            'max_frames': 100,
            'fps': 10,
            'resolution': [1920, 1080]
        },
        'hypergraph': {
            'enabled': False,
            'min_group_size': 3,
            'max_group_size': 20,
            'max_cliques_per_window': 10000
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Merge user config with defaults
        for section, values in user_config.items():
            if section in defaults:
                defaults[section].update(values)
            else:
                defaults[section] = values
    
    return defaults


def main():
    parser = argparse.ArgumentParser(
        description='Analyze temporal networks using Stochastic Block Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data.dat --output results/
  python main.py --input data.csv --output results/ --config config.yaml
  python main.py --input data.dat --output results/ --animate

For more information, see README.md
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input data file (temporal edge list)')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to output directory')
    parser.add_argument('--config', '-c', default=None,
                        help='Path to configuration YAML file')
    parser.add_argument('--animate', action='store_true',
                        help='Generate network animation (slow)')
    parser.add_argument('--no-dynamic-sbm', action='store_true',
                        help='Skip dynamic SBM analysis (faster)')
    parser.add_argument('--hypergraph', action='store_true',
                        help='Extract group interactions via clique analysis (Iacopini et al.)')
    parser.add_argument('--min-group-size', type=int, default=None,
                        help='Minimum group/clique size (default: 3)')
    parser.add_argument('--max-group-size', type=int, default=None,
                        help='Maximum group/clique size (default: 20)')
    parser.add_argument('--max-cliques-per-window', type=int, default=None,
                        help='Safety limit: max cliques per window (default: 10000)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.animate:
        config['animation']['enabled'] = True
    if args.no_dynamic_sbm:
        config['dynamic_sbm']['enabled'] = False
    if args.hypergraph:
        config['hypergraph']['enabled'] = True
    if args.min_group_size is not None:
        config['hypergraph']['min_group_size'] = args.min_group_size
    if args.max_group_size is not None:
        config['hypergraph']['max_group_size'] = args.max_group_size
    if args.max_cliques_per_window is not None:
        config['hypergraph']['max_cliques_per_window'] = args.max_cliques_per_window
    
    print("=" * 60)
    print("TEMPORAL NETWORK SBM ANALYSIS")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Load data
    print("-" * 40)
    print("LOADING DATA")
    print("-" * 40)
    
    temporal_edges, node_list, node_to_idx, n_nodes = load_temporal_edges(
        args.input,
        separator=config['input']['separator'],
        skip_header=config['input']['skip_header']
    )
    
    dataset_info = get_dataset_info(temporal_edges, n_nodes)
    print()
    
    # 2. Aggregate to static graph
    static_edges = aggregate_to_static(temporal_edges, node_to_idx)
    print(f"Static graph: {len(static_edges)} unique edges")
    
    # 3. Create time windows
    time_windows = create_time_windows(
        temporal_edges, node_to_idx,
        window_size=config['time_windows']['window_size_seconds'],
        window_step=config['time_windows']['window_step_seconds']
    )
    print()
    
    # 4. Static analysis
    print("-" * 40)
    print("STATIC NETWORK ANALYSIS")
    print("-" * 40)
    
    metrics, top_nodes, node_centrality, g = run_static_analysis(
        n_nodes, static_edges, node_list
    )
    
    # Get degrees for visualization
    degrees = g.get_out_degrees(g.get_vertices())
    print()
    
    # 5. Temporal analysis
    print("-" * 40)
    print("TEMPORAL ANALYSIS")
    print("-" * 40)
    
    temporal_stats, timeline = run_temporal_analysis(
        temporal_edges, time_windows, node_list, n_nodes
    )
    print()
    
    # 6. SBM analysis
    print("-" * 40)
    print("STOCHASTIC BLOCK MODEL")
    print("-" * 40)
    
    sbm_results, block_summary, block_assignment = run_sbm_analysis(
        g, node_list,
        max_blocks=config['sbm']['max_blocks'],
        equilibrate=config['sbm']['equilibrate']
    )
    print()
    
    # 7. Dynamic SBM (optional)
    dynamic_results = {}
    if config['dynamic_sbm']['enabled']:
        print("-" * 40)
        print("DYNAMIC STOCHASTIC BLOCK MODEL")
        print("-" * 40)
        
        # Use subset of windows for dynamic SBM
        max_dyn_windows = config['dynamic_sbm']['max_windows']
        step = max(1, len(time_windows) // max_dyn_windows)
        windows_for_dsbm = time_windows[::step][:max_dyn_windows]
        
        dynamic_results = run_dynamic_sbm(
            n_nodes, windows_for_dsbm, node_list,
            max_blocks=config['dynamic_sbm']['max_blocks']
        )
        print()
    
    # 7b. Hypergraph group extraction (optional)
    hypergraph_results = {}
    if config['hypergraph']['enabled']:
        print("-" * 40)
        print("HYPERGRAPH GROUP EXTRACTION (CLIQUES)")
        print("-" * 40)
        
        # Build time windows dict format for hypergraph analysis
        hg_time_windows = [{'t_start': w['t_start'], 't_end': w['t_end']} 
                          for w in time_windows]
        
        hypergraph_results = run_hypergraph_analysis(
            temporal_edges,
            hg_time_windows,
            min_group_size=config['hypergraph']['min_group_size'],
            max_group_size=config['hypergraph']['max_group_size'],
            max_cliques_per_window=config['hypergraph']['max_cliques_per_window'],
            verbose=True
        )
        print()
    
    # 8. Write outputs
    print("-" * 40)
    print("WRITING OUTPUTS")
    print("-" * 40)
    
    write_all_outputs(
        metrics, temporal_stats, top_nodes,
        sbm_results, block_summary, dynamic_results,
        args.input, args.output
    )
    
    # Write hypergraph outputs if enabled
    if hypergraph_results:
        write_hypergraph_outputs(hypergraph_results, args.output)
    print()
    
    # 9. Create visualizations
    print("-" * 40)
    print("CREATING VISUALIZATIONS")
    print("-" * 40)
    
    create_all_visualizations(
        g, degrees, node_centrality,
        sbm_results, dynamic_results, timeline,
        args.output,
        dpi=config['visualization']['dpi']
    )
    
    # Hypergraph visualizations
    if hypergraph_results:
        create_hypergraph_visualizations(
            hypergraph_results,
            args.output,
            dpi=config['visualization']['dpi']
        )
    print()
    
    # 10. Animation (optional)
    if config['animation']['enabled']:
        print("-" * 40)
        print("CREATING ANIMATION")
        print("-" * 40)
        
        create_animation_frames(
            n_nodes, time_windows, args.output,
            max_frames=config['animation']['max_frames'],
            resolution=tuple(config['animation']['resolution']),
            fps=config['animation']['fps']
        )
        print()
    
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output}")
    print()


if __name__ == '__main__':
    main()
