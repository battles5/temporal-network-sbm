"""
Output Writer Module
====================

Handles writing results to CSV and text files.
"""

import csv
import os
import numpy as np
from typing import Dict, List
from datetime import datetime


def write_metrics_csv(
    metrics: dict,
    temporal_stats: dict,
    sbm_results: dict,
    output_path: str
):
    """Write all metrics to a CSV file."""
    
    all_metrics = {}
    
    # Static metrics
    for key, value in metrics.items():
        if not isinstance(value, (list, dict, np.ndarray)):
            all_metrics[key] = value
    
    # Temporal metrics
    for key, value in temporal_stats.items():
        if not isinstance(value, (list, dict, np.ndarray)):
            all_metrics[f'temporal_{key}'] = value
    
    # SBM metrics
    if sbm_results:
        all_metrics['sbm_n_blocks'] = sbm_results.get('n_blocks', 0)
        all_metrics['sbm_description_length'] = sbm_results.get('description_length', 0)
        if 'icl' in sbm_results:
            all_metrics['sbm_icl'] = sbm_results.get('icl', 0)
        if 'modularity' in sbm_results:
            all_metrics['sbm_modularity'] = sbm_results.get('modularity', 0)
        if 'modularity_max' in sbm_results:
            all_metrics['sbm_modularity_max'] = sbm_results.get('modularity_max', 0)
        if 'assortativity_coefficient' in sbm_results:
            all_metrics['sbm_assortativity_coefficient'] = sbm_results.get('assortativity_coefficient', 0)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in sorted(all_metrics.items()):
            writer.writerow([key, value])
    
    print(f"  Saved: {output_path}")


def write_top_nodes_csv(
    top_nodes: List[dict],
    sbm_block_assignment: np.ndarray,
    output_path: str
):
    """Write top nodes with their centrality measures."""
    
    fieldnames = ['node_id', 'node_index', 'degree', 'degree_rank',
                  'betweenness', 'betweenness_rank', 'closeness', 'closeness_rank',
                  'eigenvector', 'eigenvector_rank', 'sbm_block']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for node in top_nodes:
            row = node.copy()
            if sbm_block_assignment is not None:
                row['sbm_block'] = int(sbm_block_assignment[node['node_index']])
            else:
                row['sbm_block'] = ''
            writer.writerow(row)
    
    print(f"  Saved: {output_path}")


def write_sbm_results_csv(
    sbm_results: dict,
    block_summary: List[dict],
    output_path: str
):
    """Write SBM block information."""
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['block', 'size', 'internal_density', 'sample_nodes'])
        
        for item in block_summary:
            nodes_str = ','.join(map(str, item['nodes']))
            if item['nodes_truncated']:
                nodes_str += ',...'
            writer.writerow([item['block'], item['size'], 
                           item['internal_density'], nodes_str])
    
    print(f"  Saved: {output_path}")


def write_dynamic_sbm_csv(
    dynamic_results: dict,
    output_path: str
):
    """Write Dynamic SBM results."""
    
    # Write window results
    window_path = output_path.replace('.csv', '_windows.csv')
    with open(window_path, 'w', newline='') as f:
        if dynamic_results['window_results']:
            fieldnames = list(dynamic_results['window_results'][0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dynamic_results['window_results'])
    
    print(f"  Saved: {window_path}")
    
    # Write transition matrix
    if dynamic_results['transition_matrix'].size > 0:
        trans_path = output_path.replace('.csv', '_transitions.csv')
        np.savetxt(trans_path, dynamic_results['transition_matrix'], 
                   delimiter=',', fmt='%.4f',
                   header=','.join(f'to_block_{b}' for b in dynamic_results['block_labels']))
        print(f"  Saved: {trans_path}")
    
    # Write stability
    stability_path = output_path.replace('.csv', '_stability.csv')
    with open(stability_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['block', 'stability'])
        for block, stab in dynamic_results['block_stability'].items():
            writer.writerow([block, stab])
    
    print(f"  Saved: {stability_path}")


def write_summary(
    metrics: dict,
    temporal_stats: dict,
    top_nodes: List[dict],
    sbm_results: dict,
    dynamic_results: dict,
    input_file: str,
    output_path: str
):
    """Write human-readable summary report."""
    
    lines = []
    lines.append("=" * 60)
    lines.append("TEMPORAL NETWORK ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Dataset: {os.path.basename(input_file)}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Basic statistics
    lines.append("-" * 40)
    lines.append("NETWORK STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Nodes: {metrics['n_nodes']}")
    lines.append(f"Edges: {metrics['n_edges']}")
    lines.append(f"Density: {metrics['density']:.4f}")
    lines.append(f"Connected: {'Yes' if metrics['is_connected'] else 'No'}")
    if not metrics['is_connected']:
        lines.append(f"Components: {metrics['n_components']}")
        lines.append(f"Largest component: {metrics['largest_component_size']} nodes")
    lines.append(f"Diameter: {metrics['diameter']}")
    lines.append(f"Average path length: {metrics['avg_path_length']:.3f}")
    lines.append("")
    
    # Degree statistics
    lines.append("-" * 40)
    lines.append("DEGREE STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Mean degree: {metrics['degree_mean']:.2f}")
    lines.append(f"Std deviation: {metrics['degree_std']:.2f}")
    lines.append(f"Min degree: {metrics['degree_min']}")
    lines.append(f"Max degree: {metrics['degree_max']}")
    lines.append("")
    
    # Clustering
    lines.append("-" * 40)
    lines.append("CLUSTERING")
    lines.append("-" * 40)
    lines.append(f"Clustering coefficient: {metrics['clustering_coefficient']:.4f}")
    lines.append(f"Transitivity: {metrics['transitivity']:.4f}")
    lines.append("")
    
    # Centralization
    lines.append("-" * 40)
    lines.append("CENTRALIZATION")
    lines.append("-" * 40)
    lines.append(f"Degree centralization: {metrics['centralization_degree']:.4f}")
    lines.append(f"Betweenness centralization: {metrics['centralization_betweenness']:.4f}")
    lines.append(f"Closeness centralization: {metrics['centralization_closeness']:.4f}")
    
    # Interpretation
    centr_deg = metrics['centralization_degree']
    if centr_deg > 0.5:
        interp = "highly centralized (star-like structure)"
    elif centr_deg > 0.3:
        interp = "moderately centralized"
    elif centr_deg > 0.1:
        interp = "slightly centralized"
    else:
        interp = "decentralized (relatively uniform)"
    lines.append(f"Interpretation: Network is {interp}")
    lines.append("")
    
    # Top nodes
    lines.append("-" * 40)
    lines.append("TOP 10 NODES BY DEGREE")
    lines.append("-" * 40)
    for i, node in enumerate(top_nodes[:10]):
        lines.append(f"{i+1}. Node {node['node_id']}: degree={node['degree']:.3f}, "
                    f"betweenness={node['betweenness']:.4f}")
    lines.append("")
    
    # SBM
    if sbm_results:
        lines.append("-" * 40)
        lines.append("STOCHASTIC BLOCK MODEL")
        lines.append("-" * 40)
        lines.append(f"Optimal blocks: {sbm_results['n_blocks']}")
        lines.append(f"Description length (MDL): {sbm_results['description_length']:.2f}")
        if 'icl' in sbm_results:
            lines.append(f"ICL (Integrated Classification Likelihood): {sbm_results['icl']:.2f}")
        if 'modularity' in sbm_results:
            lines.append(f"Modularity Q: {sbm_results['modularity']:.4f}")
        if 'modularity_max' in sbm_results:
            lines.append(f"Modularity Qmax: {sbm_results['modularity_max']:.4f}")
        if 'assortativity_coefficient' in sbm_results:
            lines.append(f"Assortativity coefficient (Q/Qmax): {sbm_results['assortativity_coefficient']:.4f}")
        lines.append("Block sizes:")
        for block, size in sorted(sbm_results['block_sizes'].items()):
            density = sbm_results.get('block_densities', {}).get(block, 0)
            lines.append(f"  Block {block}: {size} nodes (density={density:.3f})")
        lines.append("")
    
    # Temporal
    if temporal_stats:
        lines.append("-" * 40)
        lines.append("TEMPORAL ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"Duration: {temporal_stats['duration_minutes']:.1f} minutes")
        lines.append(f"Unique timestamps: {temporal_stats['n_timestamps']}")
        lines.append(f"Time windows analyzed: {temporal_stats['n_time_windows']}")
        lines.append(f"Avg edges per window: {temporal_stats['avg_edges_per_window']:.1f}")
        lines.append(f"Peak activity: {temporal_stats['peak_activity_edges']} edges")
        lines.append("")
    
    # Dynamic SBM
    if dynamic_results and dynamic_results.get('n_windows_analyzed', 0) > 0:
        lines.append("-" * 40)
        lines.append("DYNAMIC SBM")
        lines.append("-" * 40)
        lines.append(f"Windows analyzed: {dynamic_results['n_windows_analyzed']}")
        
        if dynamic_results['block_stability']:
            lines.append("Block stability (probability of staying in same block):")
            for block, stab in sorted(dynamic_results['block_stability'].items()):
                lines.append(f"  Block {block}: {stab:.3f}")
        
        if dynamic_results['movers']:
            lines.append(f"Nodes that changed blocks: {len(dynamic_results['movers'])}")
            lines.append("Top 5 mobile nodes:")
            for mover in dynamic_results['movers'][:5]:
                lines.append(f"  Node {mover['node_id']}: {mover['n_block_changes']} changes")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved: {output_path}")


def write_all_outputs(
    metrics: dict,
    temporal_stats: dict,
    top_nodes: List[dict],
    sbm_results: dict,
    block_summary: List[dict],
    dynamic_results: dict,
    input_file: str,
    output_dir: str
):
    """Write all output files."""
    
    print("\nWriting output files...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics CSV
    write_metrics_csv(
        metrics, temporal_stats, sbm_results,
        os.path.join(output_dir, 'metrics.csv')
    )
    
    # Top nodes CSV
    block_assignment = sbm_results.get('block_assignment') if sbm_results else None
    write_top_nodes_csv(
        top_nodes, block_assignment,
        os.path.join(output_dir, 'top_nodes.csv')
    )
    
    # SBM results
    if sbm_results and block_summary:
        write_sbm_results_csv(
            sbm_results, block_summary,
            os.path.join(output_dir, 'sbm_results.csv')
        )
    
    # Dynamic SBM results
    if dynamic_results and dynamic_results.get('n_windows_analyzed', 0) > 0:
        write_dynamic_sbm_csv(
            dynamic_results,
            os.path.join(output_dir, 'dynamic_sbm.csv')
        )
    
    # Summary report
    write_summary(
        metrics, temporal_stats, top_nodes,
        sbm_results, dynamic_results,
        input_file,
        os.path.join(output_dir, 'summary.txt')
    )
    
    print(f"All outputs saved to {output_dir}")
