"""
Animation Module
================

Creates animated visualization of temporal network evolution.
"""

import numpy as np
import os
from graph_tool.all import Graph, graph_draw, sfdp_layout
import matplotlib.pyplot as plt
from typing import List, Tuple, Set


def create_animation_frames(
    n_nodes: int,
    time_windows: List[Tuple[int, int, Set[Tuple[int, int]]]],
    output_dir: str,
    max_frames: int = 100,
    resolution: Tuple[int, int] = (1920, 1080),
    fps: int = 10
) -> str:
    """
    Create animation frames showing network evolution over time.
    
    Parameters
    ----------
    n_nodes : int
        Total number of nodes
    time_windows : list
        List of (start_time, end_time, edges) tuples
    output_dir : str
        Output directory
    max_frames : int
        Maximum number of frames to generate
    resolution : tuple
        Output resolution (width, height)
    fps : int
        Frames per second for video
        
    Returns
    -------
    video_path : str
        Path to generated MP4 video
    """
    print(f"\nGenerating animation ({min(len(time_windows), max_frames)} frames)...")
    
    frames_dir = os.path.join(output_dir, 'animation_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Limit frames
    windows_to_use = time_windows[:max_frames]
    
    # Build full graph for layout
    g_full = Graph(directed=False)
    g_full.add_vertex(n_nodes)
    all_edges = set()
    for _, _, edges in windows_to_use:
        all_edges.update(edges)
    for e1, e2 in all_edges:
        g_full.add_edge(e1, e2)
    
    # Compute stable layout
    print("  Computing layout...")
    pos = sfdp_layout(g_full, K=2.0, max_iter=200)
    
    # Compute degree for coloring
    degree_full = g_full.degree_property_map("total")
    max_degree = max(degree_full.a) if max(degree_full.a) > 0 else 1
    
    cmap = plt.cm.plasma
    min_t = windows_to_use[0][0] if windows_to_use else 0
    
    # Generate frames
    for frame_idx, (t_start, t_end, edges) in enumerate(windows_to_use):
        g = Graph(directed=False)
        g.add_vertex(n_nodes)
        for e1, e2 in edges:
            g.add_edge(e1, e2)
        
        degree = g.degree_property_map("total")
        
        # Node properties
        vcolor = g.new_vertex_property("vector<double>")
        vsize = g.new_vertex_property("double")
        
        for v in g.vertices():
            d = degree[v]
            d_full = degree_full[v]
            
            norm_d = d_full / max_degree
            rgba = list(cmap(norm_d))
            
            if d > 0:
                rgba[3] = 1.0
                rgba[0] = min(1.0, rgba[0] * 1.3)
                rgba[1] = min(1.0, rgba[1] * 1.3)
                rgba[2] = min(1.0, rgba[2] * 1.3)
                size = 12 + 33 * (d / max_degree)
            else:
                rgba[3] = 0.3
                size = 6
            
            vcolor[v] = rgba
            vsize[v] = size
        
        # Edge properties
        ecolor = g.new_edge_property("vector<double>")
        for e in g.edges():
            d1 = degree_full[e.source()]
            d2 = degree_full[e.target()]
            d_avg = (d1 + d2) / 2
            norm_d = d_avg / max_degree
            rgba = list(cmap(norm_d))
            rgba[3] = 0.85
            rgba[0] = min(1.0, rgba[0] * 1.4)
            rgba[1] = min(1.0, rgba[1] * 1.4)
            rgba[2] = min(1.0, rgba[2] * 1.4)
            ecolor[e] = rgba
        
        output_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
        
        graph_draw(g, pos=pos,
                   vertex_fill_color=vcolor,
                   vertex_size=vsize,
                   vertex_pen_width=1.0,
                   vertex_color=[1, 1, 1, 0.8],
                   edge_color=ecolor,
                   edge_pen_width=1.5,
                   output_size=resolution,
                   output=output_path,
                   bg_color=[0.05, 0.05, 0.1, 1])
        
        if (frame_idx + 1) % 20 == 0 or frame_idx == 0:
            minutes = (t_start - min_t) / 60
            print(f"    Frame {frame_idx + 1}/{len(windows_to_use)} (t={minutes:.1f} min)")
    
    # Create video with ffmpeg
    print("  Creating video...")
    video_path = os.path.join(output_dir, 'figures', 'network_animation.mp4')
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-vf', 'scale=-2:1080',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Animation saved: {video_path}")
            return video_path
        else:
            print(f"  Warning: ffmpeg failed - {result.stderr}")
            return None
    except FileNotFoundError:
        print("  Warning: ffmpeg not found, video not created")
        print(f"  Frames saved to: {frames_dir}")
        return None
