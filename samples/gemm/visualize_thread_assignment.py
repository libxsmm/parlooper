#!/usr/bin/env python3
"""
Visualize thread work assignments as 2D heatmaps.

This script reads a file with thread work assignments in the format:
"Thread T, M-block M, N-block N, K-layer K"

and creates 2D images where:
- Each pixel position (M, N) is colored based on Thread ID (T)
- Each pixel displays the thread number
- Separate images are created for each K-layer
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patheffects
import argparse


def parse_file(filename):
    """
    Parse the input file and extract thread assignments.
    
    Returns:
        list of tuples: [(thread_id, m_block, n_block, k_layer), ...]
    """
    pattern = r'Thread\s+(\d+),\s+M-block\s+(\d+),\s+N-block\s+(\d+),\s+K-layer\s+(\d+)'
    assignments = []
    
    with open(filename, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                thread_id = int(match.group(1))
                m_block = int(match.group(2))
                n_block = int(match.group(3))
                k_layer = int(match.group(4))
                assignments.append((thread_id, m_block, n_block, k_layer))
    
    return assignments


def infer_dimensions(assignments):
    """
    Infer the dimensions and parameters from the data.
    
    Returns:
        dict: {'max_thread': int, 'max_m': int, 'max_n': int, 'max_k': int}
    """
    if not assignments:
        raise ValueError("No valid assignments found in the file")
    
    max_thread = max(a[0] for a in assignments)
    max_m = max(a[1] for a in assignments)
    max_n = max(a[2] for a in assignments)
    max_k = max(a[3] for a in assignments)
    
    return {
        'max_thread': max_thread,
        'max_m': max_m,
        'max_n': max_n,
        'max_k': max_k
    }


def create_grids(assignments, dims):
    """
    Create 2D grids for each K-layer.
    
    Returns:
        list of numpy arrays: One grid per K-layer with thread IDs at each position
    """
    num_k_layers = dims['max_k'] + 1
    m_size = dims['max_m'] + 1
    n_size = dims['max_n'] + 1
    
    # Initialize grids with -1 (indicating unassigned)
    grids = [np.full((m_size, n_size), -1, dtype=int) for _ in range(num_k_layers)]
    
    # Fill in the thread assignments
    for thread_id, m_block, n_block, k_layer in assignments:
        grids[k_layer][m_block, n_block] = thread_id
    
    return grids


def plot_grids(grids, dims, output_prefix='thread_assignment', show_numbers=True):
    """
    Plot each K-layer as a separate heatmap.
    
    Args:
        grids: List of 2D numpy arrays with thread IDs
        dims: Dictionary with dimension information
        output_prefix: Prefix for output image files
        show_numbers: Whether to display thread numbers in each cell
    """
    num_threads = dims['max_thread'] + 1
    num_k_layers = dims['max_k'] + 1
    
    # Create a colormap with distinct colors for each thread
    # Using tab20, tab20b, tab20c for up to 60 distinct colors
    # For more threads, we'll use a continuous colormap
    if num_threads <= 20:
        cmap = plt.colormaps.get_cmap('tab20').resampled(num_threads)
    elif num_threads <= 40:
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.tab20b(np.linspace(0, 1, min(20, num_threads - 20)))
        colors = np.vstack([colors1, colors2])
        cmap = ListedColormap(colors)
    elif num_threads <= 60:
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.tab20b(np.linspace(0, 1, 20))
        colors3 = plt.cm.tab20c(np.linspace(0, 1, min(20, num_threads - 40)))
        colors = np.vstack([colors1, colors2, colors3])
        cmap = ListedColormap(colors)
    else:
        cmap = plt.colormaps.get_cmap('nipy_spectral').resampled(num_threads)
    
    for k_layer in range(num_k_layers):
        grid = grids[k_layer]
        m_size, n_size = grid.shape
        
        # Create figure with appropriate size
        fig_width = max(12, n_size * 0.3)
        fig_height = max(10, m_size * 0.3)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create a masked array to handle unassigned cells
        masked_grid = np.ma.masked_where(grid == -1, grid)
        
        # Plot the heatmap
        im = ax.imshow(masked_grid, cmap=cmap, vmin=0, vmax=num_threads - 1, 
                      aspect='auto', interpolation='nearest')
        
        # Add thread numbers to each cell if requested
        if show_numbers:
            # Determine font size based on grid size
            font_size = max(4, min(10, 300 / max(m_size, n_size)))
            
            for i in range(m_size):
                for j in range(n_size):
                    if grid[i, j] != -1:
                        text = ax.text(j, i, str(grid[i, j]),
                                     ha="center", va="center",
                                     color="white", fontsize=font_size,
                                     weight="bold")
                        text.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])
        
        # Customize the plot
        ax.set_xlabel('N-block', fontsize=12)
        ax.set_ylabel('M-block', fontsize=12)
        ax.set_title(f'Thread Assignment - K-layer {k_layer}\n'
                    f'Grid size: {m_size} × {n_size}, Threads: 0-{dims["max_thread"]}',
                    fontsize=14, weight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Thread ID', rotation=270, labelpad=20, fontsize=12)
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, n_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, m_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
        
        # Set major ticks
        tick_interval = max(1, max(m_size, n_size) // 20)
        ax.set_xticks(np.arange(0, n_size, tick_interval))
        ax.set_yticks(np.arange(0, m_size, tick_interval))
        
        plt.tight_layout()
        
        # Save the figure
        output_filename = f'{output_prefix}_k{k_layer}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_filename}')
        
        plt.close()
    
    print(f'\nGenerated {num_k_layers} images for K-layers 0 to {dims["max_k"]}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize thread work assignments as 2D heatmaps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python visualize_thread_assignment.py thread_work_assignment.txt
    python visualize_thread_assignment.py input.txt -o output --no-numbers
        """
    )
    parser.add_argument('input_file', help='Input file with thread assignments')
    parser.add_argument('-o', '--output', default='thread_assignment',
                       help='Output filename prefix (default: thread_assignment)')
    parser.add_argument('--no-numbers', action='store_true',
                       help='Do not display thread numbers in cells')
    
    args = parser.parse_args()
    
    print(f'Reading file: {args.input_file}')
    assignments = parse_file(args.input_file)
    print(f'Found {len(assignments)} thread assignments')
    
    dims = infer_dimensions(assignments)
    print(f'\nInferred dimensions:')
    print(f'  M-blocks: 0 to {dims["max_m"]} ({dims["max_m"] + 1} total)')
    print(f'  N-blocks: 0 to {dims["max_n"]} ({dims["max_n"] + 1} total)')
    print(f'  K-layers: 0 to {dims["max_k"]} ({dims["max_k"] + 1} total)')
    print(f'  Threads:  0 to {dims["max_thread"]} ({dims["max_thread"] + 1} total)')
    
    grids = create_grids(assignments, dims)
    print(f'\nCreated {len(grids)} grids')
    
    print(f'\nGenerating visualizations...')
    plot_grids(grids, dims, output_prefix=args.output, 
              show_numbers=not args.no_numbers)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
