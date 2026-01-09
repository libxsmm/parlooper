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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patheffects
from mpl_toolkits.mplot3d import Axes3D
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


def plot_grids(grids, dims, output_prefix='thread_assignment', show_numbers=True, combined=True, cuboid=False, interactive=False):
    """
    Plot each K-layer as a separate heatmap.
    
    Args:
        grids: List of 2D numpy arrays with thread IDs
        dims: Dictionary with dimension information
        output_prefix: Prefix for output image files
        show_numbers: Whether to display thread numbers in each cell
        combined: If True and multiple K-layers exist, plot all in one image side by side
        cuboid: If True and multiple K-layers exist, create a 3D cuboid visualization
        interactive: If True, show interactive plot that can be rotated with mouse
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
    
    # If interactive mode is requested, use cuboid/3D visualization
    if interactive:
        plot_cuboid_grids(grids, dims, cmap, num_threads, output_prefix, interactive)
    # If cuboid mode and multiple K-layers, create 3D visualization
    elif num_k_layers > 1 and cuboid:
        plot_cuboid_grids(grids, dims, cmap, num_threads, output_prefix, interactive)
    # If multiple K-layers and combined mode, create a single figure with subplots
    elif num_k_layers > 1 and combined:
        plot_combined_grids(grids, dims, cmap, num_threads, output_prefix, show_numbers)
    else:
        plot_separate_grids(grids, dims, cmap, num_threads, output_prefix, show_numbers)


def plot_cuboid_grids(grids, dims, cmap, num_threads, output_prefix, interactive=False):
    """Plot all K-layers stacked in 3D to create a cuboid visualization."""
    num_k_layers = len(grids)
    m_size, n_size = grids[0].shape
    
    # Spacing factor between K-layers (increase for more separation)
    k_spacing = 32.0
    
    # If interactive mode, create a single interactive plot
    if interactive:
        plot_interactive_cuboid(grids, dims, cmap, num_threads, k_spacing)
        return
    
    # Create multiple views of the 3D cuboid
    elevation_angles = [30, 15, 45]
    azimuth_angles = [45, 225, 135]
    
    for view_idx, (elev, azim) in enumerate(zip(elevation_angles, azimuth_angles)):
        fig = plt.figure(figsize=(16, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each K-layer as a surface
        for k_layer in range(num_k_layers):
            grid = grids[k_layer]
            
            # Create meshgrid for the layer
            X, Y = np.meshgrid(np.arange(n_size), np.arange(m_size))
            Z = np.full_like(X, k_layer * k_spacing, dtype=float)
            
            # Normalize thread IDs for coloring
            grid_normalized = grid.astype(float) / (num_threads - 1)
            grid_normalized = np.ma.masked_where(grid == -1, grid_normalized)
            
            # Get colors from colormap
            colors = cmap(grid_normalized)
            
            # Plot the surface
            surf = ax.plot_surface(X, Y, Z, facecolors=colors, 
                                  shade=False, alpha=0.9,
                                  antialiased=True, linewidth=0)
            
            # Add wireframe for better visibility
            ax.plot_wireframe(X, Y, Z, color='black', alpha=0.1, linewidth=0.3)
        
        # Set labels and title
        ax.set_xlabel('N-block', fontsize=12, labelpad=10)
        ax.set_ylabel('M-block', fontsize=12, labelpad=10)
        ax.set_zlabel('K-layer', fontsize=12, labelpad=10)
        ax.set_title(f'3D Thread Assignment Cuboid\n'
                    f'Grid size: {m_size} × {n_size} × {num_k_layers}, '
                    f'Threads: 0-{dims["max_thread"]}',
                    fontsize=14, weight='bold', pad=20)
        
        # Set the viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Set axis limits
        ax.set_xlim(0, n_size - 1)
        ax.set_ylim(0, m_size - 1)
        ax.set_zlim(0, (num_k_layers - 1) * k_spacing)
        
        # Set aspect ratio to honor M and N dimensions
        ax.set_box_aspect([n_size, m_size, (num_k_layers - 1) * k_spacing if num_k_layers > 1 else 1])
        
        # Add colorbar
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(np.arange(num_threads))
        mappable.set_clim(0, num_threads - 1)
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
        cbar.set_label('Thread ID', rotation=270, labelpad=20, fontsize=12)
        
        # Customize grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_filename = f'{output_prefix}_cuboid_view{view_idx+1}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f'Saved 3D cuboid view {view_idx+1}: {output_filename}')
        
        plt.close()
    
    # Create an interactive rotating view (animated GIF if imageio is available)
    try:
        create_rotating_cuboid(grids, dims, cmap, num_threads, output_prefix)
    except ImportError:
        print('Note: Install imageio for animated rotating view: pip install imageio')


def plot_interactive_cuboid_plotly(grids, dims, cmap, num_threads, k_spacing):
    """Create an interactive 3D cuboid using plotly (works in browsers)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    num_k_layers = len(grids)
    m_size, n_size = grids[0].shape
    
    # Create figure
    fig = go.Figure()
    
    # Collect all vertices, faces, colors, and hover data
    all_vertices_x = []
    all_vertices_y = []
    all_vertices_z = []
    all_i = []
    all_j = []
    all_k = []
    all_facecolors = []
    all_hovertext = []
    
    vertex_offset = 0
    
    # Plot each K-layer using individual rectangles (quads as two triangles)
    for k_layer in range(num_k_layers):
        grid = grids[k_layer]
        z_level = k_layer * k_spacing
        
        # For each cell in the grid, create a rectangle (two triangles)
        for m_idx in range(m_size):
            for n_idx in range(n_size):
                thread_id = grid[m_idx, n_idx]
                if thread_id == -1:
                    continue  # Skip unassigned cells
                
                # Get color for this thread
                rgba = cmap(thread_id / (num_threads - 1))
                color_str = f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
                
                # Create hover text
                hover_text = f'<b>N-block:</b> {n_idx}<br><b>M-block:</b> {m_idx}<br><b>K-layer:</b> {k_layer}<br><b>Thread ID:</b> {thread_id}'
                
                # Define the 4 corners of this cell rectangle
                # Bottom-left, bottom-right, top-right, top-left
                corners_n = [n_idx, n_idx + 1, n_idx + 1, n_idx]
                corners_m = [m_idx, m_idx, m_idx + 1, m_idx + 1]
                
                # Add vertices
                for cn, cm in zip(corners_n, corners_m):
                    all_vertices_x.append(cn - 0.5)  # Center the cells
                    all_vertices_y.append(cm - 0.5)
                    all_vertices_z.append(z_level)
                
                # Create two triangles for this rectangle
                # Triangle 1: vertices 0, 1, 2
                all_i.append(vertex_offset + 0)
                all_j.append(vertex_offset + 1)
                all_k.append(vertex_offset + 2)
                all_facecolors.append(color_str)
                all_hovertext.append(hover_text)
                
                # Triangle 2: vertices 0, 2, 3
                all_i.append(vertex_offset + 0)
                all_j.append(vertex_offset + 2)
                all_k.append(vertex_offset + 3)
                all_facecolors.append(color_str)
                all_hovertext.append(hover_text)
                
                vertex_offset += 4
    
    # Create single Mesh3d for all cells
    fig.add_trace(go.Mesh3d(
        x=all_vertices_x,
        y=all_vertices_y,
        z=all_vertices_z,
        i=all_i,
        j=all_j,
        k=all_k,
        facecolor=all_facecolors,
        text=all_hovertext,
        hoverinfo='text',
        showlegend=False,
        lighting=dict(ambient=0.8, diffuse=0.8, specular=0.1),
        flatshading=True
    ))
    
    # Add wireframe grid lines for each layer
    for k_layer in range(num_k_layers):
        z_level = k_layer * k_spacing
        
        # Grid lines along N direction
        for m_idx in range(m_size + 1):
            fig.add_trace(go.Scatter3d(
                x=[0 - 0.5, n_size - 0.5],
                y=[m_idx - 0.5, m_idx - 0.5],
                z=[z_level, z_level],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Grid lines along M direction
        for n_idx in range(n_size + 1):
            fig.add_trace(go.Scatter3d(
                x=[n_idx - 0.5, n_idx - 0.5],
                y=[0 - 0.5, m_size - 0.5],
                z=[z_level, z_level],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interactive 3D Thread Assignment Cuboid<br>' +
                 f'Grid size: {m_size} × {n_size} × {num_k_layers}, Threads: 0-{dims["max_thread"]}<br>' +
                 f'<sub>Click and drag to rotate | Scroll to zoom | Double-click to reset</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='N-block', range=[0, n_size - 1]),
            yaxis=dict(title='M-block', range=[0, m_size - 1]),
            zaxis=dict(title='K-layer', range=[0, (num_k_layers - 1) * k_spacing]),
            aspectmode='manual',
            aspectratio=dict(
                x=n_size / max(m_size, n_size, (num_k_layers - 1) * k_spacing if num_k_layers > 1 else 1),
                y=m_size / max(m_size, n_size, (num_k_layers - 1) * k_spacing if num_k_layers > 1 else 1),
                z=((num_k_layers - 1) * k_spacing if num_k_layers > 1 else 1) / max(m_size, n_size, (num_k_layers - 1) * k_spacing if num_k_layers > 1 else 1)
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1200,
        height=900,
        showlegend=False
    )
    
    # Save to HTML file
    output_file = 'thread_assignment_interactive.html'
    fig.write_html(output_file)
    print(f'\n✓ Saved interactive 3D visualization: {output_file}')
    print(f'  Open this file in a web browser to interact with the 3D cuboid.')
    print(f'  Controls: Click+drag to rotate, scroll to zoom, double-click to reset view.\n')
    
    return True


def plot_interactive_cuboid(grids, dims, cmap, num_threads, k_spacing):
    """Create an interactive 3D cuboid that can be rotated with the mouse."""
    # Try to use plotly for better interactive support (works in browsers)
    try:
        return plot_interactive_cuboid_plotly(grids, dims, cmap, num_threads, k_spacing)
    except ImportError:
        print('Note: plotly not installed. Falling back to matplotlib.')
        print('For better interactive visualization, install plotly: pip install plotly')
        pass
    
    num_k_layers = len(grids)
    m_size, n_size = grids[0].shape
    
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each K-layer as a surface
    for k_layer in range(num_k_layers):
        grid = grids[k_layer]
        
        # Create meshgrid for the layer
        X, Y = np.meshgrid(np.arange(n_size), np.arange(m_size))
        Z = np.full_like(X, k_layer * k_spacing, dtype=float)
        
        # Normalize thread IDs for coloring
        grid_normalized = grid.astype(float) / (num_threads - 1)
        grid_normalized = np.ma.masked_where(grid == -1, grid_normalized)
        
        # Get colors from colormap
        colors = cmap(grid_normalized)
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, 
                              shade=False, alpha=0.9,
                              antialiased=True, linewidth=0)
        
        # Add wireframe for better visibility
        ax.plot_wireframe(X, Y, Z, color='black', alpha=0.1, linewidth=0.3)
    
    # Set labels and title
    ax.set_xlabel('N-block', fontsize=12, labelpad=10)
    ax.set_ylabel('M-block', fontsize=12, labelpad=10)
    ax.set_zlabel('K-layer', fontsize=12, labelpad=10)
    ax.set_title(f'Interactive 3D Thread Assignment Cuboid\n'
                f'Grid size: {m_size} × {n_size} × {num_k_layers}, '
                f'Threads: 0-{dims["max_thread"]}\n'
                f'(Click and drag to rotate)',
                fontsize=14, weight='bold', pad=20)
    
    # Set the initial viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Set axis limits
    ax.set_xlim(0, n_size - 1)
    ax.set_ylim(0, m_size - 1)
    ax.set_zlim(0, (num_k_layers - 1) * k_spacing)
    
    # Set aspect ratio to honor M and N dimensions
    ax.set_box_aspect([n_size, m_size, (num_k_layers - 1) * k_spacing if num_k_layers > 1 else 1])
    
    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(np.arange(num_threads))
    mappable.set_clim(0, num_threads - 1)
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
    cbar.set_label('Thread ID', rotation=270, labelpad=20, fontsize=12)
    
    # Customize grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print('\nDisplaying interactive 3D cuboid...')
    print('Use your mouse to rotate the view:')
    print('  - Left click + drag: Rotate')
    print('  - Right click + drag: Pan')
    print('  - Scroll wheel: Zoom')
    print('Close the window to continue...\n')
    
    # Check if we can actually display
    import os
    if 'DISPLAY' not in os.environ and matplotlib.get_backend() not in ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg']:
        print('WARNING: No display available!')
        print('You are likely running on a remote server without X11 forwarding.')
        print('To view the interactive plot, you need to:')
        print('  1. Enable X11 forwarding: ssh -X user@server')
        print('  2. Or use a local machine with a display')
        print('  3. Or remove the --interactive flag to generate static images instead\n')
    
    try:
        plt.show()
    except Exception as e:
        print(f'Error displaying interactive plot: {e}')
        print('Falling back to saving a static image...')
        plt.savefig('interactive_cuboid_fallback.png', dpi=150, bbox_inches='tight')
        print('Saved: interactive_cuboid_fallback.png')
        plt.close()


def create_rotating_cuboid(grids, dims, cmap, num_threads, output_prefix):
    """Create a rotating animation of the cuboid."""
    import imageio
    
    num_k_layers = len(grids)
    m_size, n_size = grids[0].shape
    
    # Spacing factor between K-layers (increase for more separation)
    k_spacing = 2.0
    
    frames = []
    num_frames = 36  # 10 degree increments
    
    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each K-layer
        for k_layer in range(num_k_layers):
            grid = grids[k_layer]
            X, Y = np.meshgrid(np.arange(n_size), np.arange(m_size))
            Z = np.full_like(X, k_layer * k_spacing, dtype=float)
            
            grid_normalized = grid.astype(float) / (num_threads - 1)
            grid_normalized = np.ma.masked_where(grid == -1, grid_normalized)
            colors = cmap(grid_normalized)
            
            surf = ax.plot_surface(X, Y, Z, facecolors=colors, 
                                  shade=False, alpha=0.9,
                                  antialiased=True, linewidth=0)
            ax.plot_wireframe(X, Y, Z, color='black', alpha=0.1, linewidth=0.3)
        
        ax.set_xlabel('N-block', fontsize=12, labelpad=10)
        ax.set_ylabel('M-block', fontsize=12, labelpad=10)
        ax.set_zlabel('K-layer', fontsize=12, labelpad=10)
        ax.set_title(f'3D Thread Assignment Cuboid (Rotating)\n'
                    f'Grid: {m_size}×{n_size}×{num_k_layers}, Threads: 0-{dims["max_thread"]}',
                    fontsize=14, weight='bold', pad=20)
        
        # Rotate view
        azim = frame_idx * 10
        ax.view_init(elev=25, azim=azim)
        
        ax.set_xlim(0, n_size - 1)
        ax.set_ylim(0, m_size - 1)
        ax.set_zlim(0, (num_k_layers - 1) * k_spacing)
        
        # Set aspect ratio to honor M and N dimensions
        ax.set_box_aspect([n_size, m_size, (num_k_layers - 1) * k_spacing if num_k_layers > 1 else 1])
        
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(np.arange(num_threads))
        mappable.set_clim(0, num_threads - 1)
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.1, shrink=0.8)
        cbar.set_label('Thread ID', rotation=270, labelpad=20, fontsize=12)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to image
        # Draw the canvas to ensure it's rendered
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape((height, width, 4))
        
        # Convert RGBA to RGB
        image = image[:, :, :3]
        frames.append(image)
        
        plt.close()
    
    # Save as animated GIF
    output_filename = f'{output_prefix}_cuboid_rotating.gif'
    imageio.mimsave(output_filename, frames, fps=10, loop=0)
    print(f'Saved rotating animation: {output_filename}')


def plot_combined_grids(grids, dims, cmap, num_threads, output_prefix, show_numbers):
    """Plot all K-layers side by side in a single image."""
    num_k_layers = len(grids)
    m_size, n_size = grids[0].shape
    
    # Create figure with subplots arranged horizontally
    fig_width = max(16, num_k_layers * n_size * 0.25)
    fig_height = max(8, m_size * 0.25)
    
    fig, axes = plt.subplots(1, num_k_layers, figsize=(fig_width, fig_height))
    
    # Ensure axes is iterable even for single subplot
    if num_k_layers == 1:
        axes = [axes]
    
    # Determine font size based on grid size
    font_size = max(3, min(10, 250 / max(m_size, n_size)))
    
    for k_layer, (grid, ax) in enumerate(zip(grids, axes)):
        # Create a masked array to handle unassigned cells
        masked_grid = np.ma.masked_where(grid == -1, grid)
        
        # Plot the heatmap
        im = ax.imshow(masked_grid, cmap=cmap, vmin=0, vmax=num_threads - 1, 
                      aspect='auto', interpolation='nearest')
        
        # Add thread numbers to each cell if requested
        if show_numbers and m_size <= 64 and n_size <= 64:  # Only for reasonable sizes
            for i in range(m_size):
                for j in range(n_size):
                    if grid[i, j] != -1:
                        text = ax.text(j, i, str(grid[i, j]),
                                     ha="center", va="center",
                                     color="white", fontsize=font_size,
                                     weight="bold")
                        text.set_path_effects([patheffects.withStroke(linewidth=0.5, foreground='black')])
        
        # Customize each subplot
        ax.set_xlabel('N-block', fontsize=10)
        ax.set_ylabel('M-block', fontsize=10)
        ax.set_title(f'K-layer {k_layer}', fontsize=12, weight='bold')
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, n_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, m_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.1)
        
        # Set major ticks
        tick_interval = max(1, max(m_size, n_size) // 10)
        ax.set_xticks(np.arange(0, n_size, tick_interval))
        ax.set_yticks(np.arange(0, m_size, tick_interval))
        ax.tick_params(labelsize=8)
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Thread ID', rotation=270, labelpad=15, fontsize=11)
    
    # Add overall title
    fig.suptitle(f'Thread Assignment - All K-layers\n'
                f'Grid size: {m_size} × {n_size}, Threads: 0-{dims["max_thread"]}',
                fontsize=14, weight='bold', y=0.98)
    
    # Use constrained_layout compatible approach or suppress the warning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        plt.tight_layout(rect=[0, 0, 0.93, 0.95])
    
    # Save the combined figure
    output_filename = f'{output_prefix}_combined.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f'Saved combined image: {output_filename}')
    
    plt.close()


def plot_separate_grids(grids, dims, cmap, num_threads, output_prefix, show_numbers):
    """Plot each K-layer as a separate image."""
    num_k_layers = len(grids)
def plot_separate_grids(grids, dims, cmap, num_threads, output_prefix, show_numbers):
    """Plot each K-layer as a separate image."""
    num_k_layers = len(grids)
    
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
    parser.add_argument('--separate', action='store_true',
                       help='Generate separate images instead of combined view for multiple K-layers')
    parser.add_argument('--cuboid', action='store_true',
                       help='Generate 3D cuboid visualization with stacked K-layers')
    parser.add_argument('--interactive', action='store_true',
                       help='Show interactive 3D plot that can be rotated with mouse (requires --cuboid)')
    
    args = parser.parse_args()
    
    # Set interactive backend if requested
    if args.interactive:
        # Try to use an interactive backend
        try:
            matplotlib.use('TkAgg')
        except:
            try:
                matplotlib.use('Qt5Agg')
            except:
                try:
                    matplotlib.use('GTK3Agg')
                except:
                    print('Warning: Could not set interactive backend. Interactive mode may not work.')
                    print('You may need to set DISPLAY variable or install a GUI backend (tkinter, PyQt5, etc.)')
    
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
              show_numbers=not args.no_numbers,
              combined=not args.separate,
              cuboid=args.cuboid,
              interactive=args.interactive)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
