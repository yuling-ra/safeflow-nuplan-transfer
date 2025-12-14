#!/usr/bin/env python3
"""
Script to generate a 2D grid map from a track segment.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from matplotlib.path import Path as MplPath

def load_track_data(track_name, base_path="racetrack-database"):
    """
    Load track data from CSV files.

    Args:
        track_name (str): The name of the track.
        base_path (str): The base directory of the racetrack database.

    Returns:
        tuple: A tuple containing track_data (DataFrame) and raceline_data (DataFrame).
    """
    track_file = os.path.join(base_path, "tracks", f"{track_name}.csv")
    raceline_file = os.path.join(base_path, "racelines", f"{track_name}.csv")
    
    if not os.path.exists(track_file):
        raise FileNotFoundError(f"Track file not found: {track_file}")
    
    track_data = pd.read_csv(track_file, comment='#')
    track_data.columns = ['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m']
    
    raceline_data = None
    if os.path.exists(raceline_file):
        raceline_data = pd.read_csv(raceline_file, comment='#')
        raceline_data.columns = ['x_m', 'y_m']
    
    return track_data, raceline_data

def calculate_track_boundaries(track_data):
    """
    Calculate the left and right boundaries of the track.

    Args:
        track_data (DataFrame): DataFrame containing centerline and width information.

    Returns:
        tuple: A tuple of numpy arrays for the left and right boundaries.
    """
    x = track_data['x_m'].values
    y = track_data['y_m'].values
    w_right = track_data['w_tr_right_m'].values
    w_left = track_data['w_tr_left_m'].values
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    tangent_length = np.sqrt(dx**2 + dy**2)
    tangent_length[tangent_length == 0] = 1e-10
    dx_norm = dx / tangent_length
    dy_norm = dy / tangent_length
    
    nx = -dy_norm
    ny = dx_norm
    
    left_boundary_x = x + w_left * nx
    left_boundary_y = y + w_left * ny
    right_boundary_x = x - w_right * nx
    right_boundary_y = y - w_right * ny
    
    left_boundary = np.column_stack([left_boundary_x, left_boundary_y])
    right_boundary = np.column_stack([right_boundary_x, right_boundary_y])
    
    return left_boundary, right_boundary

def create_track_grid(left_boundary, right_boundary, raceline, resolution):
    """
    Create a grid map of the track from its boundaries.

    Args:
        left_boundary (np.array): Nx2 array of left boundary points.
        right_boundary (np.array): Nx2 array of right boundary points.
        raceline (np.array): Mx2 array of raceline points.
        resolution (float): The resolution of the grid in pixels per meter.

    Returns:
        tuple: A tuple containing the grid_map, raceline_grid coordinates,
               and the translation_offset used to normalize coordinates.
    """
    all_points = np.vstack([left_boundary, right_boundary, raceline]) if raceline.size > 0 else np.vstack([left_boundary, right_boundary])
    
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    
    left_boundary_norm = left_boundary - min_coords
    right_boundary_norm = right_boundary - min_coords
    raceline_norm = raceline - min_coords
    
    dims = max_coords - min_coords
    
    grid_width = int(np.ceil(dims[0] * resolution))
    grid_height = int(np.ceil(dims[1] * resolution))
    
    grid_map = np.ones((grid_height, grid_width), dtype=np.uint8)
    
    track_polygon_verts = np.vstack([left_boundary_norm, right_boundary_norm[::-1]]) * resolution
    
    x_coords, y_coords = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
    grid_points = np.vstack([x_coords.ravel(), y_coords.ravel()]).T
    
    path = MplPath(track_polygon_verts)
    inside = path.contains_points(grid_points)
    inside_mask = inside.reshape((grid_height, grid_width))
    
    grid_map[inside_mask] = 0
    
    raceline_grid = np.round(raceline_norm * resolution).astype(int)
    if raceline_grid.size > 0:
        raceline_grid[:, 0] = np.clip(raceline_grid[:, 0], 0, grid_width - 1)
        raceline_grid[:, 1] = np.clip(raceline_grid[:, 1], 0, grid_height - 1)
    
    return grid_map, raceline_grid, min_coords

def main():
    parser = argparse.ArgumentParser(description='Generate a 2D grid map from a track segment.')
    parser.add_argument('--track', type=str, default='Nuerburgring', help='Track name')
    parser.add_argument('--segment', type=float, nargs=2, default=[0.27, 0.47], help='Start and end percentages of the track segment')
    parser.add_argument('--width-multiplier', type=float, default=5.0, help='Factor to expand track width')
    parser.add_argument('--resolution', type=float, default=1.0, help='Grid resolution (pixels per meter)')
    parser.add_argument('--output-file', type=str, default='nuerburgring_segment_map.npz', help='Output file for the map data (.npz)')
    parser.add_argument('--visualization-file', type=str, default='nuerburgring_segment_map.png', help='Output file for the visualization (.png)')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot')
    
    args = parser.parse_args()

    try:
        track_data, raceline_data = load_track_data(args.track)
        
        start_percent, end_percent = args.segment
        track_len = len(track_data)
        start_idx = int(track_len * start_percent)
        end_idx = int(track_len * end_percent)
        track_data_segment = track_data.iloc[start_idx:end_idx]

        raceline_points = np.array([])
        if raceline_data is not None:
            raceline_len = len(raceline_data)
            start_idx_raceline = int(raceline_len * start_percent)
            end_idx_raceline = int(raceline_len * end_percent)
            raceline_data_segment = raceline_data.iloc[start_idx_raceline:end_idx_raceline]
            raceline_points = raceline_data_segment[['x_m', 'y_m']].values
        else:
            print("Warning: No raceline data found. Using centerline as fallback.")
            raceline_points = track_data_segment[['x_m', 'y_m']].values

        track_data_expanded = track_data_segment.copy()
        track_data_expanded['w_tr_right_m'] *= args.width_multiplier
        track_data_expanded['w_tr_left_m'] *= args.width_multiplier
        
        left_boundary, right_boundary = calculate_track_boundaries(track_data_expanded)
        
        print("Creating grid map...")
        grid_map, raceline_grid, translation_offset = create_track_grid(left_boundary, right_boundary, raceline_points, args.resolution)
        
        print(f"Saving map data to {args.output_file}...")
        np.savez(
            args.output_file,
            grid_map=grid_map,
            raceline_grid=raceline_grid,
            resolution=args.resolution,
            translation_offset=translation_offset,
        )

        print(f"Saving visualization to {args.visualization_file}...")
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid_map, cmap='gray', origin='lower')
        
        if raceline_grid.shape[0] > 0:
            ax.plot(raceline_grid[:, 0], raceline_grid[:, 1], 'r-', linewidth=2, label='Raceline')

        ax.set_title(f'{args.track} Segment ({start_percent:.0%}-{end_percent:.0%}) - Grid Map')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(args.visualization_file, dpi=300)
        
        if not args.no_show:
            plt.show()

        print("Done.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main() 