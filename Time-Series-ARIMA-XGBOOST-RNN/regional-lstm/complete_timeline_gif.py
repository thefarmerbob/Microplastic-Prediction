#!/usr/bin/env python3
"""
Complete Timeline GIF: Ground Truth + SA-ConvLSTM Predictions
=============================================================

This script creates a comprehensive animated GIF showing:
1. 40 ground truth NetCDF files (historical data)
2. 5 SA-ConvLSTM prediction days
3. DBSCAN clustering applied to predictions
4. Proper geographic coordinates for Japan region

The timeline shows the transition from actual historical data to predicted future data.

Author: Assistant
Date: 2024
"""

import os
os.environ['MallocStackLogging'] = '0'
os.environ['MALLOC_STACK_LOGGING'] = '0'

import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.cluster import DBSCAN
from pathlib import Path
import imageio.v2 as imageio
from scipy.spatial import ConvexHull
from skimage.transform import resize
import json
from datetime import datetime, timedelta
import torch
import wandb

# Import SA-ConvLSTM functions directly
from sa_convlstm_microplastics import (
    Args, extract_data_and_clusters, extract_timestamps_from_filenames,
    forecast_future, SA_ConvLSTM_Model, lat_lon_to_indices
)

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def process_and_plot_to_array_raw(nc_file):
    """
    Convert a netCDF file to a Japan region cropped raw array (no normalization).
    Uses the same coordinates as SA-ConvLSTM for consistency.
    """
    ds = xr.open_dataset(nc_file)
    data = ds['mp_concentration']
    data_array = data.values
    data_array_2d = data_array.squeeze()
    
    # Try to get the actual lat/lon coordinates from the dataset
    lats = None
    lons = None
    
    # Check for different possible coordinate variable names
    possible_lat_names = ['lat', 'latitude', 'y', 'lat_1', 'lat_2']
    possible_lon_names = ['lon', 'longitude', 'x', 'lon_1', 'lon_2']
    
    for lat_name in possible_lat_names:
        if lat_name in ds.variables:
            lats = ds[lat_name].values
            break
    
    for lon_name in possible_lon_names:
        if lon_name in ds.variables:
            lons = ds[lon_name].values
            break
    
    # Apply Japan region cropping if coordinates are available
    if lats is not None and lons is not None:
        # Japan region coordinates (broader) - SAME AS SA-ConvLSTM
        japan_sw_lat, japan_sw_lon = 25.35753, 118.85766
        japan_ne_lat, japan_ne_lon = 36.98134, 145.47117
        
        # Convert to grid indices
        japan_sw_lat_idx, japan_sw_lon_idx = lat_lon_to_indices(japan_sw_lat, japan_sw_lon, lats, lons)
        japan_ne_lat_idx, japan_ne_lon_idx = lat_lon_to_indices(japan_ne_lat, japan_ne_lon, lats, lons)
        
        # Ensure proper ordering
        japan_lat_start = min(japan_sw_lat_idx, japan_ne_lat_idx)
        japan_lat_end = max(japan_sw_lat_idx, japan_ne_lat_idx)
        japan_lon_start = min(japan_sw_lon_idx, japan_ne_lon_idx)
        japan_lon_end = max(japan_sw_lon_idx, japan_ne_lon_idx)
        
        # Crop the data to Japan region
        data_array_2d = data_array_2d[japan_lat_start:japan_lat_end, japan_lon_start:japan_lon_end]
    else:
        print("Warning: No coordinate information found, using full dataset")
    
    # Return raw data WITHOUT normalization
    ds.close()
    return data_array_2d

def apply_dbscan_to_forecast_simple(forecast_array, date_str, threshold=0.46, eps=3, min_samples=4, is_prediction=False):
    """
    Apply DBSCAN clustering to a forecast array and create visualization.
    
    Args:
        forecast_array: 2D numpy array from SA-ConvLSTM or ground truth
        date_str: Date string for the forecast
        threshold: Concentration threshold for clustering
        eps: DBSCAN eps parameter
        min_samples: DBSCAN minimum samples parameter
        is_prediction: Boolean indicating if this is a prediction or ground truth
    """
    h, w = forecast_array.shape
    
    # Calculate average concentration
    avg_concentration = np.nanmean(forecast_array)
    
    # Create mask for clustering (only for predictions)
    if is_prediction:
        valid_data_mask = ~np.isnan(forecast_array)
        threshold_mask = forecast_array > threshold
        combined_mask = valid_data_mask & threshold_mask
        
        filtered_indices = np.where(combined_mask)
        X = np.column_stack((filtered_indices[0], filtered_indices[1]))
        
        if len(X) == 0:
            n_clusters, n_noise = 0, 0
        else:
            # Apply DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = db.labels_
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
    else:
        # No clustering for ground truth data
        n_clusters, n_noise = 0, 0
        X = np.array([])
        labels = np.array([])
    
    # Create visualization with proper geographic coordinates
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Japan region geographic bounds (SAME AS DBSCAN SA PREDICTIONS)
    lon_min, lon_max = 118.86, 145.47
    lat_min, lat_max = 25.36, 36.98
    
    # Display the forecast with geographic extent
    img = ax.imshow(forecast_array, aspect='auto', cmap='viridis', origin='lower',
                   extent=[lon_min, lon_max, lat_min, lat_max])
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, label='Microplastics Concentration', shrink=0.8)
    
    # Plot clusters if any exist (only for predictions)
    if is_prediction and len(X) > 0 and n_clusters > 0:
        # Convert cluster coordinates to geographic coordinates
        unique_labels = set(labels)
        colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, len(unique_labels))]
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                continue  # Skip noise points for now
            
            class_member_mask = labels == k
            xy = X[class_member_mask]
            
            # Convert array indices to geographic coordinates
            xy_geo = np.zeros_like(xy, dtype=float)
            xy_geo[:, 1] = lon_min + (xy[:, 1] / w) * (lon_max - lon_min)  # longitude
            xy_geo[:, 0] = lat_min + (xy[:, 0] / h) * (lat_max - lat_min)  # latitude
            
            # Calculate cluster centroid
            centroid_lon = np.mean(xy_geo[:, 1])
            centroid_lat = np.mean(xy_geo[:, 0])
            
            # Draw convex hull around clusters
            if len(xy_geo) >= 3:
                try:
                    hull = ConvexHull(xy_geo)
                    for simplex in hull.simplices:
                        ax.plot(xy_geo[simplex, 1], xy_geo[simplex, 0], 'r-', lw=2, alpha=0.8)
                except:
                    pass  # Skip if hull can't be computed
            
            # Add cluster center marker and coordinates label
            ax.plot(centroid_lon, centroid_lat, 'ro', markersize=8, markeredgecolor='white', 
                   markeredgewidth=2, alpha=0.9)
            
            # Add coordinate text next to the cluster center
            coord_text = f'C{k}\n({centroid_lat:.2f}°N,\n{centroid_lon:.2f}°E)'
            ax.annotate(coord_text, (centroid_lon, centroid_lat), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                       ha='left', va='bottom')
    
    # Add title and labels
    data_type = "PREDICTION" if is_prediction else "GROUND TRUTH"
    cluster_info = f"Clusters: {n_clusters}, Noise: {n_noise}" if is_prediction else "Historical Data"
    
    ax.set_title(f'{data_type} - {date_str}\n{cluster_info}\nJapan Region')
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    if is_prediction:
        stats_text = (f"Avg. Concentration: {avg_concentration:.4f}\n"
                     f"Threshold: {threshold}\nClusters: {n_clusters}")
    else:
        stats_text = (f"Avg. Concentration: {avg_concentration:.2e}\n"
                     f"Raw Data (Historical)")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig, n_clusters, n_noise

def create_complete_timeline_gif():
    """
    Create a comprehensive GIF showing ground truth data followed by SA-ConvLSTM predictions.
    """
    print("="*60)
    print("CREATING COMPLETE TIMELINE GIF")
    print("="*60)
    
    # Check if forecast model exists
    model_path = Path("sa_convlstm_forecast_model.pth")
    if not model_path.exists():
        print(f"Forecast model not found: {model_path}")
        print("Please run sa_convlstm_microplastics.py first to generate the model!")
        return
    
    # Load NetCDF files
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    if not nc_files:
        print("No CYGNSS data files found!")
        return
    
    print(f"Found {len(nc_files)} NetCDF files")
    
    # Extract timestamps and get the last 40 files for ground truth
    timestamps = extract_timestamps_from_filenames(nc_files)
    
    # Use the last 40 files as ground truth (same as what the model was trained on)
    last_40_files = nc_files[-40:]
    last_40_timestamps = timestamps[-40:]
    
    print(f"Using last 40 files for ground truth timeline")
    print(f"Ground truth period: {last_40_timestamps[0].strftime('%Y-%m-%d')} to {last_40_timestamps[-1].strftime('%Y-%m-%d')}")
    
    # Create args and load data for predictions
    args = Args()
    
    print("Loading Japan region data for predictions...")
    data = extract_data_and_clusters(nc_files)
    
    # Load the trained forecast model
    print("Loading trained SA-ConvLSTM model...")
    forecast_model = SA_ConvLSTM_Model(args)
    forecast_model.load_state_dict(torch.load(model_path, map_location=device))
    forecast_model.to(device)
    forecast_model.eval()
    
    print("Generating predictions...")
    
    # Initialize wandb in disabled mode to avoid logging
    os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(mode='disabled')
    
    # Generate forecasts using the actual trained model
    forecasts, forecast_dates = forecast_future(forecast_model, data, timestamps, args, num_forecast_days=5)
    
    # Finish wandb session
    wandb.finish()
    
    print(f"Generated forecasts with shape: {forecasts.shape}")
    
    # Create output directory
    output_dir = Path("complete_timeline_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process ground truth files and predictions
    all_results = []
    image_files = []
    
    print("\nProcessing ground truth files...")
    for i, (nc_file, timestamp) in enumerate(zip(last_40_files, last_40_timestamps)):
        date_str = timestamp.strftime('%Y-%m-%d')
        print(f"Processing ground truth {i+1}/40: {date_str}")
        
        # Process ground truth file
        try:
            gt_data = process_and_plot_to_array_raw(nc_file)
            
            # Create visualization (no clustering for ground truth)
            fig, n_clusters, n_noise = apply_dbscan_to_forecast_simple(
                gt_data, date_str, threshold=0.46, eps=3, min_samples=4, is_prediction=False
            )
            
            # Save the plot
            image_file = output_dir / f"timeline_{i+1:02d}_gt_{date_str}.png"
            fig.savefig(image_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            image_files.append(image_file)
            
            all_results.append({
                'frame': i + 1,
                'date': date_str,
                'type': 'ground_truth',
                'clusters': 0,  # No clustering for ground truth
                'noise_points': 0,
                'avg_concentration': float(np.nanmean(gt_data)),
                'min_concentration': float(np.nanmin(gt_data)),
                'max_concentration': float(np.nanmax(gt_data)),
                'file': str(image_file)
            })
            
        except Exception as e:
            print(f"  Error processing {nc_file.name}: {e}")
            continue
    
    print("\nProcessing prediction files...")
    for i, (forecast, date) in enumerate(zip(forecasts, forecast_dates)):
        date_str = date.strftime('%Y-%m-%d')
        frame_num = len(last_40_files) + i + 1
        print(f"Processing prediction {i+1}/5: {date_str}")
        
        # Apply DBSCAN clustering to predictions
        fig, n_clusters, n_noise = apply_dbscan_to_forecast_simple(
            forecast, date_str, threshold=0.46, eps=3, min_samples=4, is_prediction=True
        )
        
        # Save the plot
        image_file = output_dir / f"timeline_{frame_num:02d}_pred_{date_str}.png"
        fig.savefig(image_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        image_files.append(image_file)
        
        all_results.append({
            'frame': frame_num,
            'date': date_str,
            'type': 'prediction',
            'clusters': n_clusters,
            'noise_points': n_noise,
            'avg_concentration': float(np.mean(forecast)),
            'min_concentration': float(np.min(forecast)),
            'max_concentration': float(np.max(forecast)),
            'file': str(image_file)
        })
    
    # Create animated GIF
    print(f"\nCreating animated GIF from {len(image_files)} frames...")
    if image_files:
        gif_images = []
        target_shape = None
        
        for i, file in enumerate(image_files):
            img = imageio.imread(file)
            
            # Set target shape from first image
            if target_shape is None:
                target_shape = img.shape
                print(f"Target image shape: {target_shape}")
            
            # Resize if necessary to match target shape
            if img.shape != target_shape:
                print(f"Resizing image {i+1} from {img.shape} to {target_shape}")
                img = resize(img, target_shape, preserve_range=True, anti_aliasing=True)
                img = img.astype(np.uint8)
            
            gif_images.append(img)
        
        gif_path = output_dir / "complete_timeline_animation.gif"
        imageio.mimsave(gif_path, gif_images, duration=0.8)  # Slightly faster than 1 second per frame
        print(f"Animation saved: {gif_path}")
    
    # Save results summary
    summary = {
        'script_info': {
            'name': 'Complete Timeline: Ground Truth + SA-ConvLSTM Predictions',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device)
        },
        'timeline_info': {
            'ground_truth_files': len(last_40_files),
            'prediction_days': len(forecasts),
            'total_frames': len(all_results),
            'ground_truth_period': f"{last_40_timestamps[0].strftime('%Y-%m-%d')} to {last_40_timestamps[-1].strftime('%Y-%m-%d')}",
            'prediction_period': f"{forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}"
        },
        'model_info': {
            'model_file': str(model_path),
            'forecast_shape': list(forecasts.shape),
            'data_files_used': len(nc_files)
        },
        'frame_results': all_results,
        'summary_stats': {
            'total_frames': len(all_results),
            'ground_truth_frames': len([r for r in all_results if r['type'] == 'ground_truth']),
            'prediction_frames': len([r for r in all_results if r['type'] == 'prediction']),
            'avg_clusters_in_predictions': np.mean([r['clusters'] for r in all_results if r['type'] == 'prediction']),
            'total_images': len(image_files)
        }
    }
    
    summary_file = output_dir / "complete_timeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("COMPLETE TIMELINE GIF CREATED")
    print("="*60)
    print(f"✓ Processed {summary['timeline_info']['ground_truth_files']} ground truth files")
    print(f"✓ Processed {summary['timeline_info']['prediction_days']} prediction days")
    print(f"✓ Total frames: {summary['summary_stats']['total_frames']}")
    print(f"✓ Average clusters in predictions: {summary['summary_stats']['avg_clusters_in_predictions']:.1f}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Summary saved: {summary_file}")
    print(f"✓ Animation saved: complete_timeline_animation.gif")
    print("="*60)

if __name__ == "__main__":
    create_complete_timeline_gif()