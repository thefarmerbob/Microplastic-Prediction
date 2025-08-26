import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from skimage.transform import resize
import pandas as pd
from datetime import datetime, timedelta
import cv2
from PIL import Image
import imageio

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import the existing model and functions
from analysis_scripts.sa_convlstm import SA_ConvLSTM_Model
from training_scripts.sa_convlstm_microplastics import extract_data_and_clusters, create_sequences_from_data
from attention_analysis_final import (
    occlusion_sensitivity_analysis, 
    map_region_to_grid, 
    extract_geographic_coords_from_data
)

def create_yearly_forecast_data(nc_files, num_forecasts=4):
    """
    Create data for yearly forecasts by sampling from different time periods.
    """
    print(f"Creating {num_forecasts} forecast starting points from {len(nc_files)} files...")
    
    # Load all data
    all_data = extract_data_and_clusters(nc_files)
    print(f"Total data points: {len(all_data)}")
    
    # Create sequences for the entire dataset
    frame_num = 3  # Use 3 frames for sequence (model expects this)
    X_all, y_all = create_sequences_from_data(all_data, frame_num)
    
    print(f"Total sequences available: {len(X_all)}")
    
    # Sample evenly across the year
    if len(X_all) < num_forecasts:
        print(f"Warning: Only {len(X_all)} sequences available, using all of them")
        num_forecasts = len(X_all)
    
    # Create evenly spaced indices
    indices = np.linspace(0, len(X_all)-1, num_forecasts, dtype=int)
    
    forecast_data = []
    for i, idx in enumerate(indices):
        forecast_data.append({
            'sequence': X_all[idx],
            'target': y_all[idx] if idx < len(y_all) else None,
            'day': i + 1,
            'original_idx': idx
        })
    
    print(f"Created {len(forecast_data)} forecast starting points")
    return forecast_data

def run_occlusion_for_forecast(model, forecast_data_point, target_region_pixels, patch_size=4):
    """
    Run occlusion analysis for a single forecast.
    """
    # Prepare input sequence
    input_sequence = torch.FloatTensor(forecast_data_point['sequence'][np.newaxis, ...]).to(device)
    
    # Run occlusion sensitivity analysis
    sensitivity_map = occlusion_sensitivity_analysis(
        model, input_sequence, target_region_pixels, patch_size=patch_size
    )
    
    return sensitivity_map

def create_occlusion_image(sensitivity_map, target_region_pixels, day, img_size=64, threshold_percentile=80, global_vmin=None, global_vmax=None):
    """
    Create an image showing the top 20% most influential areas (threshold_percentile=80).
    """
    # Apply threshold to show top 20% most influential areas
    threshold = np.percentile(sensitivity_map, threshold_percentile)
    thresholded = sensitivity_map.copy()
    thresholded[thresholded < threshold] = 0
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot the thresholded sensitivity map with consistent color scale
    if global_vmin is not None and global_vmax is not None:
        im = ax.imshow(np.flipud(thresholded), cmap='hot', aspect='auto', vmin=global_vmin, vmax=global_vmax)
    else:
        im = ax.imshow(np.flipud(thresholded), cmap='hot', aspect='auto')
    
    ax.set_title(f'Day {day}: Top 20% Most Influential Areas\n(Occlusion Sensitivity)', fontsize=14)
    ax.axis('off')
    
    # Add target region overlay
    lat_range = target_region_pixels['lat_range']
    lon_range = target_region_pixels['lon_range']
    
    rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                       lon_range[1] - lon_range[0], 
                       lat_range[1] - lat_range[0],
                       linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Save the image
    output_dir = Path('yearly_occlusion_frames')
    output_dir.mkdir(exist_ok=True)
    
    filename = output_dir / f'occlusion_day_{day:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def create_averaged_occlusion_map(sensitivity_maps, target_region_pixels, title, filename, 
                                   img_size=64, threshold_percentile=80, global_vmin=None, global_vmax=None,
                                   scaling_method='enhanced_visibility', gamma=0.15):
    """
    Create an averaged occlusion sensitivity map from multiple sensitivity maps.
    
    Args:
        scaling_method: 'linear', 'log', 'power_law', or 'percentile'
        gamma: For power law scaling, controls the curve (lower = brighter low values)
    """
    # Apply threshold to each map and average
    thresholded_maps = []
    for sensitivity_map in sensitivity_maps:
        threshold = np.percentile(sensitivity_map, threshold_percentile)
        thresholded = sensitivity_map.copy()
        thresholded[thresholded < threshold] = 0
        thresholded_maps.append(thresholded)
    
    # Calculate average
    averaged_map = np.mean(thresholded_maps, axis=0)
    
    # Apply scaling transformation to make low values more visible
    display_map = averaged_map.copy()
    
    if scaling_method == 'enhanced_visibility':
        # Enhanced visibility: combination of power law + percentile scaling + saturation boost
        if np.any(display_map > 0):
            # First apply aggressive power law scaling to brighten low values
            display_map_nonzero = display_map[display_map > 0]
            max_val = np.max(display_map_nonzero)
            
            # Normalize, apply power law, then enhance contrast
            display_map = display_map / max_val
            display_map = np.power(display_map, gamma)  # Lower gamma = brighter low values
            
            # Apply saturation boost - stretch the range
            saturation_factor = 1.5
            display_map = np.clip(display_map * saturation_factor, 0, 1)
            
            # Use percentile-based scaling for better contrast
            display_map_nonzero = display_map[display_map > 0]
            if len(display_map_nonzero) > 0:
                p1 = np.percentile(display_map_nonzero, 1)   # Very low threshold
                p98 = np.percentile(display_map_nonzero, 98) # High threshold for saturation
                
                # Stretch contrast between these percentiles
                display_map = np.clip((display_map - p1) / (p98 - p1 + 1e-8), 0, 1)
                
                vmin_display = 0
                vmax_display = 1
            else:
                vmin_display = 0
                vmax_display = 1
        else:
            vmin_display = 0
            vmax_display = 1
    elif scaling_method == 'log' and np.any(display_map > 0):
        # Logarithmic scaling - add small epsilon to avoid log(0)
        epsilon = np.min(display_map[display_map > 0]) * 0.01
        display_map = display_map + epsilon
        display_map = np.log(display_map)
        vmin_display = np.log(global_vmin + epsilon) if global_vmin is not None else None
        vmax_display = np.log(global_vmax + epsilon) if global_vmax is not None else None
    elif scaling_method == 'power_law':
        # Power law scaling (gamma correction) - brightens low values
        if global_vmax is not None and global_vmax > 0:
            # Normalize to [0, 1], apply power law, then scale back
            display_map = display_map / global_vmax
            display_map = np.power(display_map, gamma)
            display_map = display_map * global_vmax
        vmin_display = global_vmin
        vmax_display = global_vmax
    elif scaling_method == 'percentile':
        # Use percentile-based scaling to focus on data distribution
        if np.any(display_map > 0):
            p5 = np.percentile(display_map[display_map > 0], 5)
            p95 = np.percentile(display_map[display_map > 0], 95)
            vmin_display = p5
            vmax_display = p95
        else:
            vmin_display = global_vmin
            vmax_display = global_vmax
    else:  # linear scaling
        vmin_display = global_vmin
        vmax_display = global_vmax
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot the averaged sensitivity map with enhanced visibility
    # Use a more saturated colormap for better contrast
    colormap = 'plasma'  # More saturated than 'hot', better for subtle differences
    if scaling_method == 'enhanced_visibility':
        # For enhanced visibility, use an even more contrasty colormap
        colormap = 'inferno'  # Highest contrast and saturation
    
    if vmin_display is not None and vmax_display is not None:
        im = ax.imshow(np.flipud(display_map), cmap=colormap, aspect='auto', 
                      vmin=vmin_display, vmax=vmax_display)
    else:
        im = ax.imshow(np.flipud(display_map), cmap=colormap, aspect='auto')
    
    # Add scaling method to title
    scaling_info = f" ({scaling_method.replace('_', ' ').title()} Scaling)"
    ax.set_title(f'{title}\n(Averaged Occlusion Sensitivity - Top 20% Areas){scaling_info}', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add target region overlay
    lat_range = target_region_pixels['lat_range']
    lon_range = target_region_pixels['lon_range']
    
    rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                       lon_range[1] - lon_range[0], 
                       lat_range[1] - lat_range[0],
                       linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Occlusion Sensitivity', rotation=270, labelpad=20)
    
    # Save the image
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def assign_data_to_periods(forecast_data, all_sensitivity_maps):
    """
    Assign data to different time periods (months and seasons).
    """
    # Since we don't have actual dates, we'll simulate a year's worth of data
    # by evenly distributing the forecast points across 365 days
    total_forecasts = len(forecast_data)
    
    # Create month and season assignments
    monthly_data = {i: {'maps': [], 'days': []} for i in range(1, 13)}  # 12 months
    seasonal_data = {
        'Winter': {'maps': [], 'days': []},    # Dec, Jan, Feb
        'Spring': {'maps': [], 'days': []},    # Mar, Apr, May
        'Summer': {'maps': [], 'days': []},    # Jun, Jul, Aug
        'Fall': {'maps': [], 'days': []}       # Sep, Oct, Nov
    }
    
    for i, (forecast_point, sensitivity_map) in enumerate(zip(forecast_data, all_sensitivity_maps)):
        # Calculate which day of year this represents (1-365)
        day_of_year = int((i / total_forecasts) * 365) + 1
        
        # Calculate month (1-12)
        month = min(12, max(1, int((day_of_year - 1) / 30.44) + 1))  # Average days per month
        
        # Calculate season
        if month in [12, 1, 2]:
            season = 'Winter'
        elif month in [3, 4, 5]:
            season = 'Spring'
        elif month in [6, 7, 8]:
            season = 'Summer'
        else:  # [9, 10, 11]
            season = 'Fall'
        
        # Add to monthly data
        monthly_data[month]['maps'].append(sensitivity_map)
        monthly_data[month]['days'].append(day_of_year)
        
        # Add to seasonal data
        seasonal_data[season]['maps'].append(sensitivity_map)
        seasonal_data[season]['days'].append(day_of_year)
    
    return monthly_data, seasonal_data

def create_gif_from_images(image_dir, output_filename='yearly_occlusion_analysis.gif', duration=0.1):
    """
    Create a GIF from the sequence of occlusion images.
    """
    image_files = sorted(Path(image_dir).glob('occlusion_day_*.png'))
    
    if not image_files:
        print("No images found to create GIF!")
        return None
    
    print(f"Creating GIF from {len(image_files)} images...")
    
    # Read all images and resize to common dimensions
    images = []
    target_size = None
    
    for i, img_file in enumerate(image_files):
        img = imageio.v2.imread(img_file)  # Use v2 to avoid deprecation warning
        
        # Set target size from first image
        if target_size is None:
            target_size = img.shape[:2]  # (height, width)
            print(f"Target image size: {target_size}")
        
        # Resize if needed
        if img.shape[:2] != target_size:
            # Resize using PIL for better quality
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            img = np.array(img_pil)
        
        images.append(img)
        
        if i % 50 == 0:
            print(f"  Processed {i+1}/{len(image_files)} images...")
    
    # Create GIF
    output_path = Path(output_filename)
    imageio.v2.mimsave(output_path, images, duration=duration, loop=0)
    
    print(f"GIF saved as: {output_path}")
    return output_path

def main():
    """Main function to create yearly occlusion analysis GIF."""
    
    print("="*70)
    print("YEARLY OCCLUSION SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Target region coordinates (Tsushima area)
    target_sw_lat, target_sw_lon = 34.02837, 129.11613
    target_ne_lat, target_ne_lon = 34.76456, 129.55801
    
    print(f"Target Region (Tsushima area):")
    print(f"  Southwest: ({target_sw_lat}, {target_sw_lon})")
    print(f"  Northeast: ({target_ne_lat}, {target_ne_lon})")
    print(f"\nNote: Using broader Japan region bounds (25.35753°N-36.98134°N, 118.85766°E-145.47117°E)")
    print(f"      to match sa_convlstm_microplastics.py training data")
    
    # Load geographic coordinates
    print("\nExtracting geographic coordinates...")
    try:
        full_lats, full_lons = extract_geographic_coords_from_data()
        print(f"Coordinate grid shape: {full_lats.shape} x {full_lons.shape}")
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return
    
    # Set up model configuration
    class Args:
        def __init__(self):
            self.batch_size = 1
            self.gpu_num = 1
            self.img_size = 64
            self.num_layers = 1
            self.frame_num = 3
            self.input_dim = 1
            self.hidden_dim = 32
            self.patch_size = 4
    
    args = Args()
    
    # Map target region to grid
    print("\nMapping target region to model grid...")
    target_region_pixels = map_region_to_grid(
        target_sw_lat, target_sw_lon, target_ne_lat, target_ne_lon,
        full_lats, full_lons, args.img_size
    )
    
    print(f"Target region mapped to grid: {target_region_pixels}")
    
    # Load trained model (from sa_convlstm_microplastics.py)
    model_path = 'models/sa_convlstm_japan_microplastics.pth'
    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found!")
        print("Please run sa_convlstm_microplastics.py first to generate the model weights.")
        return
    
    print("\nLoading trained SA-ConvLSTM model...")
    model = SA_ConvLSTM_Model(args)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load data files
    print("\nLoading microplastic data files...")
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    
    if len(nc_files) == 0:
        print("Error: No NetCDF files found!")
        return
    
    print(f"Found {len(nc_files)} data files")
    
    # Create yearly forecast data
    num_forecasts = min(365, len(nc_files) // 3)  # Limit based on available data
    print(f"\nCreating {num_forecasts} forecasts for yearly analysis...")
    
    forecast_data = create_yearly_forecast_data(nc_files, num_forecasts)
    
    # Process each forecast and collect sensitivity maps
    print("\nRunning occlusion analysis for each forecast...")
    
    all_sensitivity_maps = []
    for i, forecast_point in enumerate(forecast_data):
        if i % 10 == 0:
            print(f"Processing forecast {i+1}/{len(forecast_data)} (Day {forecast_point['day']})...")
        
        # Run occlusion analysis
        sensitivity_map = run_occlusion_for_forecast(
            model, forecast_point, target_region_pixels, patch_size=4
        )
        
        # Store the sensitivity map for global scaling calculation
        all_sensitivity_maps.append(sensitivity_map)
    
    # Calculate global min/max for consistent color scaling across all images
    print("\nCalculating global color scale for consistent visualization...")
    all_thresholded_maps = []
    for sensitivity_map in all_sensitivity_maps:
        threshold = np.percentile(sensitivity_map, 80)  # Top 20%
        thresholded = sensitivity_map.copy()
        thresholded[thresholded < threshold] = 0
        all_thresholded_maps.append(thresholded)
    
    global_vmin = np.min([np.min(m) for m in all_thresholded_maps if np.any(m > 0)])
    global_vmax = np.max([np.max(m) for m in all_thresholded_maps])
    
    print(f"Global color scale: {global_vmin:.6f} to {global_vmax:.6f}")
    
    # Now create images with consistent color scaling
    print("\nCreating images with consistent color scaling...")
    for i, (forecast_point, sensitivity_map) in enumerate(zip(forecast_data, all_sensitivity_maps)):
        # Create and save image with global color scale
        image_file = create_occlusion_image(
            sensitivity_map, target_region_pixels, forecast_point['day'],
            global_vmin=global_vmin, global_vmax=global_vmax
        )
    
    # Create GIF from all images
    print("\nCreating GIF from all occlusion images...")
    gif_path = create_gif_from_images(
        'yearly_occlusion_frames', 
        'yearly_occlusion_analysis.gif', 
        duration=0.1  # 10 FPS
    )
    
    # Create averaged occlusion maps
    print("\nCreating averaged occlusion maps...")
    
    # 1. Overall yearly average
    print("Creating yearly average...")
    yearly_avg_file = create_averaged_occlusion_map(
        all_sensitivity_maps, target_region_pixels, 
        "Yearly Average Occlusion Sensitivity", 
        "yearly_average_occlusion.png",
        global_vmin=global_vmin, global_vmax=global_vmax,
        scaling_method='enhanced_visibility', gamma=0.15  # Enhanced visibility for better contrast
    )
    
    # 2. Assign data to time periods
    monthly_data, seasonal_data = assign_data_to_periods(forecast_data, all_sensitivity_maps)
    
    # 3. Create monthly averages
    print("Creating monthly averages...")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    monthly_files = []
    for month_num in range(1, 13):
        if len(monthly_data[month_num]['maps']) > 0:
            month_file = create_averaged_occlusion_map(
                monthly_data[month_num]['maps'], target_region_pixels,
                f"{month_names[month_num-1]} Average Occlusion Sensitivity",
                f"monthly_average_occlusion_{month_names[month_num-1].lower()}.png",
                global_vmin=global_vmin, global_vmax=global_vmax,
                scaling_method='enhanced_visibility', gamma=0.15
            )
            monthly_files.append(month_file)
            print(f"  ✓ {month_names[month_num-1]}: {len(monthly_data[month_num]['maps'])} samples")
    
    # 4. Create seasonal averages
    print("Creating seasonal averages...")
    seasonal_files = []
    for season_name, season_data in seasonal_data.items():
        if len(season_data['maps']) > 0:
            season_file = create_averaged_occlusion_map(
                season_data['maps'], target_region_pixels,
                f"{season_name} Average Occlusion Sensitivity",
                f"seasonal_average_occlusion_{season_name.lower()}.png",
                global_vmin=global_vmin, global_vmax=global_vmax,
                scaling_method='enhanced_visibility', gamma=0.15
            )
            seasonal_files.append(season_file)
            print(f"  ✓ {season_name}: {len(season_data['maps'])} samples")
    
    print("\n" + "="*70)
    print("YEARLY OCCLUSION ANALYSIS COMPLETE")
    print("="*70)
    print(f"Generated files:")
    print(f"✓ {len(forecast_data)} individual occlusion images in 'yearly_occlusion_frames/'")
    print(f"✓ yearly_occlusion_analysis.gif")
    print(f"✓ yearly_average_occlusion.png (overall average)")
    print(f"✓ {len(monthly_files)} monthly average images")
    print(f"✓ {len(seasonal_files)} seasonal average images")
    print("\nInterpretation:")
    print("- The GIF shows how the most influential areas change throughout the year")
    print("- Averaged maps show persistent patterns over different time scales")
    print("- Brighter/hotter colors indicate areas that more strongly influence")
    print("  microplastic concentration in the Tsushima region")
    print("- The cyan dashed box shows the target region (Tsushima)")
    print("- Only the top 20% most influential areas are shown in each frame")
    print("- All visualizations use the same color scale for comparison")

if __name__ == "__main__":
    main()