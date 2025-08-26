#!/usr/bin/env python3
"""
Show Ground Truth Data for Any Specified Date
==============================================

Simple script to display the ground truth microplastics data for any specified date
in the Japan region with proper geographic coordinates.

Usage: python show_ground_truth_date.py [YYYYMMDD]
Example: python show_ground_truth_date.py 20190501

Author: Assistant
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

def lat_lon_to_indices(lat, lon, lats, lons):
    """Convert lat/lon coordinates to grid indices"""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def find_file_for_date(nc_files, target_date_str):
    """Find the NetCDF file for the target date"""
    for nc_file in nc_files:
        filename = Path(nc_file).name
        # Extract date from filename: cyg.ddmi.s20190501 -> 20190501
        if '.s' + target_date_str in filename:
            return nc_file
    return None

def show_ground_truth_japan(target_date):
    """Show ground truth data for the specified date in Japan region"""
    
    print(f"Loading ground truth data for {target_date}")
    
    # Load NetCDF files
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    if not nc_files:
        print("No CYGNSS data files found!")
        return False
    
    # Find the specific file for our target date
    target_file = find_file_for_date(nc_files, target_date)
    if target_file is None:
        print(f"No file found for date {target_date}")
        print("\nAvailable files:")
        for f in nc_files[:20]:  # Show first 20 files
            filename = Path(f).name
            if '.s' in filename:
                date_part = filename.split('.')[2][1:9] if len(filename.split('.')) > 2 else "unknown"
                print(f"  {filename} -> Date: {date_part}")
        return False
    
    print(f"Found file: {Path(target_file).name}")
    
    # Load and process the NetCDF file
    ds = xr.open_dataset(target_file)
    data = ds['mp_concentration']
    data_array = data.values
    data_array_2d = data_array.squeeze()
    
    # Get coordinate information
    lats = None
    lons = None
    
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
    
    if lats is None or lons is None:
        print("Warning: No coordinate information found!")
        ds.close()
        return False
    
    # Crop to Japan region (same coordinates as SA-ConvLSTM)
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
    japan_data = data_array_2d[japan_lat_start:japan_lat_end, japan_lon_start:japan_lon_end]
    
    print(f"Japan region cropped to: lat[{japan_lat_start}:{japan_lat_end}], lon[{japan_lon_start}:{japan_lon_end}]")
    print(f"Japan region shape: {japan_data.shape}")
    print(f"Data range: {np.nanmin(japan_data):.2e} to {np.nanmax(japan_data):.2e}")
    print(f"Average concentration: {np.nanmean(japan_data):.2e}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Japan region geographic bounds for display
    lon_min, lon_max = 118.86, 145.47
    lat_min, lat_max = 25.36, 36.98
    
    # Flip data vertically for correct geographic orientation
    japan_data_flipped = np.flipud(japan_data)
    
    # Display the data with geographic extent
    img = ax.imshow(japan_data_flipped, cmap='viridis', aspect='auto', origin='lower',
                   extent=[lon_min, lon_max, lat_min, lat_max])
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, label='Microplastics Concentration', shrink=0.8)
    
    # Format the date for display
    date_obj = datetime.strptime(target_date, '%Y%m%d')
    formatted_date = date_obj.strftime('%B %d, %Y')
    
    # Add title and labels
    ax.set_title(f'Ground Truth Microplastics Data\n{formatted_date} - Japan Region', fontsize=16)
    ax.set_xlabel('Longitude (°E)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f"Data Statistics:\n"
                 f"Min: {np.nanmin(japan_data):.2e}\n"
                 f"Max: {np.nanmax(japan_data):.2e}\n"
                 f"Mean: {np.nanmean(japan_data):.2e}\n"
                 f"Shape: {japan_data.shape}")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'ground_truth_{target_date}_japan.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_filename}")
    
    # Don't show plot automatically to avoid hanging
    print("Plot created successfully!")
    
    # Close the dataset
    ds.close()
    return True

if __name__ == "__main__":
    # Check if date is provided as command line argument
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = "20190501"  # Default to 2019-05-01
    
    print(f"Showing ground truth for date: {target_date}")
    success = show_ground_truth_japan(target_date)
    
    if not success:
        print("\nFailed to create visualization. Please check the date and try again.")
        print("Date format should be YYYYMMDD (e.g., 20190501)")