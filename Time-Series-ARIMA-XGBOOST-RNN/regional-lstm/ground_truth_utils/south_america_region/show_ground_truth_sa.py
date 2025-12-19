#!/usr/bin/env python3
"""
Show Ground Truth Data for South America Region
================================================

Simple script to display the ground truth microplastics data for a specific date
in the South America region with proper geographic coordinates.

Region: South America (cropped)
Author: Assistant
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def lat_lon_to_indices(lat, lon, lats, lons):
    """Convert lat/lon coordinates to grid indices"""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def find_file_for_date(nc_files, target_date_str):
    """Find the NetCDF file for the target date"""
    for nc_file in nc_files:
        filename = Path(nc_file).name
        # Extract date from filename: cyg.ddmi.s20240208 -> 20240208
        if '.s' + target_date_str in filename:
            return nc_file
    return None

def show_ground_truth_sa(target_date="20190501"):
    """Show ground truth data for the specified date in South America region"""
    
    print(f"Loading ground truth data for {target_date}")
    
    # Load NetCDF files
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    if not nc_files:
        print("No CYGNSS data files found!")
        return
    
    # Find the specific file for our target date
    target_file = find_file_for_date(nc_files, target_date)
    if target_file is None:
        print(f"No file found for date {target_date}")
        print("Available files:")
        for f in nc_files[:10]:  # Show first 10 files
            print(f"  {f.name}")
        return
    
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
        return
    
    # Debug: Print coordinate ranges
    print(f"\nCoordinate information:")
    print(f"Latitude range: {np.min(lats):.2f} to {np.max(lats):.2f}")
    print(f"Longitude range: {np.min(lons):.2f} to {np.max(lons):.2f}")
    print(f"Latitude shape: {lats.shape}")
    print(f"Longitude shape: {lons.shape}")
    
    # Crop to South America region
    sa_sw_lat, sa_sw_lon = -39.88971, -63.26431
    sa_ne_lat, sa_ne_lon = -34.62727, -51.16903
    
    # Convert negative longitudes to 0-360 format if needed
    if np.min(lons) >= 0 and np.max(lons) <= 360:
        print(f"\nLongitudes are in 0-360 format, converting target coordinates...")
        if sa_sw_lon < 0:
            sa_sw_lon = sa_sw_lon + 360
        if sa_ne_lon < 0:
            sa_ne_lon = sa_ne_lon + 360
        print(f"Converted SW lon: {sa_sw_lon:.2f}, NE lon: {sa_ne_lon:.2f}")
    
    # Convert to grid indices
    sa_sw_lat_idx, sa_sw_lon_idx = lat_lon_to_indices(sa_sw_lat, sa_sw_lon, lats, lons)
    sa_ne_lat_idx, sa_ne_lon_idx = lat_lon_to_indices(sa_ne_lat, sa_ne_lon, lats, lons)
    
    print(f"\nFound indices:")
    print(f"SW: lat_idx={sa_sw_lat_idx}, lon_idx={sa_sw_lon_idx}")
    print(f"NE: lat_idx={sa_ne_lat_idx}, lon_idx={sa_ne_lon_idx}")
    
    # Ensure proper ordering
    sa_lat_start = min(sa_sw_lat_idx, sa_ne_lat_idx)
    sa_lat_end = max(sa_sw_lat_idx, sa_ne_lat_idx)
    sa_lon_start = min(sa_sw_lon_idx, sa_ne_lon_idx)
    sa_lon_end = max(sa_sw_lon_idx, sa_ne_lon_idx)
    
    # Crop the data to South America region
    sa_data = data_array_2d[sa_lat_start:sa_lat_end, sa_lon_start:sa_lon_end]
    
    print(f"South America region cropped to: lat[{sa_lat_start}:{sa_lat_end}], lon[{sa_lon_start}:{sa_lon_end}]")
    print(f"South America region shape: {sa_data.shape}")
    print(f"Data range: {np.nanmin(sa_data):.2e} to {np.nanmax(sa_data):.2e}")
    print(f"Average concentration: {np.nanmean(sa_data):.2e}")
    
    # Create visualization (larger figure for better detail)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # South America region geographic bounds for display
    lon_min, lon_max = -63.26, -51.17
    lat_min, lat_max = -39.89, -34.63
    
    # Flip data vertically for correct geographic orientation
    sa_data_flipped = np.flipud(sa_data)
    
    # Display the data with geographic extent (no interpolation to preserve resolution)
    img = ax.imshow(sa_data_flipped, cmap='viridis', aspect='auto', origin='lower',
                   extent=[lon_min, lon_max, lat_min, lat_max], interpolation='none')
    
    # Add colorbar
    cbar = plt.colorbar(img, ax=ax, label='Microplastics Concentration', shrink=0.8)
    
    # Format the date for display
    date_obj = datetime.strptime(target_date, '%Y%m%d')
    formatted_date = date_obj.strftime('%B %d, %Y')
    
    # Add title and labels
    ax.set_title(f'Ground Truth Microplastics Data\n{formatted_date} - South America Region', fontsize=16)
    ax.set_xlabel('Longitude (°W)', fontsize=12)
    ax.set_ylabel('Latitude (°S)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (f"Data Statistics:\n"
                 f"Min: {np.nanmin(sa_data):.2e}\n"
                 f"Max: {np.nanmax(sa_data):.2e}\n"
                 f"Mean: {np.nanmean(sa_data):.2e}\n"
                 f"Shape: {sa_data.shape}")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save the plot with higher resolution
    output_filename = f'ground_truth_{target_date}_sa.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_filename}")
    
    plt.show()
    
    # Close the dataset
    ds.close()

if __name__ == "__main__":
    # Show ground truth for 2019-05-01
    show_ground_truth_sa("20190501")

