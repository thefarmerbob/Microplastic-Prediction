#!/usr/bin/env python3
"""
Script to create cropped GIF using only the last 365 NetCDF files from the CYGNSS data.
This script applies the same cropping as section-cropped.py but only to the most recent year of data.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as imageio
import sys

def lat_lon_to_indices(lat, lon, lats, lons):
    """Convert lat/lon coordinates to grid indices"""
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))
    return lat_idx, lon_idx

def process_and_plot_cropped(nc_file, output_dir, vmin=None, vmax=None):
    """
    Process a single NetCDF file and create a cropped visualization.
    This replicates the cropping logic from section-cropped.py
    
    Args:
        nc_file: Path to NetCDF file
        output_dir: Directory to save image
        vmin: Minimum value for colorbar (if None, will be calculated)
        vmax: Maximum value for colorbar (if None, will be calculated)
    """
    ds = xr.open_dataset(nc_file)
    
    data = ds['mp_concentration']
    data_array = data.values
    data_array_2d = data_array.squeeze()
    data_array_2d_flipped = np.flipud(data_array_2d)

    # Calculate GLOBAL average microplastic concentration using full dataset, ignoring NaNs
    global_average_concentration = np.nanmean(data_array_2d)
    
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
    
    if lats is None or lons is None:
        print("Could not find lat/lon coordinates in dataset")
        print(f"Available variables: {list(ds.variables.keys())}")
        ds.close()
        return None
    
    # Define the display region coordinates (East Asia region from section-cropped.py)
    display_sw_lat, display_sw_lon = 22.30649, 118.07623
    display_ne_lat, display_ne_lon = 36.96467, 143.9533
    
    # Convert display region coordinates to grid indices
    display_sw_lat_idx, display_sw_lon_idx = lat_lon_to_indices(display_sw_lat, display_sw_lon, lats, lons)
    display_ne_lat_idx, display_ne_lon_idx = lat_lon_to_indices(display_ne_lat, display_ne_lon, lats, lons)
    
    # Ensure proper ordering for display region
    display_lat_start = min(display_sw_lat_idx, display_ne_lat_idx)
    display_lat_end = max(display_sw_lat_idx, display_ne_lat_idx)
    display_lon_start = min(display_sw_lon_idx, display_ne_lon_idx)
    display_lon_end = max(display_sw_lon_idx, display_ne_lon_idx)
    
    # Crop the data for display
    data_cropped = data_array_2d[display_lat_start:display_lat_end, display_lon_start:display_lon_end]
    data_cropped_flipped = np.flipud(data_cropped)
    
    # Define the regions of interest with coordinates
    # Japan region (broader)
    japan_sw_lat, japan_sw_lon = 25.35753, 118.85766
    japan_ne_lat, japan_ne_lon = 36.98134, 145.47117
    
    # Tsushima region
    tsushima_sw_lat, tsushima_sw_lon = 34.02837, 129.11613
    tsushima_ne_lat, tsushima_ne_lon = 34.76456, 129.55801
    
    # Tokyo region
    tokyo_sw_lat, tokyo_sw_lon = 34.80503, 139.10416
    tokyo_ne_lat, tokyo_ne_lon = 35.30061, 139.81791
    
    # Kyoto region
    kyoto_sw_lat, kyoto_sw_lon = 36.25588, 135.17307
    kyoto_ne_lat, kyoto_ne_lon = 36.64146, 135.96607
    
    # Osaka region
    osaka_sw_lat, osaka_sw_lon = 32.88645, 134.44579
    osaka_ne_lat, osaka_ne_lon = 33.17125, 135.09421
    
    # Set up display extent for the cropped region
    display_extent = [lons[display_lon_start], lons[display_lon_end], 
                     lats[display_lat_start], lats[display_lat_end]]
    
    # Create figure with proper geographic aspect ratio
    height, width = data_cropped_flipped.shape
    fig_width = 12
    fig_height = fig_width * (height / width)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Display the cropped data with fixed colorbar scale
    img = ax.imshow(data_cropped_flipped, aspect='auto', cmap='viridis', extent=display_extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    
    # Draw the Japan region box (blue box) - only if it's within the display region
    if (japan_sw_lon >= display_extent[0] and japan_ne_lon <= display_extent[1] and
        japan_sw_lat >= display_extent[2] and japan_ne_lat <= display_extent[3]):
        japan_rect = plt.Rectangle((japan_sw_lon, japan_sw_lat), 
                                 japan_ne_lon - japan_sw_lon, japan_ne_lat - japan_sw_lat,
                                 linewidth=3, edgecolor='blue', facecolor='none', 
                                 label='Japan Region')
        ax.add_patch(japan_rect)
    
    # Draw the Tsushima region box (red box) - only if it's within the display region
    if (tsushima_sw_lon >= display_extent[0] and tsushima_ne_lon <= display_extent[1] and
        tsushima_sw_lat >= display_extent[2] and tsushima_ne_lat <= display_extent[3]):
        tsushima_rect = plt.Rectangle((tsushima_sw_lon, tsushima_sw_lat), 
                                    tsushima_ne_lon - tsushima_sw_lon, tsushima_ne_lat - tsushima_sw_lat,
                                    linewidth=2, edgecolor='red', facecolor='none', 
                                    label='Tsushima Region')
        ax.add_patch(tsushima_rect)
    
    # Draw the Tokyo region box (pink box) - only if it's within the display region
    if (tokyo_sw_lon >= display_extent[0] and tokyo_ne_lon <= display_extent[1] and
        tokyo_sw_lat >= display_extent[2] and tokyo_ne_lat <= display_extent[3]):
        tokyo_rect = plt.Rectangle((tokyo_sw_lon, tokyo_sw_lat), 
                                 tokyo_ne_lon - tokyo_sw_lon, tokyo_ne_lat - tokyo_sw_lat,
                                 linewidth=2, edgecolor='pink', facecolor='none', 
                                 label='Tokyo Region')
        ax.add_patch(tokyo_rect)
    
    # Draw the Kyoto region box (magenta box) - only if it's within the display region
    if (kyoto_sw_lon >= display_extent[0] and kyoto_ne_lon <= display_extent[1] and
        kyoto_sw_lat >= display_extent[2] and kyoto_ne_lat <= display_extent[3]):
        kyoto_rect = plt.Rectangle((kyoto_sw_lon, kyoto_sw_lat), 
                                 kyoto_ne_lon - kyoto_sw_lon, kyoto_ne_lat - kyoto_sw_lat,
                                 linewidth=2, edgecolor='magenta', facecolor='none', 
                                 label='Kyoto Region')
        ax.add_patch(kyoto_rect)
    
    # Draw the Osaka region box (dark red box) - only if it's within the display region
    if (osaka_sw_lon >= display_extent[0] and osaka_ne_lon <= display_extent[1] and
        osaka_sw_lat >= display_extent[2] and osaka_ne_lat <= display_extent[3]):
        osaka_rect = plt.Rectangle((osaka_sw_lon, osaka_sw_lat), 
                                 osaka_ne_lon - osaka_sw_lon, osaka_ne_lat - osaka_sw_lat,
                                 linewidth=2, edgecolor='darkred', facecolor='none', 
                                 label='Osaka Region')
        ax.add_patch(osaka_rect)
    
    # Add colorbar
    cbar = fig.colorbar(img, ax=ax, label='Microplastic Concentration', shrink=0.8)
    
    # Extract date from filename for title
    file_name = Path(nc_file).name
    date_str = file_name.split('.')[2][1:]  # Remove 's' prefix
    date_title = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    ax.set_title(f"Microplastic Concentration Map - {date_title}\\n(East Asia Region)", fontsize=14, pad=20)
    
    # Save the image
    image_file = output_dir / f"cropped_timeseries_{Path(nc_file).stem}.png"
    fig.savefig(image_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    ds.close()
    return image_file

def calculate_global_colorbar_range(nc_files):
    """
    Calculate the global min/max values across all NetCDF files for consistent colorbar.
    
    Args:
        nc_files: List of NetCDF file paths
        
    Returns:
        tuple: (vmin, vmax) for colorbar
    """
    print("Calculating global colorbar range...")
    all_values = []
    
    for i, nc_file in enumerate(nc_files):
        if i % 50 == 0:  # Progress indicator
            print(f"Analyzing file {i+1}/{len(nc_files)} for colorbar range")
        
        try:
            ds = xr.open_dataset(nc_file)
            data = ds['mp_concentration']
            data_array = data.values
            data_array_2d = data_array.squeeze()
            
            # Get non-NaN values
            valid_values = data_array_2d[~np.isnan(data_array_2d)]
            if len(valid_values) > 0:
                all_values.extend([valid_values.min(), valid_values.max()])
            
            ds.close()
        except Exception as e:
            print(f"Warning: Could not read {nc_file} for colorbar range: {e}")
            continue
    
    if not all_values:
        print("Warning: No valid data found for colorbar range, using default")
        return 0, 1
    
    vmin = np.min(all_values)
    vmax = np.max(all_values)
    
    print(f"Global colorbar range: {vmin:.4f} to {vmax:.4f}")
    return vmin, vmax

def create_cropped_gif_last_365():
    """
    Create a cropped GIF using only the last 365 NetCDF files.
    """
    print("="*60)
    print("CREATING CROPPED GIF WITH LAST 365 FILES")
    print("="*60)
    
    # Get all NetCDF files from CYGNSS data
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    
    if not nc_files:
        print("❌ Error: No CYGNSS data files found!")
        return
    
    print(f"Found {len(nc_files)} total NetCDF files")
    
    # Get the last 365 files
    if len(nc_files) < 365:
        print(f"Warning: Only {len(nc_files)} files available, using all of them")
        last_365_files = nc_files
    else:
        last_365_files = nc_files[-365:]
    
    print(f"Processing last {len(last_365_files)} files...")
    print(f"Date range: {last_365_files[0].name.split('.')[2][1:]} to {last_365_files[-1].name.split('.')[2][1:]}")
    
    # Calculate global colorbar range for consistent scaling
    vmin, vmax = calculate_global_colorbar_range(last_365_files)
    
    # Create output directory for cropped images
    output_dir = Path("cropped_timeseries_images_last365_fixed_colorbar")
    output_dir.mkdir(exist_ok=True)
    
    # Process each file and create cropped images
    image_files = []
    
    for i, nc_file in enumerate(last_365_files):
        if i % 50 == 0:  # Progress indicator
            print(f"Processing file {i+1}/{len(last_365_files)}")
        
        try:
            image_file = process_and_plot_cropped(nc_file, output_dir, vmin=vmin, vmax=vmax)
            if image_file:
                image_files.append(image_file)
        except Exception as e:
            print(f"❌ Error processing {nc_file}: {e}")
            continue
    
    if not image_files:
        print("❌ Error: No images were created!")
        return
    
    print(f"✅ Successfully created {len(image_files)} cropped images")
    
    # Create the GIF
    print("\\nCreating cropped GIF...")
    images = []
    for i, image_file in enumerate(image_files):
        if i % 50 == 0:  # Progress indicator
            print(f"Loading image {i+1}/{len(image_files)} for GIF")
        images.append(imageio.imread(image_file))
    
    # Create both normal and fast versions with fixed colorbar
    output_gif_normal = "microplastic_timeseries_cropped_last365_fixed_colorbar.gif"
    output_gif_fast = "microplastic_timeseries_cropped_last365_fixed_colorbar_fast.gif"
    
    imageio.mimsave(output_gif_normal, images, duration=0.8)
    imageio.mimsave(output_gif_fast, images, duration=0.4)
    
    print("\\n" + "="*60)
    print("✅ CROPPED GIFS WITH FIXED COLORBAR CREATED SUCCESSFULLY!")
    print("="*60)
    print("Created files:")
    print(f"  • {output_gif_normal} (normal speed)")
    print(f"  • {output_gif_fast} (2x speed)")
    print(f"  • Both contain the last 365 days of cropped East Asia data ({len(images)} frames)")
    print(f"  • Fixed colorbar range: {vmin:.4f} to {vmax:.4f}")
    print(f"  • Images saved in: {output_dir}")

def main():
    """Main function"""
    try:
        create_cropped_gif_last_365()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()