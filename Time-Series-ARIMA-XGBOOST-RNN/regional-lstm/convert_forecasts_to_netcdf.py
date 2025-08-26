#!/usr/bin/env python3
"""
Convert Existing SA-ConvLSTM Forecast Arrays to NetCDF Files
===========================================================

This script converts the existing forecast arrays to NetCDF format
so they can be processed by the DBSCAN script.

Author: Assistant
Date: 2024
"""

import numpy as np
import xarray as xr
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def convert_forecasts_to_netcdf():
    """Convert existing forecast arrays and dates to NetCDF files."""
    print("="*60)
    print("CONVERTING EXISTING FORECASTS TO NETCDF FORMAT")
    print("="*60)
    
    # Load existing forecast data
    print("Loading existing forecast arrays...")
    try:
        forecasts = np.load('sa_convlstm_forecast_arrays.npy')
        print(f"✓ Loaded forecast arrays with shape: {forecasts.shape}")
    except FileNotFoundError:
        print("ERROR: sa_convlstm_forecast_arrays.npy not found!")
        print("Please run sa_convlstm_microplastics.py first to generate forecasts.")
        return
    
    # Load forecast dates from JSON
    print("Loading forecast metadata...")
    try:
        with open('sa_convlstm_forecast_results.json', 'r') as f:
            results = json.load(f)
        forecast_dates = results['forecast_dates']
        print(f"✓ Loaded forecast dates: {forecast_dates}")
    except FileNotFoundError:
        print("ERROR: sa_convlstm_forecast_results.json not found!")
        print("Using default dates...")
        from datetime import timedelta
        base_date = datetime(2025, 6, 17)
        forecast_dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                         for i in range(forecasts.shape[0])]
    
    # Create forecast directory
    forecast_dir = Path("forecast_netcdf")
    forecast_dir.mkdir(exist_ok=True)
    print(f"✓ Created directory: {forecast_dir}")
    
    # Define geographic coordinates for Japan region  
    # Based on sa_convlstm_microplastics.py coordinates
    lat_min, lat_max = 25.35753, 36.98134
    lon_min, lon_max = 118.85766, 145.47117
    
    # Create coordinate arrays for the 64x64 grid
    img_size = forecasts.shape[1]  # Should be 64
    lats = np.linspace(lat_max, lat_min, img_size)  # Reversed for geographic orientation
    lons = np.linspace(lon_min, lon_max, img_size)
    
    print(f"Geographic setup:")
    print(f"  Grid size: {img_size}x{img_size}")
    print(f"  Latitude range: {lat_min:.5f}°N to {lat_max:.5f}°N")
    print(f"  Longitude range: {lon_min:.5f}°E to {lon_max:.5f}°E")
    
    saved_files = []
    
    # Convert each forecast day
    for i, (forecast, date_str) in enumerate(zip(forecasts, forecast_dates)):
        print(f"\nProcessing forecast {i+1}/{len(forecasts)}: {date_str}")
        
        # Create filename similar to CYGNSS format
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        date_filename = date_obj.strftime("%Y%m%d")
        filename = f"sa_convlstm_forecast.s{date_filename}-120000-e{date_filename}-120000.l3.grid-microplastic.japan.nc"
        filepath = forecast_dir / filename
        
        # Flip the forecast vertically to match geographic orientation
        # (model outputs are upside-down relative to geographic coordinates)
        forecast_geo = np.flipud(forecast)
        
        print(f"  Original forecast shape: {forecast.shape}")
        print(f"  Geographic forecast shape: {forecast_geo.shape}")
        print(f"  Concentration range: {np.nanmin(forecast_geo):.6f} to {np.nanmax(forecast_geo):.6f}")
        
        # Create xarray dataset with proper dimensions and coordinates
        ds = xr.Dataset(
            data_vars={
                'mp_concentration': (
                    ['lat', 'lon'], 
                    forecast_geo,
                    {
                        'long_name': 'Microplastic Concentration Forecast',
                        'units': 'normalized_concentration',
                        'source': 'SA-ConvLSTM Model',
                        '_FillValue': np.nan,
                        'valid_min': np.nanmin(forecast_geo),
                        'valid_max': np.nanmax(forecast_geo)
                    }
                )
            },
            coords={
                'lat': (
                    ['lat'], 
                    lats,
                    {
                        'long_name': 'Latitude',
                        'units': 'degrees_north',
                        'standard_name': 'latitude'
                    }
                ),
                'lon': (
                    ['lon'], 
                    lons,
                    {
                        'long_name': 'Longitude', 
                        'units': 'degrees_east',
                        'standard_name': 'longitude'
                    }
                )
            },
            attrs={
                'title': 'SA-ConvLSTM Microplastic Concentration Forecast',
                'institution': 'MP-Prediction Project',
                'source': 'SA-ConvLSTM Deep Learning Model',
                'forecast_date': date_str,
                'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'geographic_region': 'Japan',
                'model_resolution': f'{img_size}x{img_size}',
                'lat_bounds': f'{lat_min:.5f}N to {lat_max:.5f}N',
                'lon_bounds': f'{lon_min:.5f}E to {lon_max:.5f}E',
                'data_normalization': 'Global normalization applied (0-1 range)',
                'conventions': 'CF-1.6'
            }
        )
        
        # Save the NetCDF file
        ds.to_netcdf(filepath)
        ds.close()
        
        saved_files.append(str(filepath))
        print(f"  ✓ Saved: {filename}")
    
    print(f"\n" + "="*60)
    print("CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✓ Converted {len(forecasts)} forecast arrays to NetCDF format")
    print(f"✓ Files saved to: {forecast_dir}")
    print(f"✓ Geographic coordinates: Japan region ({img_size}x{img_size} grid)")
    print(f"✓ Ready for DBSCAN clustering analysis!")
    
    print(f"\nGenerated files:")
    for file_path in saved_files:
        filename = Path(file_path).name
        print(f"  - {filename}")
    
    print(f"\nNext step: Run 'python dbscan_forecast_netcdf.py' to apply DBSCAN clustering")
    print("="*60)

if __name__ == "__main__":
    convert_forecasts_to_netcdf()