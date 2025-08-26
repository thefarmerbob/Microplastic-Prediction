import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

def extract_regional_averages_from_sa_convlstm_forecasts():
    """
    Extract regional averages from SA-ConvLSTM forecast NetCDF files
    and compare with attention LSTM predictions.
    """
    print("=" * 60)
    print("EXTRACTING REGIONAL AVERAGES FROM SA-CONVLSTM FORECASTS")
    print("=" * 60)
    
    # Define the same regions as in the attention LSTM
    regions_config = {
        'Global': [25.35753, 36.98134, 118.85766, 145.47117],  # Entire Japan region
        'Northern_Japan': [33.0, 36.98134, 118.85766, 145.47117],
        'Central_Japan': [30.0, 33.0, 118.85766, 145.47117],
        'Southern_Japan': [25.35753, 30.0, 118.85766, 145.47117],
        'Tsushima_Region': [34.0, 35.0, 129.0, 130.0]  # Tsushima Island area
    }
    
    # Look for SA-ConvLSTM forecast files
    forecast_dir = Path("/Users/maradumitru/MP-Prediction/Time-Series-ARIMA-XGBOOST-RNN/regional-lstm/forecast_netcdf")
    
    if not forecast_dir.exists():
        print(f"Forecast directory not found: {forecast_dir}")
        print("Please run the SA-ConvLSTM model first to generate forecasts.")
        return
    
    forecast_files = sorted(forecast_dir.glob("sa_convlstm_forecast*.nc"))
    
    if not forecast_files:
        print(f"No SA-ConvLSTM forecast files found in {forecast_dir}")
        return
    
    print(f"Found {len(forecast_files)} SA-ConvLSTM forecast files")
    
    # Extract regional averages from each forecast
    regional_averages = {region: [] for region in regions_config.keys()}
    forecast_dates = []
    
    for forecast_file in forecast_files:
        print(f"Processing: {forecast_file.name}")
        
        try:
            # Load forecast data
            ds = xr.open_dataset(forecast_file)
            forecast_data = ds['mp_concentration'].values
            lats = ds['lat'].values
            lons = ds['lon'].values
            
            # Extract date from filename
            date_str = forecast_file.name.split('.')[1][1:9]  # Extract from s20240101
            forecast_date = datetime.strptime(date_str, '%Y%m%d')
            forecast_dates.append(forecast_date)
            
            # Extract regional averages
            for region_name, bounds in regions_config.items():
                lat_min, lat_max, lon_min, lon_max = bounds
                
                # Find indices for the region
                lat_mask = (lats >= lat_min) & (lats <= lat_max)
                lon_mask = (lons >= lon_min) & (lons <= lon_max)
                
                # Extract regional data
                regional_data = forecast_data[np.ix_(lat_mask, lon_mask)]
                
                # Calculate average, ignoring NaN values
                regional_avg = np.nanmean(regional_data)
                if np.isnan(regional_avg):
                    regional_avg = 0.0
                
                regional_averages[region_name].append(regional_avg)
            
            ds.close()
            
        except Exception as e:
            print(f"Error processing {forecast_file}: {e}")
            continue
    
    # Convert to numpy arrays
    for region in regional_averages:
        regional_averages[region] = np.array(regional_averages[region])
    
    print(f"\nExtracted regional averages for {len(forecast_dates)} forecast days")
    
    # Create comparison visualization
    fig, axes = plt.subplots(len(regions_config), 1, figsize=(12, 4 * len(regions_config)))
    if len(regions_config) == 1:
        axes = [axes]
    
    fig.suptitle('SA-ConvLSTM Forecast: Regional Average Concentrations', fontsize=16)
    
    for i, (region_name, values) in enumerate(regional_averages.items()):
        days_ahead = range(1, len(values) + 1)
        axes[i].plot(days_ahead, values, 'o-', linewidth=2, markersize=6)
        axes[i].set_title(f'{region_name}')
        axes[i].set_xlabel('Days Ahead')
        axes[i].set_ylabel('Normalized Concentration')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        axes[i].text(0.02, 0.98, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('sa_convlstm_regional_forecast_averages.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'model_type': 'SA-ConvLSTM Regional Averages from Spatial Forecasts',
        'forecast_dates': [date.strftime('%Y-%m-%d') for date in forecast_dates],
        'regional_averages': {region: values.tolist() for region, values in regional_averages.items()},
        'statistics': {
            region: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            } for region, values in regional_averages.items()
        },
        'regions_config': regions_config
    }
    
    with open('sa_convlstm_regional_averages.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV for easy analysis
    df_data = {'date': forecast_dates}
    df_data.update({region: values for region, values in regional_averages.items()})
    df = pd.DataFrame(df_data)
    df.to_csv('sa_convlstm_regional_averages.csv', index=False)
    
    print(f"\n✓ Regional averages extracted and saved")
    print(f"✓ Visualization saved: sa_convlstm_regional_forecast_averages.png")
    print(f"✓ Results saved: sa_convlstm_regional_averages.json")
    print(f"✓ CSV saved: sa_convlstm_regional_averages.csv")
    
    # Print summary statistics
    print(f"\nRegional Average Statistics:")
    for region_name, values in regional_averages.items():
        print(f"  {region_name}:")
        print(f"    Mean: {np.mean(values):.6f}")
        print(f"    Std:  {np.std(values):.6f}")
        print(f"    Range: {np.min(values):.6f} - {np.max(values):.6f}")
    
    return regional_averages, forecast_dates

def compare_with_attention_lstm():
    """
    Compare SA-ConvLSTM regional averages with Attention LSTM predictions.
    """
    print("\n" + "=" * 60)
    print("COMPARING SA-CONVLSTM vs ATTENTION LSTM REGIONAL FORECASTS")
    print("=" * 60)
    
    # Load SA-ConvLSTM regional averages (from above function)
    try:
        with open('sa_convlstm_regional_averages.json', 'r') as f:
            sa_convlstm_results = json.load(f)
    except FileNotFoundError:
        print("SA-ConvLSTM regional averages not found. Run extract_regional_averages_from_sa_convlstm_forecasts() first.")
        return
    
    # Load Attention LSTM forecasts
    try:
        attention_lstm_forecasts = np.load('regional_forecasts.npy')
        with open('attention_lstm_regional_results.json', 'r') as f:
            attention_lstm_results = json.load(f)
    except FileNotFoundError:
        print("Attention LSTM results not found. Run the attention LSTM model first.")
        return
    
    # Extract data
    sa_convlstm_data = sa_convlstm_results['regional_averages']
    regions = list(sa_convlstm_data.keys())
    
    # Create comparison plot
    fig, axes = plt.subplots(len(regions), 1, figsize=(15, 4 * len(regions)))
    if len(regions) == 1:
        axes = [axes]
    
    fig.suptitle('Model Comparison: SA-ConvLSTM vs Attention LSTM Regional Forecasts', fontsize=16)
    
    for i, region in enumerate(regions):
        sa_values = np.array(sa_convlstm_data[region])
        attention_values = attention_lstm_forecasts[:, i] if i < attention_lstm_forecasts.shape[1] else np.zeros_like(sa_values)
        
        # Trim to same length
        min_len = min(len(sa_values), len(attention_values))
        sa_values = sa_values[:min_len]
        attention_values = attention_values[:min_len]
        
        days_ahead = range(1, min_len + 1)
        
        axes[i].plot(days_ahead, sa_values, 'o-', label='SA-ConvLSTM (from spatial)', 
                    color='blue', linewidth=2, markersize=4)
        axes[i].plot(days_ahead, attention_values, 's-', label='Attention LSTM (direct)', 
                    color='red', linewidth=2, markersize=4)
        
        axes[i].set_title(f'{region} - Forecast Comparison')
        axes[i].set_xlabel('Days Ahead')
        axes[i].set_ylabel('Normalized Concentration')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Calculate correlation
        if min_len > 1:
            correlation = np.corrcoef(sa_values, attention_values)[0, 1]
            mae_diff = np.mean(np.abs(sa_values - attention_values))
            
            axes[i].text(0.02, 0.98, f'Correlation: {correlation:.3f}\nMAE Diff: {mae_diff:.4f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_comparison_regional_forecasts.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate overall comparison metrics
    comparison_metrics = {}
    for i, region in enumerate(regions):
        sa_values = np.array(sa_convlstm_data[region])
        attention_values = attention_lstm_forecasts[:, i] if i < attention_lstm_forecasts.shape[1] else np.zeros_like(sa_values)
        
        min_len = min(len(sa_values), len(attention_values))
        sa_values = sa_values[:min_len]
        attention_values = attention_values[:min_len]
        
        if min_len > 1:
            correlation = np.corrcoef(sa_values, attention_values)[0, 1]
            mae_diff = np.mean(np.abs(sa_values - attention_values))
            mse_diff = np.mean((sa_values - attention_values) ** 2)
            
            comparison_metrics[region] = {
                'correlation': float(correlation),
                'mae_difference': float(mae_diff),
                'mse_difference': float(mse_diff),
                'sa_convlstm_mean': float(np.mean(sa_values)),
                'attention_lstm_mean': float(np.mean(attention_values))
            }
    
    # Save comparison results
    comparison_results = {
        'comparison_type': 'SA-ConvLSTM vs Attention LSTM Regional Forecasts',
        'metrics': comparison_metrics,
        'summary': {
            'avg_correlation': float(np.mean([m['correlation'] for m in comparison_metrics.values() if not np.isnan(m['correlation'])])),
            'avg_mae_difference': float(np.mean([m['mae_difference'] for m in comparison_metrics.values()]))
        }
    }
    
    with open('model_comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"✓ Model comparison completed")
    print(f"✓ Comparison plot saved: model_comparison_regional_forecasts.png")
    print(f"✓ Comparison results saved: model_comparison_results.json")
    
    print(f"\nComparison Summary:")
    print(f"  Average correlation: {comparison_results['summary']['avg_correlation']:.3f}")
    print(f"  Average MAE difference: {comparison_results['summary']['avg_mae_difference']:.6f}")
    
    for region, metrics in comparison_metrics.items():
        print(f"  {region}:")
        print(f"    Correlation: {metrics['correlation']:.3f}")
        print(f"    MAE difference: {metrics['mae_difference']:.6f}")

if __name__ == "__main__":
    # Extract regional averages from SA-ConvLSTM forecasts
    regional_averages, forecast_dates = extract_regional_averages_from_sa_convlstm_forecasts()
    
    # Compare with Attention LSTM (if available)
    compare_with_attention_lstm()



