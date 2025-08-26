import os
# MUST be set BEFORE any other imports to suppress macOS warnings
os.environ['MallocStackLogging'] = '0'
os.environ['MALLOC_STACK_LOGGING'] = '0'

import sys
print(sys.executable)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.transform import resize
import os
from datetime import datetime, timedelta
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import wandb

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AttentionLSTM(nn.Module):
    """
    Attention-based LSTM for regional microplastic concentration forecasting.
    Takes regional averages as input and predicts future regional averages.
    """
    
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, num_regions=5, seq_length=30, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        
        self.input_size = input_size  # Number of regions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_regions = num_regions
        self.seq_length = seq_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Additional layers for better representation
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_regions)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of the attention LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_regions)
            
        Returns:
            output: Predicted values of shape (batch_size, num_regions)
        """
        batch_size, seq_length, _ = x.shape
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        
        # Apply layer normalization
        lstm_out = self.layer_norm1(lstm_out)
        
        # Apply self-attention
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        attn_output = lstm_out + self.dropout(attn_output)
        attn_output = self.layer_norm2(attn_output)
        
        # Feed-forward network with residual connection
        ff_output = self.ff_network(attn_output)
        ff_output = attn_output + self.dropout(ff_output)
        
        # Take the last time step for prediction
        last_output = ff_output[:, -1, :]  # (batch_size, hidden_size)
        
        # Generate final output
        output = self.output_layer(last_output)  # (batch_size, num_regions)
        
        return output, attn_weights

class RegionalDataProcessor:
    """
    Processes spatial microplastic data to extract regional averages.
    """
    
    def __init__(self, regions_config):
        """
        Initialize with region definitions.
        
        Args:
            regions_config: Dictionary defining regions with their lat/lon bounds
        """
        self.regions_config = regions_config
    
    def extract_regional_averages_from_image(self, image, lats, lons):
        """
        Extract regional averages from a spatial image.
        
        Args:
            image: 2D numpy array representing spatial data
            lats: 1D array of latitude coordinates
            lons: 1D array of longitude coordinates
            
        Returns:
            List of regional averages
        """
        regional_averages = []
        
        for region_name, bounds in self.regions_config.items():
            lat_min, lat_max, lon_min, lon_max = bounds
            
            # Find indices for the region
            lat_mask = (lats >= lat_min) & (lats <= lat_max)
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
            
            # Extract regional data
            if len(image.shape) == 2:
                regional_data = image[np.ix_(lat_mask, lon_mask)]
            else:
                # Handle case where image has extra dimensions
                regional_data = image[lat_mask][:, lon_mask]
            
            # Calculate average, ignoring NaN values
            regional_avg = np.nanmean(regional_data)
            if np.isnan(regional_avg):
                regional_avg = 0.0
                
            regional_averages.append(regional_avg)
        
        return regional_averages

def load_cygnss_data_and_extract_regions(nc_files, regions_config, max_files=None):
    """
    Load CYGNSS data and extract regional time series.
    
    Args:
        nc_files: List of NetCDF file paths
        regions_config: Dictionary defining regions
        max_files: Maximum number of files to process (for testing)
        
    Returns:
        tuple: (regional_data, timestamps, global_stats)
    """
    print("Loading CYGNSS data and extracting regional averages...")
    
    processor = RegionalDataProcessor(regions_config)
    regional_time_series = []
    timestamps = []
    
    # Limit files for testing if specified
    if max_files:
        nc_files = nc_files[:max_files]
    
    # Get global statistics for normalization
    print("Computing global statistics for normalization...")
    all_data = []
    
    for i, nc_file in enumerate(nc_files[:50]):  # Sample first 50 files for stats
        if i % 10 == 0:
            print(f"  Sampling file {i+1}/50 for global statistics")
        
        try:
            ds = xr.open_dataset(nc_file)
            data = ds['mp_concentration'].values.squeeze()
            
            if not np.all(np.isnan(data)):
                all_data.append(data.flatten())
            ds.close()
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")
            continue
    
    if all_data:
        all_data = np.concatenate(all_data)
        global_min = np.nanmin(all_data)
        global_max = np.nanmax(all_data)
        global_mean = np.nanmean(all_data)
        global_std = np.nanstd(all_data)
    else:
        global_min, global_max, global_mean, global_std = 0, 1, 0, 1
    
    print(f"Global statistics: min={global_min:.2f}, max={global_max:.2f}, mean={global_mean:.2f}, std={global_std:.2f}")
    
    # Process all files
    print(f"Processing {len(nc_files)} files...")
    
    for i, nc_file in enumerate(nc_files):
        if i % 50 == 0:
            print(f"  Processing file {i+1}/{len(nc_files)}")
        
        try:
            # Extract timestamp from filename
            filename = Path(nc_file).name
            date_part = filename.split('.')[2][1:9]  # Extract s20180816 -> 20180816
            timestamp = datetime.strptime(date_part, '%Y%m%d')
            
            # Load data
            ds = xr.open_dataset(nc_file)
            data = ds['mp_concentration']
            data_array = data.values.squeeze()
            
            # Get coordinates
            lats = ds.coords['lat'].values if 'lat' in ds.coords else ds.coords['latitude'].values
            lons = ds.coords['lon'].values if 'lon' in ds.coords else ds.coords['longitude'].values
            
            # Crop to Japan region (same as existing models)
            japan_sw_lat, japan_sw_lon = 25.35753, 118.85766
            japan_ne_lat, japan_ne_lon = 36.98134, 145.47117
            
            lat_mask = (lats >= japan_sw_lat) & (lats <= japan_ne_lat)
            lon_mask = (lons >= japan_sw_lon) & (lons <= japan_ne_lon)
            
            # Apply cropping
            data_cropped = data_array[np.ix_(lat_mask, lon_mask)]
            lats_cropped = lats[lat_mask]
            lons_cropped = lons[lon_mask]
            
            # Normalize data
            if global_max > global_min:
                data_normalized = (data_cropped - global_min) / (global_max - global_min)
            else:
                data_normalized = np.zeros_like(data_cropped)
            
            # Replace NaN with 0
            data_normalized = np.nan_to_num(data_normalized, nan=0.0)
            
            # Extract regional averages
            regional_averages = processor.extract_regional_averages_from_image(
                data_normalized, lats_cropped, lons_cropped
            )
            
            regional_time_series.append(regional_averages)
            timestamps.append(timestamp)
            
            ds.close()
            
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")
            continue
    
    regional_data = np.array(regional_time_series)
    print(f"Extracted regional data shape: {regional_data.shape}")
    print(f"Time span: {timestamps[0]} to {timestamps[-1]}")
    
    global_stats = {
        'min': global_min,
        'max': global_max,
        'mean': global_mean,
        'std': global_std
    }
    
    return regional_data, timestamps, global_stats

def create_sequences(data, seq_length, forecast_horizon=1):
    """
    Create sequences for training the attention LSTM.
    
    Args:
        data: Regional time series data (time_steps, num_regions)
        seq_length: Length of input sequences
        forecast_horizon: Number of steps to forecast ahead
        
    Returns:
        tuple: (X, y) where X is input sequences and y is targets
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon])
    
    return np.array(X), np.array(y)

def train_attention_lstm(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """
    Train the attention LSTM model.
    """
    print(f"\nStarting Attention LSTM training for {epochs} epochs...")
    
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float().squeeze(1)  # Remove extra dimension
            
            optimizer.zero_grad()
            outputs, attention_weights = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float().squeeze(1)
                
                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_attention_lstm_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'best_val_loss': best_val_loss
        })
    
    print(f"\nAttention LSTM training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return history

def evaluate_attention_lstm(model, test_loader, region_names, test_timestamps):
    """
    Evaluate the attention LSTM model.
    """
    print("\nEvaluating Attention LSTM model...")
    
    model.eval()
    predictions = []
    targets = []
    attention_weights_list = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float().squeeze(1)
            
            outputs, attention_weights = model(batch_x)
            
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
            attention_weights_list.append(attention_weights.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    
    # Calculate per-region metrics
    region_metrics = {}
    for i, region_name in enumerate(region_names):
        region_mae = mean_absolute_error(targets[:, i], predictions[:, i])
        region_mse = mean_squared_error(targets[:, i], predictions[:, i])
        region_rmse = np.sqrt(region_mse)
        
        region_metrics[region_name] = {
            'mae': region_mae,
            'mse': region_mse,
            'rmse': region_rmse
        }
    
    print(f"Overall Test Metrics:")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    print(f"\nPer-Region Metrics:")
    for region_name, metrics in region_metrics.items():
        print(f"  {region_name}:")
        print(f"    MAE: {metrics['mae']:.6f}")
        print(f"    RMSE: {metrics['rmse']:.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(len(region_names), 1, figsize=(15, 4 * len(region_names)))
    if len(region_names) == 1:
        axes = [axes]
    
    fig.suptitle('Attention LSTM: Regional Microplastic Concentration Predictions', fontsize=16)
    
    for i, region_name in enumerate(region_names):
        # Plot time series
        time_indices = range(len(predictions))
        axes[i].plot(time_indices, targets[:, i], label='Actual', color='blue', alpha=0.7)
        axes[i].plot(time_indices, predictions[:, i], label='Predicted', color='red', alpha=0.7)
        
        axes[i].set_title(f'{region_name} - MAE: {region_metrics[region_name]["mae"]:.4f}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Normalized Concentration')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_lstm_regional_predictions.png', dpi=150, bbox_inches='tight')
    
    # Log to wandb
    wandb.log({
        "regional_predictions": wandb.Image('attention_lstm_regional_predictions.png'),
        "test_mae": mae,
        "test_mse": mse,
        "test_rmse": rmse
    })
    
    # Log per-region metrics
    for region_name, metrics in region_metrics.items():
        wandb.log({
            f"test_mae_{region_name}": metrics['mae'],
            f"test_rmse_{region_name}": metrics['rmse']
        })
    
    return mae, rmse, predictions, targets, region_metrics

def forecast_future_regional(model, last_sequence, forecast_days=30, region_names=None):
    """
    Forecast future regional averages using the trained model.
    """
    print(f"\nForecasting {forecast_days} days into the future...")
    
    model.eval()
    forecasts = []
    
    # Start with the last sequence
    current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for day in range(forecast_days):
            # Predict next day
            prediction, attention_weights = model(current_sequence)
            next_day_pred = prediction.cpu().numpy().squeeze()
            
            forecasts.append(next_day_pred)
            
            # Update sequence for next prediction (sliding window)
            # Remove first day and add the new prediction
            new_sequence = current_sequence[:, 1:, :].clone()
            next_day_tensor = torch.FloatTensor(next_day_pred).unsqueeze(0).unsqueeze(0).to(device)
            current_sequence = torch.cat([new_sequence, next_day_tensor], dim=1)
    
    forecasts = np.array(forecasts)
    print(f"Generated forecasts shape: {forecasts.shape}")
    
    # Create forecast visualization
    if region_names is None:
        region_names = [f'Region_{i+1}' for i in range(forecasts.shape[1])]
    
    fig, axes = plt.subplots(len(region_names), 1, figsize=(15, 4 * len(region_names)))
    if len(region_names) == 1:
        axes = [axes]
    
    fig.suptitle(f'Attention LSTM: {forecast_days}-Day Regional Forecast', fontsize=16)
    
    for i, region_name in enumerate(region_names):
        time_indices = range(forecast_days)
        axes[i].plot(time_indices, forecasts[:, i], label='Forecast', color='green', linewidth=2)
        
        axes[i].set_title(f'{region_name} Forecast')
        axes[i].set_xlabel('Days Ahead')
        axes[i].set_ylabel('Normalized Concentration')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_lstm_regional_forecast.png', dpi=150, bbox_inches='tight')
    
    # Log to wandb
    wandb.log({
        "regional_forecast": wandb.Image('attention_lstm_regional_forecast.png'),
        "forecast_days": forecast_days,
        "forecast_mean": np.mean(forecasts),
        "forecast_std": np.std(forecasts)
    })
    
    return forecasts

def main():
    print("=" * 60)
    print("ATTENTION LSTM FOR REGIONAL MICROPLASTIC FORECASTING")
    print("=" * 60)
    
    # Define regions (you can customize these based on your needs)
    regions_config = {
        'Global': [25.35753, 36.98134, 118.85766, 145.47117],  # Entire Japan region
        'Northern_Japan': [33.0, 36.98134, 118.85766, 145.47117],
        'Central_Japan': [30.0, 33.0, 118.85766, 145.47117],
        'Southern_Japan': [25.35753, 30.0, 118.85766, 145.47117],
        'Tsushima_Region': [34.0, 35.0, 129.0, 130.0]  # Tsushima Island area
    }
    
    region_names = list(regions_config.keys())
    print(f"Defined {len(region_names)} regions: {region_names}")
    
    # Get CYGNSS data files
    nc_files = sorted(Path("/Users/maradumitru/Downloads/CYGNSS-data").glob("cyg.ddmi*.nc"))
    print(f"Found {len(nc_files)} CYGNSS data files")
    
    # Extract regional time series
    max_files = 1000  # Limit for faster testing
    regional_data, timestamps, global_stats = load_cygnss_data_and_extract_regions(
        nc_files, regions_config, max_files=max_files
    )
    
    # Model parameters
    seq_length = 30  # Use 30 days of history
    forecast_horizon = 1  # Predict 1 day ahead
    hidden_size = 128
    num_layers = 2
    
    # Create sequences
    X, y = create_sequences(regional_data, seq_length, forecast_horizon)
    print(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data temporally
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    print(f"Data splits:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Convert to PyTorch tensors and create data loaders
    batch_size = 32
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize wandb
    wandb.init(
        project="attention-lstm-microplastics",
        name=f"attention-lstm-regional-{len(region_names)}regions",
        config={
            "model_type": "Attention LSTM Regional",
            "framework": "PyTorch",
            "sequence_length": seq_length,
            "forecast_horizon": forecast_horizon,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_regions": len(region_names),
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "max_files": max_files,
            "device": str(device),
            "regions": region_names
        }
    )
    
    # Create model
    model = AttentionLSTM(
        input_size=len(region_names),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_regions=len(region_names),
        seq_length=seq_length
    )
    
    print(f"\nModel Architecture:")
    print(f"  Input size: {len(region_names)} regions")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Sequence length: {seq_length}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Train model
    history = train_attention_lstm(model, train_loader, val_loader, epochs=50)
    
    # Load best model
    model.load_state_dict(torch.load('best_attention_lstm_model.pth'))
    
    # Evaluate model
    test_timestamps = timestamps[n_train + n_val + seq_length:]
    mae, rmse, predictions, targets, region_metrics = evaluate_attention_lstm(
        model, test_loader, region_names, test_timestamps
    )
    
    # Generate forecasts
    last_sequence = regional_data[-seq_length:]  # Use last 30 days
    forecasts = forecast_future_regional(model, last_sequence, forecast_days=30, region_names=region_names)
    
    # Save results
    results = {
        'model_type': 'Attention LSTM Regional Forecasting',
        'framework': 'PyTorch',
        'regions': region_names,
        'sequence_length': seq_length,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'test_metrics': {
            'overall_mae': float(mae),
            'overall_rmse': float(rmse)
        },
        'regional_metrics': {name: {k: float(v) for k, v in metrics.items()} 
                           for name, metrics in region_metrics.items()},
        'forecast_statistics': {
            'forecast_days': 30,
            'mean_concentration': float(np.mean(forecasts)),
            'std_concentration': float(np.std(forecasts)),
            'min_concentration': float(np.min(forecasts)),
            'max_concentration': float(np.max(forecasts))
        },
        'global_normalization_stats': {k: float(v) for k, v in global_stats.items()}
    }
    
    with open('attention_lstm_regional_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model and forecasts
    torch.save(model.state_dict(), 'attention_lstm_regional_model.pth')
    np.save('regional_forecasts.npy', forecasts)
    np.save('regional_test_predictions.npy', predictions)
    np.save('regional_test_targets.npy', targets)
    
    print("\n" + "=" * 60)
    print("ATTENTION LSTM REGIONAL FORECASTING COMPLETED")
    print("=" * 60)
    print(f"✓ Overall Test MAE: {mae:.6f}")
    print(f"✓ Overall Test RMSE: {rmse:.6f}")
    print(f"✓ Model parameters: {total_params:,}")
    print(f"✓ Regions analyzed: {len(region_names)}")
    print(f"✓ Generated 30-day forecast")
    print("\nFiles generated:")
    print("- attention_lstm_regional_predictions.png")
    print("- attention_lstm_regional_forecast.png")
    print("- attention_lstm_regional_results.json")
    print("- attention_lstm_regional_model.pth")
    print("- regional_forecasts.npy")
    print("=" * 60)
    
    wandb.finish()

if __name__ == "__main__":
    main()
