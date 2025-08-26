#!/usr/bin/env python3
"""
Demo script for using the trained Attention LSTM model for regional microplastic forecasting.
This script shows how to load the model and generate new forecasts.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import the model class (assuming it's in the same directory)
import sys
sys.path.append('.')
from attention_lstm_regional_forecaster import AttentionLSTM

def load_trained_model(model_path='attention_lstm_regional_model.pth', 
                      results_path='attention_lstm_regional_results.json'):
    """Load the trained Attention LSTM model."""
    
    # Load model configuration
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract model parameters
    num_regions = len(results['regions'])
    hidden_size = results['hidden_size']
    num_layers = results['num_layers']
    seq_length = results['sequence_length']
    
    # Create model
    model = AttentionLSTM(
        input_size=num_regions,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_regions=num_regions,
        seq_length=seq_length
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"✓ Loaded Attention LSTM model")
    print(f"  - Regions: {results['regions']}")
    print(f"  - Sequence length: {seq_length}")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, results

def demo_forecasting():
    """Demonstrate forecasting with the trained model."""
    
    print("=" * 60)
    print("ATTENTION LSTM REGIONAL FORECASTING DEMO")
    print("=" * 60)
    
    # Check if model files exist
    model_path = 'attention_lstm_regional_model.pth'
    results_path = 'attention_lstm_regional_results.json'
    
    if not Path(model_path).exists() or not Path(results_path).exists():
        print("Error: Model files not found!")
        print("Please run 'python attention_lstm_regional_forecaster.py' first to train the model.")
        return
    
    # Load the trained model
    model, results = load_trained_model(model_path, results_path)
    region_names = results['regions']
    seq_length = results['sequence_length']
    
    # Load actual test data for demonstration
    try:
        test_predictions = np.load('regional_test_predictions.npy')
        test_targets = np.load('regional_test_targets.npy')
        forecasts = np.load('regional_forecasts.npy')
        
        print(f"\n✓ Loaded test data:")
        print(f"  - Test samples: {len(test_predictions)}")
        print(f"  - Forecast days: {len(forecasts)}")
        
    except FileNotFoundError:
        print("Warning: Test data files not found. Using synthetic data for demo.")
        # Create synthetic data for demo
        test_predictions = np.random.rand(50, len(region_names)) * 0.5
        test_targets = test_predictions + np.random.normal(0, 0.05, test_predictions.shape)
        forecasts = np.random.rand(30, len(region_names)) * 0.5
    
    # Create a demo forecast using the last sequence from test data
    print(f"\n📊 DEMONSTRATION: Recent Model Performance")
    print("-" * 40)
    
    # Show recent predictions vs actual
    recent_samples = min(10, len(test_predictions))
    for i in range(recent_samples):
        print(f"Sample {i+1}:")
        for j, region in enumerate(region_names):
            pred = test_predictions[i, j]
            actual = test_targets[i, j]
            error = abs(pred - actual)
            print(f"  {region:15s}: Pred={pred:6.4f}, Actual={actual:6.4f}, Error={error:6.4f}")
        print()
    
    # Show forecast summary
    print(f"🔮 FUTURE FORECAST (30 days ahead)")
    print("-" * 40)
    for j, region in enumerate(region_names):
        forecast_values = forecasts[:, j]
        mean_forecast = np.mean(forecast_values)
        std_forecast = np.std(forecast_values)
        trend = "📈" if forecast_values[-1] > forecast_values[0] else "📉"
        
        print(f"{region:15s}: Mean={mean_forecast:6.4f} ±{std_forecast:6.4f} {trend}")
    
    # Create visualization
    create_demo_visualization(test_predictions, test_targets, forecasts, region_names)
    
    # Example of making a new prediction
    print(f"\n🎯 EXAMPLE: Making a New Prediction")
    print("-" * 40)
    
    # Use the last sequence from test data as example input
    if len(test_targets) >= seq_length:
        example_sequence = test_targets[-seq_length:]  # Shape: (30, 5)
        print(f"Input sequence shape: {example_sequence.shape}")
        print(f"Using last {seq_length} days of data as input...")
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(example_sequence).unsqueeze(0)  # Add batch dimension
            prediction, attention_weights = model(input_tensor)
            prediction = prediction.squeeze().numpy()
        
        print(f"\nPredicted regional concentrations for next day:")
        for i, region in enumerate(region_names):
            print(f"  {region:15s}: {prediction[i]:6.4f}")
        
        # Show attention pattern
        try:
            # Attention weights shape might vary, so handle different cases
            attn_weights = attention_weights.squeeze()
            if len(attn_weights.shape) > 1:
                attention_mean = torch.mean(attn_weights, dim=0)  # Average across heads/positions
                if len(attention_mean.shape) > 1:
                    attention_summary = torch.mean(attention_mean, dim=0)  # Further averaging if needed
                else:
                    attention_summary = attention_mean
            else:
                attention_summary = attn_weights
            
            print(f"\nAttention pattern (importance of each input day):")
            if len(attention_summary) >= 5:
                top_days = torch.topk(attention_summary, k=5)
                for i, (score, day_idx) in enumerate(zip(top_days.values, top_days.indices)):
                    print(f"  Day -{seq_length-day_idx}: {score:.4f} importance")
            else:
                print("  Attention pattern available but dimensions incompatible for display")
        except Exception as e:
            print(f"  Note: Attention visualization skipped due to shape mismatch: {e}")
    
    print(f"\n✅ Demo completed! Check 'demo_forecast_visualization.png' for results.")

def create_demo_visualization(predictions, targets, forecasts, region_names):
    """Create a comprehensive visualization for the demo."""
    
    fig, axes = plt.subplots(len(region_names), 1, figsize=(15, 4 * len(region_names)))
    if len(region_names) == 1:
        axes = [axes]
    
    fig.suptitle('Attention LSTM Regional Forecasting Demo', fontsize=16)
    
    # Show recent predictions and future forecasts
    for i, region in enumerate(region_names):
        # Plot recent test performance
        recent_samples = min(30, len(predictions))
        time_indices = range(-recent_samples, 0)
        
        axes[i].plot(time_indices, targets[-recent_samples:, i], 
                    'o-', label='Actual', color='blue', alpha=0.7, markersize=3)
        axes[i].plot(time_indices, predictions[-recent_samples:, i], 
                    's-', label='Predicted', color='red', alpha=0.7, markersize=3)
        
        # Plot future forecast
        future_indices = range(0, len(forecasts))
        axes[i].plot(future_indices, forecasts[:, i], 
                    '^-', label='Forecast', color='green', linewidth=2, markersize=4)
        
        # Add vertical line at present
        axes[i].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Present')
        
        # Calculate and display metrics
        mae = np.mean(np.abs(predictions[-recent_samples:, i] - targets[-recent_samples:, i]))
        
        axes[i].set_title(f'{region} (Recent MAE: {mae:.4f})')
        axes[i].set_xlabel('Days (negative=past, positive=future)')
        axes[i].set_ylabel('Normalized Concentration')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Highlight forecast region
        axes[i].fill_betweenx(axes[i].get_ylim(), 0, len(forecasts), 
                             alpha=0.1, color='green', label='Forecast Period')
    
    plt.tight_layout()
    plt.savefig('demo_forecast_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

def show_model_architecture():
    """Display model architecture information."""
    
    print(f"\n🏗️  MODEL ARCHITECTURE")
    print("-" * 40)
    print("Attention LSTM Regional Forecaster")
    print("├── Input Layer (5 regions × 30 days)")
    print("├── LSTM Layers (2 layers, 128 hidden units)")
    print("│   ├── Layer Normalization")
    print("│   └── Dropout (0.2)")
    print("├── Multi-Head Attention (8 heads)")
    print("│   ├── Self-attention mechanism")
    print("│   └── Residual connections")
    print("├── Feed-Forward Network")
    print("│   ├── Linear (128 → 256)")
    print("│   ├── ReLU activation")
    print("│   ├── Dropout (0.2)")
    print("│   └── Linear (256 → 128)")
    print("└── Output Layer (128 → 5 regions)")
    print()
    print("Key Features:")
    print("• Temporal attention for capturing long-range dependencies")
    print("• Regional focus for Japan microplastic concentrations")
    print("• Early stopping and learning rate scheduling")
    print("• Gradient clipping for training stability")

if __name__ == "__main__":
    show_model_architecture()
    demo_forecasting()
