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

# Import the existing model and functions from sa_tsushima.py
from sa_convlstm import SA_ConvLSTM_Model
from sa_tsushima import extract_data_and_clusters, create_sequences_from_data, temporal_train_test_split_proper
from attention_analysis_final import (
    occlusion_sensitivity_analysis, 
    map_region_to_grid, 
    extract_geographic_coords_from_data
)

def create_test_data_for_validation(nc_files, num_tests=20):
    """
    Create test data with actual targets for validation against real data.
    """
    print(f"Creating {num_tests} test cases with actual targets from real data...")
    
    # Load all data using sa_tsushima functions
    all_data = extract_data_and_clusters(nc_files)
    print(f"Total data points: {len(all_data)}")
    
    # Create proper temporal split to get test data
    frame_num = 3  # Use 3 frames for sequence
    train_data, test_data, train_end_idx, test_start_idx = temporal_train_test_split_proper(
        all_data, frame_num, test_ratio=0.2
    )
    
    # Create sequences from test data
    X_test, y_test = create_sequences_from_data(test_data, frame_num)
    
    print(f"Test sequences available: {len(X_test)}")
    print(f"Test data spans from index {test_start_idx} to {len(all_data)-1}")
    
    # Sample evenly from test data
    if len(X_test) < num_tests:
        print(f"Warning: Only {len(X_test)} test sequences available, using all of them")
        num_tests = len(X_test)
    
    # Create evenly spaced indices from test data
    indices = np.linspace(0, len(X_test)-1, num_tests, dtype=int)
    
    test_cases = []
    for i, idx in enumerate(indices):
        test_cases.append({
            'sequence': X_test[idx],
            'actual_target': y_test[idx],
            'test_day': i + 1,
            'original_test_idx': idx,
            'global_idx': test_start_idx + idx
        })
    
    print(f"Created {len(test_cases)} test cases with actual targets")
    return test_cases

def run_occlusion_for_test_case(model, test_case, target_region_pixels, patch_size=4):
    """
    Run occlusion analysis for a test case and compare with actual data.
    """
    # Prepare input sequence
    input_sequence = torch.FloatTensor(test_case['sequence'][np.newaxis, ...]).to(device)
    
    # Get baseline prediction
    model.eval()
    with torch.no_grad():
        baseline_prediction = model(input_sequence)[:, -1, 0, :, :].squeeze().cpu().numpy()
    
    # Run occlusion sensitivity analysis
    sensitivity_map = occlusion_sensitivity_analysis(
        model, input_sequence, target_region_pixels, patch_size=patch_size
    )
    
    # Get actual target
    actual_target = test_case['actual_target'].squeeze()  # Remove channel dimension
    
    # Calculate prediction accuracy metrics
    mae_baseline = np.mean(np.abs(baseline_prediction - actual_target))
    
    return {
        'sensitivity_map': sensitivity_map,
        'baseline_prediction': baseline_prediction,
        'actual_target': actual_target,
        'mae_baseline': mae_baseline
    }

def create_validation_comparison_image(result, target_region_pixels, test_day, img_size=64, 
                                     threshold_percentile=60, global_vmin=None, global_vmax=None):
    """
    Create comparison image showing: Actual, Predicted, Occlusion Sensitivity, and Error.
    """
    sensitivity_map = result['sensitivity_map']
    prediction = result['baseline_prediction']
    actual = result['actual_target']
    mae = result['mae_baseline']
    
    # Always use per-frame percentile threshold for relative importance (60th percentile = top 40%)
    threshold = np.percentile(sensitivity_map, threshold_percentile)
    thresholded = sensitivity_map.copy()
    thresholded[thresholded < threshold] = 0
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Tsushima Test Case {test_day}: Model Validation vs Occlusion Analysis\nMAE: {mae:.6f}', fontsize=16)
    
    # 1. Actual data
    im1 = axes[0, 0].imshow(np.flipud(actual), cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Actual Data\n(Ground Truth)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 2. Model prediction
    im2 = axes[0, 1].imshow(np.flipud(prediction), cmap='viridis', aspect='auto')
    axes[0, 1].set_title('SA-ConvLSTM Prediction\n(Baseline)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Occlusion sensitivity with fixed colorbar range
    if global_vmin is not None and global_vmax is not None:
        im3 = axes[1, 0].imshow(np.flipud(thresholded), cmap='hot', aspect='auto', 
                              vmin=global_vmin, vmax=global_vmax)
    else:
        im3 = axes[1, 0].imshow(np.flipud(thresholded), cmap='hot', aspect='auto')
    
    axes[1, 0].set_title('Top 40% Most Influential Areas\n(Occlusion Sensitivity)')
    axes[1, 0].axis('off')
    
    # Add target region overlay to occlusion map (Tsushima region)
    lat_range = target_region_pixels['lat_range']
    lon_range = target_region_pixels['lon_range']
    
    rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                       lon_range[1] - lon_range[0], 
                       lat_range[1] - lat_range[0],
                       linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
    axes[1, 0].add_patch(rect)
    
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar3.set_label('Occlusion Sensitivity', rotation=270, labelpad=15)
    
    # 4. Prediction error
    error = np.abs(prediction - actual)
    im4 = axes[1, 1].imshow(np.flipud(error), cmap='Reds', aspect='auto')
    axes[1, 1].set_title(f'Prediction Error\n(|Predicted - Actual|)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save the image
    output_dir = Path('tsushima_validation_frames')
    output_dir.mkdir(exist_ok=True)
    
    filename = output_dir / f'tsushima_validation_test_{test_day:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def create_gif_from_images(image_dir, output_filename='tsushima_validation_analysis.gif', duration=0.8):
    """
    Create a GIF from the sequence of validation comparison images.
    """
    image_files = sorted(Path(image_dir).glob('tsushima_validation_test_*.png'))
    
    if not image_files:
        print("No validation images found to create GIF!")
        return None
    
    print(f"Creating GIF from {len(image_files)} validation images...")
    
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
        
        if i % 10 == 0:
            print(f"  Processed {i+1}/{len(image_files)} images...")
    
    # Create GIF
    output_path = Path(output_filename)
    imageio.v2.mimsave(output_path, images, duration=duration, loop=0)
    
    print(f"GIF saved as: {output_path}")
    return output_path

def main():
    """Main function to create occlusion analysis validation against real test data for Tsushima model."""
    
    print("="*70)
    print("TSUSHIMA OCCLUSION SENSITIVITY VALIDATION AGAINST REAL DATA")
    print("="*70)
    
    # Target region coordinates (Tsushima area) - as specified by user
    target_sw_lat, target_sw_lon = 34.02837, 129.11613
    target_ne_lat, target_ne_lon = 34.76456, 129.55801
    
    print(f"Target Region (Tsushima area):")
    print(f"  Southwest: ({target_sw_lat}, {target_sw_lon})")
    print(f"  Northeast: ({target_ne_lat}, {target_ne_lon})")
    
    # Load geographic coordinates
    print("\nExtracting geographic coordinates...")
    try:
        full_lats, full_lons = extract_geographic_coords_from_data()
        print(f"Coordinate grid shape: {full_lats.shape} x {full_lons.shape}")
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        return
    
    # Set up model configuration (matching sa_tsushima.py)
    class Args:
        def __init__(self):
            self.batch_size = 8
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
    
    # Load trained model from sa_tsushima.py
    model_path = 'sa_convlstm_japan_microplastics.pth'
    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found!")
        print("Please run sa_tsushima.py first to generate the model weights.")
        return
    
    print(f"\nLoading trained SA-ConvLSTM model from sa_tsushima.py...")
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
    
    # Create test data with actual targets (limited to 20 cases for faster processing)
    num_tests = 20  # Limit to 20 test cases for speed
    print(f"\nCreating {num_tests} test cases with actual targets for validation...")
    
    test_cases = create_test_data_for_validation(nc_files, num_tests)
    
    # First pass: Calculate global statistics and run analysis
    print("\nFirst pass: Running occlusion analysis and calculating global statistics...")
    all_results = []
    all_sensitivity_maps = []
    total_mae = 0.0
    
    for i, test_case in enumerate(test_cases):
        if i % 5 == 0:
            print(f"Analyzing test case {i+1}/{len(test_cases)} (Test Day {test_case['test_day']})...")
        
        # Run occlusion analysis and get validation results
        result = run_occlusion_for_test_case(
            model, test_case, target_region_pixels, patch_size=4
        )
        all_results.append(result)
        all_sensitivity_maps.append(result['sensitivity_map'])
        total_mae += result['mae_baseline']
    
    # Calculate global statistics for consistent colorbar scaling
    print("\nCalculating global color scale and validation metrics...")
    all_maps_array = np.array(all_sensitivity_maps)
    global_vmin = np.min(all_maps_array)
    global_vmax = np.max(all_maps_array)
    average_mae = total_mae / len(test_cases)
    
    print(f"Global sensitivity range: {global_vmin:.4f} to {global_vmax:.4f}")
    print(f"Average prediction MAE: {average_mae:.6f}")
    print("Note: Each frame shows actual vs predicted data with occlusion analysis")
    
    # Second pass: Create validation comparison images
    print("\nSecond pass: Creating validation comparison images...")
    
    for i, (test_case, result) in enumerate(zip(test_cases, all_results)):
        if i % 5 == 0:
            print(f"Creating comparison {i+1}/{len(test_cases)} (Test Day {test_case['test_day']})...")
        
        # Create and save validation comparison image
        image_file = create_validation_comparison_image(
            result, target_region_pixels, test_case['test_day'],
            global_vmin=global_vmin, global_vmax=global_vmax
        )
    
    # Create GIF from all validation images
    print("\nCreating GIF from all validation comparison images...")
    gif_path = create_gif_from_images(
        'tsushima_validation_frames', 
        'tsushima_validation_analysis.gif', 
        duration=0.8  # Slower for detailed inspection
    )
    
    print("\n" + "="*70)
    print("TSUSHIMA OCCLUSION VALIDATION ANALYSIS COMPLETE")
    print("="*70)
    print(f"Generated files:")
    print(f"✓ {len(test_cases)} validation comparison images in 'tsushima_validation_frames/'")
    print(f"✓ tsushima_validation_analysis.gif")
    print(f"\nValidation Results:")
    print(f"✓ Average prediction MAE: {average_mae:.6f}")
    print(f"✓ Test cases analyzed: {len(test_cases)}")
    print(f"✓ Occlusion sensitivity range: {global_vmin:.4f} to {global_vmax:.4f}")
    print("\nInterpretation:")
    print("- Each frame shows: Actual Data | Model Prediction | Occlusion Sensitivity | Prediction Error")
    print("- This validates how well occlusion analysis explains model behavior")
    print("- Lower prediction errors suggest the model is working correctly")
    print("- Occlusion sensitivity shows which areas most influence the Tsushima region")
    print("- The cyan dashed box shows the target region (Tsushima)")
    print("- Each frame shows the top 40% most influential areas FOR THAT TEST CASE")
    print("- Compare occlusion patterns with prediction errors to validate model insights")
    print("\nSpecific to Tsushima Model:")
    print("- Uses the narrowed Japan region from sa_tsushima.py")
    print("- Target region: Tsushima Island area (34.02837°N-34.76456°N, 129.11613°E-129.55801°E)")
    print("- Model trained specifically on the focused geographic region")

if __name__ == "__main__":
    main()