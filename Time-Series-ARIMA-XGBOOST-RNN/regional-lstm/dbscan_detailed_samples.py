#!/usr/bin/env python3
"""
DBSCAN Clustering on SA-ConvLSTM Detailed Comparison Samples
===========================================================

This script applies DBSCAN clustering only to the 6 test samples that were used
to generate the detailed comparison images in sa_convlstm_comprehensive_evaluation.png.

Features:
1. Load the first 6 test predictions and ground truth samples
2. Apply DBSCAN clustering to both predictions and ground truth
3. Calculate MAE between predictions and ground truth
4. Create comprehensive comparison visualizations
5. Generate analysis summaries

Author: Assistant
Date: 2024
"""

import os
os.environ['MallocStackLogging'] = '0'
os.environ['MALLOC_STACK_LOGGING'] = '0'

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pathlib import Path
import imageio.v2 as imageio
from scipy.spatial import ConvexHull
import json
from datetime import datetime

def apply_dbscan_to_array(data_array, date_str, title_prefix="Data", threshold=0.30, eps=3, min_samples=4):
    """
    Apply DBSCAN clustering to a single data array (prediction or ground truth).
    
    Args:
        data_array: 2D numpy array (64, 64) from SA-ConvLSTM
        date_str: Date string for the data
        title_prefix: Prefix for the plot title (e.g., "Prediction", "Ground Truth")
        threshold: Concentration threshold for clustering
        eps: DBSCAN eps parameter
        min_samples: DBSCAN minimum samples parameter
    """
    print(f"Applying DBSCAN to {title_prefix.lower()} for {date_str}")
    print(f"  {title_prefix} shape: {data_array.shape}")
    print(f"  Concentration range: {np.min(data_array):.6f} to {np.max(data_array):.6f}")
    
    h, w = data_array.shape
    
    # Calculate average concentration
    avg_concentration = np.nanmean(data_array)
    print(f"  Average concentration: {avg_concentration:.6f}")
    
    # Create mask for clustering
    valid_data_mask = ~np.isnan(data_array)
    threshold_mask = data_array > threshold
    combined_mask = valid_data_mask & threshold_mask
    
    filtered_indices = np.where(combined_mask)
    X = np.column_stack((filtered_indices[0], filtered_indices[1]))
    
    if len(X) == 0:
        print(f"  No data points above threshold {threshold}")
        n_clusters, n_noise = 0, 0
        labels = np.array([])
    else:
        # Apply DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  Found {n_clusters} clusters and {n_noise} noise points")
    
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'avg_concentration': avg_concentration,
        'labels': labels,
        'points': X,
        'data_array': data_array
    }

def create_comparison_visualization(pred_data, gt_data, mae_value, date_str, sample_idx, threshold=0.30):
    """
    Create a comprehensive comparison visualization showing prediction vs ground truth
    with DBSCAN clustering results and MAE.
    """
    # Set consistent figure size and DPI to ensure uniform image dimensions
    plt.rcParams['figure.dpi'] = 100
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Japan region geographic bounds
    lon_min, lon_max = 118.86, 145.47
    lat_min, lat_max = 25.36, 36.98
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    # Plot prediction
    ax1 = axes[0, 0]
    img1 = ax1.imshow(pred_data['data_array'], aspect='auto', cmap='viridis', origin='lower', extent=extent)
    ax1.set_title(f'Prediction\nClusters: {pred_data["n_clusters"]}, Noise: {pred_data["n_noise"]}')
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    plt.colorbar(img1, ax=ax1, label='Concentration', shrink=0.8)
    
    # Add prediction clusters and get coordinate information
    pred_cluster_info = []
    if len(pred_data['points']) > 0 and pred_data['n_clusters'] > 0:
        pred_cluster_info = plot_clusters(ax1, pred_data['points'], pred_data['labels'], extent, pred_data['data_array'].shape)
    
    # Plot ground truth
    ax2 = axes[0, 1]
    img2 = ax2.imshow(gt_data['data_array'], aspect='auto', cmap='viridis', origin='lower', extent=extent)
    ax2.set_title(f'Ground Truth\nClusters: {gt_data["n_clusters"]}, Noise: {gt_data["n_noise"]}')
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    plt.colorbar(img2, ax=ax2, label='Concentration', shrink=0.8)
    
    # Add ground truth clusters and get coordinate information
    gt_cluster_info = []
    if len(gt_data['points']) > 0 and gt_data['n_clusters'] > 0:
        gt_cluster_info = plot_clusters(ax2, gt_data['points'], gt_data['labels'], extent, gt_data['data_array'].shape)
    
    # Plot error map (MAE)
    ax3 = axes[0, 2]
    error_map = np.abs(pred_data['data_array'] - gt_data['data_array'])
    img3 = ax3.imshow(error_map, aspect='auto', cmap='Reds', origin='lower', extent=extent)
    ax3.set_title(f'Absolute Error\nMAE: {mae_value:.6f}')
    ax3.set_xlabel("Longitude (°E)")
    ax3.set_ylabel("Latitude (°N)")
    plt.colorbar(img3, ax=ax3, label='Absolute Error', shrink=0.8)
    
    # Statistics comparison
    ax4 = axes[1, 0]
    ax4.axis('off')
    pred_stats = f"""Prediction Statistics:
Mean: {pred_data['avg_concentration']:.6f}
Min: {np.min(pred_data['data_array']):.6f}
Max: {np.max(pred_data['data_array']):.6f}
Std: {np.std(pred_data['data_array']):.6f}
Clusters: {pred_data['n_clusters']}
Noise Points: {pred_data['n_noise']}"""
    
    ax4.text(0.1, 0.9, pred_stats, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax5 = axes[1, 1]
    ax5.axis('off')
    gt_stats = f"""Ground Truth Statistics:
Mean: {gt_data['avg_concentration']:.6f}
Min: {np.min(gt_data['data_array']):.6f}
Max: {np.max(gt_data['data_array']):.6f}
Std: {np.std(gt_data['data_array']):.6f}
Clusters: {gt_data['n_clusters']}
Noise Points: {gt_data['n_noise']}"""
    
    ax5.text(0.1, 0.9, gt_stats, transform=ax5.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Error statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    error_stats = f"""Error Analysis:
MAE: {mae_value:.6f}
RMSE: {np.sqrt(np.mean(error_map**2)):.6f}
Max Error: {np.max(error_map):.6f}
Mean Error: {np.mean(pred_data['data_array'] - gt_data['data_array']):.6f}

Clustering Comparison:
Pred Clusters: {pred_data['n_clusters']}
GT Clusters: {gt_data['n_clusters']}
Cluster Diff: {abs(pred_data['n_clusters'] - gt_data['n_clusters'])}"""
    
    ax6.text(0.1, 0.9, error_stats, transform=ax6.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'DBSCAN Analysis: Detailed Sample {sample_idx} - {date_str}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Return figure and cluster coordinate information
    return fig, pred_cluster_info, gt_cluster_info

def plot_clusters(ax, points, labels, extent, shape):
    """Helper function to plot clusters on an axis with coordinate annotations."""
    h, w = shape
    lon_min, lon_max, lat_min, lat_max = extent
    
    unique_labels = set(labels)
    colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    cluster_info = []  # Store cluster information for return
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue  # Skip noise points
        
        class_member_mask = labels == k
        xy = points[class_member_mask]
        
        # Convert array indices to geographic coordinates
        xy_geo = np.zeros_like(xy, dtype=float)
        xy_geo[:, 1] = lon_min + (xy[:, 1] / w) * (lon_max - lon_min)  # longitude
        xy_geo[:, 0] = lat_min + (xy[:, 0] / h) * (lat_max - lat_min)  # latitude
        
        # Calculate cluster centroid
        centroid_lon = np.mean(xy_geo[:, 1])
        centroid_lat = np.mean(xy_geo[:, 0])
        
        # Store cluster information
        cluster_info.append({
            'cluster_id': int(k),
            'centroid_lon': float(centroid_lon),
            'centroid_lat': float(centroid_lat),
            'num_points': int(len(xy))
        })
        
        # Draw convex hull around clusters
        if len(xy_geo) >= 3:
            try:
                hull = ConvexHull(xy_geo)
                for simplex in hull.simplices:
                    ax.plot(xy_geo[simplex, 1], xy_geo[simplex, 0], 'r-', lw=2, alpha=0.8)
            except:
                pass  # Skip if hull can't be computed
        
        # Add cluster center marker
        ax.plot(centroid_lon, centroid_lat, 'ro', markersize=8, markeredgecolor='white', 
               markeredgewidth=2, alpha=0.9)
        
        # Add coordinate text annotation next to the cluster center
        coord_text = f'C{k}\n({centroid_lat:.2f}°N,\n{centroid_lon:.2f}°E)'
        ax.annotate(coord_text, (centroid_lon, centroid_lat), 
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8),
                   ha='left', va='bottom')
    
    return cluster_info

def load_and_analyze_detailed_samples():
    """
    Load the first 6 test samples (used for detailed comparisons) and apply DBSCAN analysis.
    """
    print("="*70)
    print("DBSCAN ANALYSIS ON SA-CONVLSTM DETAILED COMPARISON SAMPLES")
    print("="*70)
    
    # Check if test data files exist
    test_files = {
        'predictions': Path("test_predictions.npy"),
        'targets': Path("test_targets.npy"),
        'timestamps': Path("test_timestamps.npy")
    }
    
    for name, path in test_files.items():
        if not path.exists():
            print(f"❌ {name.title()} file not found: {path}")
            print("Please run sa_convlstm_microplastics.py first to generate test data!")
            return
        print(f"✓ Found {name}: {path}")
    
    # Load test data
    print("\nLoading test data...")
    predictions = np.load(test_files['predictions'])
    targets = np.load(test_files['targets'])
    timestamps = np.load(test_files['timestamps'], allow_pickle=True)
    
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Targets shape: {targets.shape}")
    print(f"✓ Total test samples: {len(timestamps)}")
    
    # Process only the first 6 samples (detailed comparison samples)
    num_detailed_samples = min(6, len(predictions))
    print(f"✓ Processing first {num_detailed_samples} samples (detailed comparison samples)")
    
    # Create output directory
    output_dir = Path("dbscan_detailed_samples")
    output_dir.mkdir(exist_ok=True)
    
    # Process each detailed sample
    results = []
    comparison_images = []
    mae_values = []
    
    print(f"\nProcessing {num_detailed_samples} detailed samples...")
    
    for i in range(num_detailed_samples):
        date_str = timestamps[i].strftime('%Y-%m-%d') if hasattr(timestamps[i], 'strftime') else str(timestamps[i])
        print(f"\n--- Processing detailed sample {i+1}/{num_detailed_samples}: {date_str} ---")
        
        # Get prediction and ground truth arrays
        pred_array = predictions[i]
        gt_array = targets[i]
        
        # Calculate MAE
        mae_value = np.mean(np.abs(pred_array - gt_array))
        mae_values.append(mae_value)
        print(f"MAE: {mae_value:.6f}")
        
        # Apply DBSCAN to both prediction and ground truth
        pred_data = apply_dbscan_to_array(pred_array, date_str, "Prediction", threshold=0.30, eps=3, min_samples=4)
        gt_data = apply_dbscan_to_array(gt_array, date_str, "Ground Truth", threshold=0.30, eps=3, min_samples=4)
        
        # Create comprehensive comparison visualization
        comparison_fig, pred_cluster_coords, gt_cluster_coords = create_comparison_visualization(pred_data, gt_data, mae_value, date_str, i+1)
        
        # Save comparison image
        comparison_file = output_dir / f"dbscan_detailed_{i+1:01d}_{date_str}.png"
        comparison_fig.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close(comparison_fig)
        
        comparison_images.append(comparison_file)
        
        # Print cluster coordinates to console
        if pred_cluster_coords:
            print(f"  Prediction Cluster Coordinates:")
            for cluster in pred_cluster_coords:
                print(f"    Cluster {cluster['cluster_id']}: ({cluster['centroid_lat']:.3f}°N, {cluster['centroid_lon']:.3f}°E) - {cluster['num_points']} points")
        
        if gt_cluster_coords:
            print(f"  Ground Truth Cluster Coordinates:")
            for cluster in gt_cluster_coords:
                print(f"    Cluster {cluster['cluster_id']}: ({cluster['centroid_lat']:.3f}°N, {cluster['centroid_lon']:.3f}°E) - {cluster['num_points']} points")
        
        # Store results
        results.append({
            'sample_index': i + 1,
            'date': date_str,
            'mae': float(mae_value),
            'prediction': {
                'clusters': pred_data['n_clusters'],
                'noise_points': pred_data['n_noise'],
                'avg_concentration': float(pred_data['avg_concentration']),
                'min_concentration': float(np.min(pred_array)),
                'max_concentration': float(np.max(pred_array)),
                'std_concentration': float(np.std(pred_array)),
                'cluster_coordinates': pred_cluster_coords
            },
            'ground_truth': {
                'clusters': gt_data['n_clusters'],
                'noise_points': gt_data['n_noise'],
                'avg_concentration': float(gt_data['avg_concentration']),
                'min_concentration': float(np.min(gt_array)),
                'max_concentration': float(np.max(gt_array)),
                'std_concentration': float(np.std(gt_array)),
                'cluster_coordinates': gt_cluster_coords
            },
            'clustering_comparison': {
                'cluster_difference': abs(pred_data['n_clusters'] - gt_data['n_clusters']),
                'noise_difference': abs(pred_data['n_noise'] - gt_data['n_noise'])
            },
            'file': str(comparison_file)
        })
        
        print(f"✓ Saved comparison: {comparison_file.name}")
    
    # Create summary statistics
    print("\n" + "="*50)
    print("CREATING SUMMARY ANALYSIS")
    print("="*50)
    
    # Calculate overall statistics
    overall_mae = np.mean(mae_values)
    mae_std = np.std(mae_values)
    
    pred_clusters = [r['prediction']['clusters'] for r in results]
    gt_clusters = [r['ground_truth']['clusters'] for r in results]
    cluster_differences = [r['clustering_comparison']['cluster_difference'] for r in results]
    
    print(f"Overall MAE: {overall_mae:.6f} ± {mae_std:.6f}")
    print(f"MAE Range: {np.min(mae_values):.6f} - {np.max(mae_values):.6f}")
    print(f"Average Prediction Clusters: {np.mean(pred_clusters):.1f}")
    print(f"Average Ground Truth Clusters: {np.mean(gt_clusters):.1f}")
    print(f"Average Cluster Difference: {np.mean(cluster_differences):.1f}")
    
    # Create animated GIF with proper size handling
    print("\nCreating animated comparison GIF...")
    if comparison_images:
        gif_images = []
        target_size = None
        
        # First pass: determine the target size (largest dimensions)
        for file in comparison_images:
            img = imageio.imread(file)
            if target_size is None:
                target_size = img.shape
            else:
                # Use the largest dimensions
                target_size = (max(target_size[0], img.shape[0]), 
                             max(target_size[1], img.shape[1]),
                             img.shape[2])  # Keep color channels the same
        
        print(f"Target image size for GIF: {target_size}")
        
        # Second pass: resize all images to the same size
        for file in comparison_images:
            img = imageio.imread(file)
            
            # If image is smaller than target, pad it with white background
            if img.shape != target_size:
                # Create white background image with target size
                padded_img = np.ones(target_size, dtype=img.dtype) * 255
                
                # Calculate padding to center the image
                pad_h = (target_size[0] - img.shape[0]) // 2
                pad_w = (target_size[1] - img.shape[1]) // 2
                
                # Place the original image in the center
                padded_img[pad_h:pad_h + img.shape[0], 
                          pad_w:pad_w + img.shape[1]] = img
                gif_images.append(padded_img)
            else:
                gif_images.append(img)
        
        gif_path = output_dir / "detailed_samples_animation.gif"
        imageio.mimsave(gif_path, gif_images, duration=3.0)  # 3 seconds per frame
        print(f"✓ Animation saved: {gif_path}")
    
    # Create summary plot
    print("Creating summary plots...")
    create_summary_plots(results, mae_values, output_dir)
    
    # Save comprehensive results
    summary = {
        'script_info': {
            'name': 'DBSCAN Analysis on SA-ConvLSTM Detailed Samples',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': f'Analysis of first {num_detailed_samples} test samples used for detailed comparisons'
        },
        'data_info': {
            'num_detailed_samples': num_detailed_samples,
            'data_shape': list(predictions[0].shape),
            'date_range': [str(timestamps[0]), str(timestamps[num_detailed_samples-1])]
        },
        'mae_analysis': {
            'overall_mae': float(overall_mae),
            'mae_std': float(mae_std),
            'min_mae': float(np.min(mae_values)),
            'max_mae': float(np.max(mae_values)),
            'mae_values': [float(x) for x in mae_values]
        },
        'clustering_analysis': {
            'avg_pred_clusters': float(np.mean(pred_clusters)),
            'avg_gt_clusters': float(np.mean(gt_clusters)),
            'avg_cluster_difference': float(np.mean(cluster_differences)),
            'total_samples_with_pred_clusters': int(np.sum([1 for x in pred_clusters if x > 0])),
            'total_samples_with_gt_clusters': int(np.sum([1 for x in gt_clusters if x > 0]))
        },
        'detailed_results': results
    }
    
    summary_file = output_dir / "detailed_samples_analysis.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("DBSCAN DETAILED SAMPLES ANALYSIS COMPLETED")
    print("="*70)
    print(f"✓ Processed {len(results)} detailed samples")
    print(f"✓ Overall MAE: {overall_mae:.6f}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Generated {len(comparison_images)} comparison images")
    print(f"✓ Created animated GIF: detailed_samples_animation.gif")
    print(f"✓ Summary saved: {summary_file}")
    print("="*70)
    
    return results, mae_values

def create_summary_plots(results, mae_values, output_dir):
    """Create summary plots for the detailed samples analysis."""
    
    # Summary plot for the 6 detailed samples
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # MAE bar chart
    ax1 = axes[0, 0]
    sample_indices = [r['sample_index'] for r in results]
    dates = [r['date'] for r in results]
    ax1.bar(sample_indices, mae_values, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('MAE for Each Detailed Sample')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('MAE')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(mae_values), color='r', linestyle='--', label=f'Mean: {np.mean(mae_values):.6f}')
    ax1.legend()
    
    # Set x-axis labels to sample indices
    ax1.set_xticks(sample_indices)
    
    # Cluster comparison bar chart
    ax2 = axes[0, 1]
    pred_clusters = [r['prediction']['clusters'] for r in results]
    gt_clusters = [r['ground_truth']['clusters'] for r in results]
    
    x = np.arange(len(sample_indices))
    width = 0.35
    
    ax2.bar(x - width/2, pred_clusters, width, label='Prediction', alpha=0.7, color='lightblue')
    ax2.bar(x + width/2, gt_clusters, width, label='Ground Truth', alpha=0.7, color='lightgreen')
    
    ax2.set_title('Cluster Count Comparison')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Number of Clusters')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sample_indices)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Concentration comparison
    ax3 = axes[1, 0]
    pred_avg_conc = [r['prediction']['avg_concentration'] for r in results]
    gt_avg_conc = [r['ground_truth']['avg_concentration'] for r in results]
    
    ax3.scatter(gt_avg_conc, pred_avg_conc, alpha=0.8, color='purple', s=100, edgecolor='black')
    ax3.plot([min(min(pred_avg_conc), min(gt_avg_conc)), max(max(pred_avg_conc), max(gt_avg_conc))], 
             [min(min(pred_avg_conc), min(gt_avg_conc)), max(max(pred_avg_conc), max(gt_avg_conc))], 
             'r--', label='Perfect Prediction')
    ax3.set_xlabel('Ground Truth Avg Concentration')
    ax3.set_ylabel('Predicted Avg Concentration')
    ax3.set_title('Average Concentration Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add sample numbers as labels
    for i, (x_val, y_val) in enumerate(zip(gt_avg_conc, pred_avg_conc)):
        ax3.annotate(f'S{i+1}', (x_val, y_val), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    # Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    cluster_diffs = [r['clustering_comparison']['cluster_difference'] for r in results]
    stats_text = f"""Summary Statistics:
    
Samples Analyzed: {len(results)}
Date Range: {dates[0]} to {dates[-1]}

MAE Statistics:
  Mean: {np.mean(mae_values):.6f}
  Std:  {np.std(mae_values):.6f}
  Min:  {np.min(mae_values):.6f}
  Max:  {np.max(mae_values):.6f}

Clustering Statistics:
  Avg Pred Clusters: {np.mean(pred_clusters):.1f}
  Avg GT Clusters:   {np.mean(gt_clusters):.1f}
  Avg Cluster Diff:  {np.mean(cluster_diffs):.1f}
  
Samples with Clusters:
  Predictions: {np.sum([1 for x in pred_clusters if x > 0])}/6
  Ground Truth: {np.sum([1 for x in gt_clusters if x > 0])}/6"""
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('DBSCAN Analysis Summary: Detailed Comparison Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_plot_path = output_dir / "detailed_samples_summary.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary plots saved: {summary_plot_path}")

if __name__ == "__main__":
    load_and_analyze_detailed_samples()