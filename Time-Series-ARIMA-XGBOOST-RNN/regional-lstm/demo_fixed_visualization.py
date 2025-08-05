#!/usr/bin/env python3
"""
Demonstration script showing the fixed visualization with:
1. Fixed colorbar scale (consistent across frames)
2. Fixed mapping threshold (global calculation)
3. 2:1 aspect ratio for better map representation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_demo_sensitivity_maps():
    """Create demo sensitivity maps to show the fixes."""
    # Simulate 5 different sensitivity maps with varying intensity patterns
    maps = []
    np.random.seed(42)  # For reproducible demo
    
    for i in range(5):
        # Create base sensitivity pattern
        x, y = np.meshgrid(np.linspace(0, 10, 64), np.linspace(0, 10, 64))
        
        # Different patterns for each "day"
        if i == 0:
            sensitivity = np.exp(-((x-5)**2 + (y-3)**2)/4) * 0.8
        elif i == 1:
            sensitivity = np.exp(-((x-3)**2 + (y-7)**2)/3) * 0.6
        elif i == 2:
            sensitivity = np.exp(-((x-7)**2 + (y-2)**2)/5) * 1.0
        elif i == 3:
            sensitivity = np.exp(-((x-2)**2 + (y-8)**2)/2) * 0.7
        else:
            sensitivity = np.exp(-((x-8)**2 + (y-5)**2)/6) * 0.9
        
        # Add some noise
        sensitivity += np.random.normal(0, 0.05, sensitivity.shape)
        sensitivity = np.maximum(sensitivity, 0)  # Ensure non-negative
        
        maps.append(sensitivity)
    
    return maps

def create_demo_occlusion_image_old(sensitivity_map, day, img_size=64, threshold_percentile=80):
    """
    OLD VERSION: Create image with variable colorbar and threshold (problems).
    """
    # Apply threshold per frame (PROBLEM: inconsistent)
    threshold = np.percentile(sensitivity_map, threshold_percentile)
    thresholded = sensitivity_map.copy()
    thresholded[thresholded < threshold] = 0
    
    # Create figure (PROBLEM: square aspect ratio)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot with variable scale (PROBLEM: colorbar changes each frame)
    im = ax.imshow(np.flipud(thresholded), cmap='hot', aspect='auto')
    ax.set_title(f'OLD: Day {day} - Variable Scale & Square\n(Each frame has different colorbar)', fontsize=12)
    ax.axis('off')
    
    # Add target region overlay (example region)
    lat_range = [20, 40]
    lon_range = [15, 35]
    rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                       lon_range[1] - lon_range[0], 
                       lat_range[1] - lat_range[0],
                       linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Add colorbar (variable scale - PROBLEM)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Save the image
    output_dir = Path('demo_old_frames')
    output_dir.mkdir(exist_ok=True)
    
    filename = output_dir / f'old_day_{day:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def create_demo_occlusion_image_fixed(sensitivity_map, day, img_size=64, threshold_percentile=80, 
                                    global_vmin=None, global_vmax=None, fixed_threshold=None):
    """
    FIXED VERSION: Create image with fixed colorbar, threshold, and proper aspect ratio.
    """
    # Apply fixed threshold (FIXED: consistent across frames)
    if fixed_threshold is not None:
        threshold = fixed_threshold
    else:
        threshold = np.percentile(sensitivity_map, threshold_percentile)
    
    thresholded = sensitivity_map.copy()
    thresholded[thresholded < threshold] = 0
    
    # Create figure with 2:1 aspect ratio (FIXED: proper map representation)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot with fixed colorbar range (FIXED: consistent scale)
    if global_vmin is not None and global_vmax is not None:
        im = ax.imshow(np.flipud(thresholded), cmap='hot', aspect='auto', 
                      vmin=global_vmin, vmax=global_vmax)
    else:
        im = ax.imshow(np.flipud(thresholded), cmap='hot', aspect='auto')
    
    ax.set_title(f'FIXED: Day {day} - Fixed Scale & 2:1 Ratio\n(Consistent colorbar & map proportions)', fontsize=12)
    ax.axis('off')
    
    # Add target region overlay (example region)
    lat_range = [20, 40]
    lon_range = [15, 35]
    rect = plt.Rectangle((lon_range[0], img_size - lat_range[1]), 
                       lon_range[1] - lon_range[0], 
                       lat_range[1] - lat_range[0],
                       linewidth=3, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Add colorbar with fixed scale and label (FIXED)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Occlusion Sensitivity', rotation=270, labelpad=15)
    
    # Save the image
    output_dir = Path('demo_fixed_frames')
    output_dir.mkdir(exist_ok=True)
    
    filename = output_dir / f'fixed_day_{day:03d}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def main():
    """Demonstrate the fixes."""
    print("="*60)
    print("DEMONSTRATION OF VISUALIZATION FIXES")
    print("="*60)
    
    # Create demo data
    print("Creating demo sensitivity maps...")
    sensitivity_maps = create_demo_sensitivity_maps()
    
    # Calculate global statistics for fixed version (NEW APPROACH)
    print("Calculating global statistics for consistent scaling...")
    all_maps_array = np.array(sensitivity_maps)
    global_vmin = np.min(all_maps_array)
    global_vmax = np.max(all_maps_array)
    fixed_threshold = np.percentile(all_maps_array, 80)
    
    print(f"Global value range: {global_vmin:.4f} to {global_vmax:.4f}")
    print(f"Fixed threshold (80th percentile): {fixed_threshold:.4f}")
    
    # Create comparison images
    print("\nCreating comparison images...")
    
    for i, sensitivity_map in enumerate(sensitivity_maps):
        day = i + 1
        
        # Create old version (with problems)
        print(f"Creating old version for day {day}...")
        create_demo_occlusion_image_old(sensitivity_map, day)
        
        # Create fixed version
        print(f"Creating fixed version for day {day}...")
        create_demo_occlusion_image_fixed(
            sensitivity_map, day,
            global_vmin=global_vmin, 
            global_vmax=global_vmax, 
            fixed_threshold=fixed_threshold
        )
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("Generated files:")
    print("✓ demo_old_frames/    - Old version (problems)")
    print("✓ demo_fixed_frames/  - Fixed version (improvements)")
    print("\nKey improvements in fixed version:")
    print("1. ✓ Fixed colorbar scale (same min/max across all frames)")
    print("2. ✓ Fixed mapping threshold (consistent 80th percentile)")
    print("3. ✓ 2:1 aspect ratio (12x6 instead of 8x8 for better map representation)")
    print("4. ✓ Proper colorbar labeling")

if __name__ == "__main__":
    main()