#!/usr/bin/env python3
"""
Script to create GIFs using only the last 365 images from the timeseries-images directory.
This script generates both the regular and cropped versions with only the most recent year of data.
"""

import imageio
from pathlib import Path
import glob
import sys

def get_last_n_images(image_dir, n=100):
    """
    Get the last n images from the directory, sorted by filename (which corresponds to date).
    
    Args:
        image_dir (str): Path to the directory containing images
        n (int): Number of most recent images to select
    
    Returns:
        list: List of Path objects for the last n images
    """
    image_dir = Path(image_dir)
    
    # Get all PNG files in the directory
    all_images = sorted(image_dir.glob("timeseries_*.png"))
    
    print(f"Found {len(all_images)} total images in {image_dir}")
    
    if len(all_images) < n:
        print(f"Warning: Only {len(all_images)} images available, using all of them")
        return all_images
    
    # Get the last n images
    last_n_images = all_images[-n:]
    
    print(f"Selected last {len(last_n_images)} images")
    print(f"Date range: {last_n_images[0].stem.split('_')[1].split('.')[2][1:]} to {last_n_images[-1].stem.split('_')[1].split('.')[2][1:]}")
    
    return last_n_images

def create_gif_from_images(image_files, output_filename, duration=0.8):
    """
    Create a GIF from a list of image files.
    
    Args:
        image_files (list): List of Path objects for image files
        output_filename (str): Name of the output GIF file
        duration (float): Duration between frames in seconds
    """
    print(f"Creating GIF: {output_filename}")
    print(f"Using {len(image_files)} images with {duration}s duration per frame")
    
    # Load images
    images = []
    for i, image_file in enumerate(image_files):
        if i % 50 == 0:  # Progress indicator
            print(f"Loading image {i+1}/{len(image_files)}")
        images.append(imageio.imread(image_file))
    
    # Create GIF
    imageio.mimsave(output_filename, images, duration=duration)
    print(f"✅ Successfully created: {output_filename}")
    print(f"   Total frames: {len(images)}")
    print(f"   Total duration: {len(images) * duration:.1f} seconds")

def main():
    """Main function to create GIFs with last 365 images."""
    
    print("="*60)
    print("CREATING GIFS WITH LAST 365 IMAGES")
    print("="*60)
    
    # Set up paths
    image_dir = Path("timeseries-images")
    
    # Check if directory exists
    if not image_dir.exists():
        print(f"❌ Error: Directory {image_dir} does not exist!")
        print("Please make sure you're running this script from the correct directory.")
        sys.exit(1)
    
    # Get the last 365 images
    try:
        last_365_images = get_last_n_images(image_dir, n=365)
        
        if not last_365_images:
            print("❌ Error: No images found!")
            sys.exit(1)
        
        # Create both versions of the GIF
        print("\n" + "="*40)
        print("Creating regular GIF...")
        create_gif_from_images(
            last_365_images, 
            "microplastic_timeseries_last365.gif",
            duration=0.8
        )
        
        print("\n" + "="*40)
        print("Creating fast version GIF...")
        create_gif_from_images(
            last_365_images, 
            "microplastic_timeseries_last365_fast.gif",
            duration=0.4
        )
        
        print("\n" + "="*60)
        print("✅ ALL GIFS CREATED SUCCESSFULLY!")
        print("="*60)
        print("Created files:")
        print("  • microplastic_timeseries_last365.gif (normal speed)")
        print("  • microplastic_timeseries_last365_fast.gif (2x speed)")
        print(f"  • Both contain the last 365 days of data ({len(last_365_images)} frames)")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()