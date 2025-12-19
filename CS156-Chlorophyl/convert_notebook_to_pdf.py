#!/usr/bin/env python3
"""
Script to convert Jupyter notebook to PDF.

This script uses nbconvert to convert a Jupyter notebook to PDF format.
It handles the conversion process and provides error messages if something goes wrong.
"""

import sys
import os
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import nbconvert
    except ImportError:
        print("Error: nbconvert is not installed.")
        print("Please install it using: pip install nbconvert")
        return False
    
    # Check if Chrome is installed (required for PDF conversion)
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "google-chrome",
        "chromium",
        "google-chrome-stable"
    ]
    
    chrome_found = False
    for chrome_path in chrome_paths:
        try:
            result = subprocess.run(
                ['which', chrome_path] if '/' not in chrome_path else ['test', '-f', chrome_path],
                capture_output=True
            )
            if result.returncode == 0:
                chrome_found = True
                break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    if not chrome_found:
        print("Warning: Chrome/Chromium not found. PDF conversion may fail.")
        print("Please install Google Chrome for PDF conversion.")
        return False
    
    return True


def find_chrome():
    """Find Chrome/Chromium executable."""
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "google-chrome",
        "chromium",
        "google-chrome-stable"
    ]
    
    for chrome_path in chrome_paths:
        try:
            if '/' in chrome_path:
                # Absolute path
                if Path(chrome_path).exists():
                    return chrome_path
            else:
                # Command in PATH
                result = subprocess.run(
                    ['which', chrome_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception:
            continue
    
    return None


def convert_notebook_to_pdf(notebook_path, output_path=None):
    """
    Convert a Jupyter notebook to PDF via HTML.
    
    Args:
        notebook_path: Path to the input notebook file
        output_path: Optional path for the output PDF file
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"Error: Notebook file not found: {notebook_path}")
        return False
    
    if not notebook_path.suffix == '.ipynb':
        print(f"Error: File is not a Jupyter notebook: {notebook_path}")
        return False
    
    # Set output path if not provided
    if output_path is None:
        output_path = notebook_path.with_suffix('.pdf')
    else:
        output_path = Path(output_path)
    
    html_path = notebook_path.with_suffix('.html')
    
    print(f"Converting {notebook_path.name} to PDF...")
    print(f"Output will be saved to: {output_path}")
    
    try:
        # Step 1: Convert notebook to HTML
        print("\nStep 1: Converting to HTML...")
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'html',
            '--output', str(html_path.stem),
            '--output-dir', str(html_path.parent),
            str(notebook_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if not html_path.exists():
            print(f"Error: HTML file not created: {html_path}")
            return False
        
        print("  ✓ HTML created successfully")
        
        # Step 2: Find Chrome
        chrome = find_chrome()
        if not chrome:
            print("\nError: Chrome/Chromium not found")
            print("Please install Google Chrome to convert to PDF")
            html_path.unlink()
            return False
        
        # Step 3: Convert HTML to PDF using Chrome
        print("\nStep 2: Converting HTML to PDF using Chrome...")
        cmd = [
            chrome,
            '--headless',
            '--disable-gpu',
            f'--print-to-pdf={output_path}',
            '--print-to-pdf-no-header',
            str(html_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        # Clean up HTML file
        if html_path.exists():
            html_path.unlink()
            print("  ✓ Cleaned up temporary HTML file")
        
        if not output_path.exists():
            print(f"\n✗ Error: PDF file not created")
            return False
        
        # Get file size
        size_bytes = output_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"\n✓ Successfully converted notebook to PDF!")
        print(f"  PDF saved to: {output_path}")
        print(f"  Size: {size_mb:.1f} MB")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during conversion:")
        print(f"  Return code: {e.returncode}")
        if e.stdout:
            print(f"  stdout: {e.stdout}")
        if e.stderr:
            print(f"  stderr: {e.stderr}")
        
        # Clean up HTML file if it exists
        if html_path.exists():
            html_path.unlink()
        
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        
        # Clean up HTML file if it exists
        if html_path.exists():
            html_path.unlink()
        
        return False


def main():
    """Main function."""
    # Default notebook path
    default_notebook = Path(__file__).parent / 'chlorophyll_convlstm_dbscan_complete.ipynb'
    
    # Get notebook path from command line argument or use default
    if len(sys.argv) > 1:
        notebook_path = Path(sys.argv[1])
    else:
        notebook_path = default_notebook
    
    # Get output path from command line if provided
    output_path = None
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    
    print("=" * 60)
    print("Jupyter Notebook to PDF Converter")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    print()
    
    # Convert notebook
    success = convert_notebook_to_pdf(notebook_path, output_path)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
