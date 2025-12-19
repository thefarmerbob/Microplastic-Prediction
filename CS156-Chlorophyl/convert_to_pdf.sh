#!/bin/bash
# Simple shell script to convert Jupyter notebook to PDF via HTML

# Default notebook path
NOTEBOOK="chlorophyll_convlstm_dbscan_complete.ipynb"

# Use command line argument if provided
if [ $# -gt 0 ]; then
    NOTEBOOK="$1"
fi

# Check if notebook exists
if [ ! -f "$NOTEBOOK" ]; then
    echo "Error: Notebook file not found: $NOTEBOOK"
    exit 1
fi

echo "Converting $NOTEBOOK to PDF..."
echo ""

# Get base name without extension
BASENAME="${NOTEBOOK%.ipynb}"
HTML_FILE="${BASENAME}.html"
PDF_FILE="${BASENAME}.pdf"

# Step 1: Convert to HTML
echo "Step 1: Converting to HTML..."
jupyter nbconvert --to html "$NOTEBOOK"

if [ $? -ne 0 ]; then
    echo "✗ Error converting to HTML"
    exit 1
fi

# Step 2: Convert HTML to PDF using Chrome
echo "Step 2: Converting HTML to PDF..."

# Find Chrome
CHROME=""
if [ -f "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
    CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
elif command -v google-chrome &> /dev/null; then
    CHROME="google-chrome"
elif command -v chromium &> /dev/null; then
    CHROME="chromium"
else
    echo "✗ Error: Chrome/Chromium not found"
    echo "Please install Google Chrome to convert to PDF"
    rm -f "$HTML_FILE"
    exit 1
fi

# Convert HTML to PDF
"$CHROME" --headless --disable-gpu --print-to-pdf="$PDF_FILE" --print-to-pdf-no-header "$HTML_FILE" 2>/dev/null

if [ $? -eq 0 ] && [ -f "$PDF_FILE" ]; then
    echo ""
    echo "✓ Successfully converted notebook to PDF!"
    echo "  PDF saved to: $PDF_FILE"
    echo "  Size: $(du -h "$PDF_FILE" | cut -f1)"
    
    # Clean up HTML file
    rm -f "$HTML_FILE"
    echo "  Cleaned up temporary HTML file"
else
    echo ""
    echo "✗ Error during PDF conversion"
    rm -f "$HTML_FILE"
    exit 1
fi
