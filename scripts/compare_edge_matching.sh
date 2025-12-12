#!/bin/bash

# Comparison script for edge-aware vs regular matching
# Shows the difference in quality and performance

echo "=== Video Mosaic Edge-Aware Matching Comparison ==="
echo ""
echo "This script compares three versions:"
echo "1. video_mosaic - Multi-region color matching (fast)"
echo "2. video_mosaic_edge - Edge-aware matching (quality)"
echo "3. video_mosaic_optimal - Optimal assignment (best quality, slow)"
echo ""

# Build all versions
echo "Building all versions..."
make video_mosaic
make video_mosaic_edge
make video_mosaic_optimal

echo ""
echo "=== Ready to Test! ==="
echo ""
echo "Test commands:"
echo ""
echo "# 1. Regular multi-region (baseline)"
echo "./video_mosaic --medium -j 8 -d pokemon_tiles"
echo ""
echo "# 2. Edge-aware (better structure preservation)"
echo "./video_mosaic_edge --medium -j 8 -d pokemon_tiles"
echo ""
echo "# 3. Optimal (best quality, slowest)"
echo "./video_mosaic_optimal --medium -j 8 -d pokemon_tiles"
echo ""
echo "=== What to Look For ==="
echo ""
echo "Edge-aware version should have:"
echo "- Sharper edges (horizontal/vertical lines preserved)"
echo "- Better texture matching (patterns aligned correctly)"
echo "- Slightly slower (~10-20% overhead)"
echo ""
echo "Press 's' in any version to save a frame for comparison!"
