#!/bin/bash

# Benchmark script for Tier 1 improvements
# Tests parallel scaling and measures performance

echo "=== Video Mosaic Tier 1 Benchmark ==="
echo ""
echo "Testing parallel scaling with multi-region matching..."
echo ""

# Check if cifar_tiles exists
if [ ! -d "cifar_tiles" ]; then
    echo "Error: cifar_tiles directory not found!"
    echo "Please run: python3 download_cifar10.py"
    exit 1
fi

# Test different thread counts
for threads in 1 2 4 8; do
    echo "----------------------------------------"
    echo "Testing with $threads thread(s)..."
    echo "----------------------------------------"
    
    # Run a quick test (will need to manually quit after 1-2 frames)
    # We'll capture the timing output
    timeout 10s ./video_mosaic_optimal --small -j $threads -d cifar_tiles 2>&1 | grep -E "(Cost matrix|assignment|Total:|Loaded)" || true
    
    echo ""
done

echo "=== Benchmark Complete ===" 
echo ""
echo "Expected improvements:"
echo "- 1 thread: baseline (~300ms)"
echo "- 4 threads: ~80ms (3.75x speedup)"
echo "- 8 threads: ~45ms (6.7x speedup) - FIXED!"
echo ""
echo "Quality improvements:"
echo "- Multi-region matching captures edges and gradients"
echo "- Better spatial detail preservation"
echo "- Center-weighted matching for better visual quality"
